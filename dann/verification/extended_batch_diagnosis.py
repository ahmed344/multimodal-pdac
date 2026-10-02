"""Four-hour follow-up: 1,000-update composition/execution controls, no production changes."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from dann.verification import batch_execution_diagnosis as b


CONTROL_SPECS = {
    'P': ('G-small','X0','native'), 'S': ('F-native','X4','native'),
    'Qflat': ('Q','X0','small'), 'Qspatial': ('Q','X4','small'),
    'QX1': ('Q','X1','small'), 'QX2': ('Q','X2','small'), 'QX3': ('Q','X3','small'),
    'LocalSmall': ('F-small','X4','native'), 'Qdispersed': ('Q-dispersed','X0','small'),
    'Gspatial': ('G-small','X4','native'),
}


def proportional_plan(tiles: b.SpatialTileDataset, train: np.ndarray, seed: int,
                      passes: int, batch_size: int = 2048) -> b.Plan:
    """Interleave shuffled slide queues by normalized cumulative row midpoints."""
    if passes < 1 or batch_size < 1:
        raise ValueError('Passes and batch size must be positive.')
    queues: dict[str, list[int]] = {}
    for i, (key, _) in enumerate(tiles.tiles):
        queues.setdefault(key[0], []).append(i)
    lengths = np.array([len(selected) for _, selected in tiles.tiles])
    row_tile = {r:i for i,(_, selected) in enumerate(tiles.tiles) for r,_ in selected}
    streams, boundaries, pass_edges, tile_order, tile_edges = [], [0], [0], [], [0]
    for p in range(passes):
        rng = np.random.default_rng(seed + 50000 + p)
        ordered = []
        for slide in sorted(queues):
            q = rng.permutation(queues[slide])
            n = lengths[q]
            positions = (n.cumsum() - n/2) / n.sum()
            ordered.extend(zip(positions.tolist(), rng.random(len(q)).tolist(), q.tolist()))
        ordered.sort()
        stream = np.array([r for _,_,t in ordered for r,_ in tiles.tiles[t][1]], dtype=np.int64)
        streams.append(stream)
        for start in range(0,len(stream),batch_size):
            rows = stream[start:start+batch_size]
            tile_order.extend(dict.fromkeys(row_tile[int(r)] for r in rows))
            tile_edges.append(len(tile_order))
            boundaries.append(boundaries[-1]+len(rows))
        pass_edges.append(len(boundaries)-1)
    plan = b.Plan(np.concatenate(streams), np.array(boundaries), np.array(pass_edges),
                  np.array(tile_order), np.array(tile_edges))
    b.verify_plan(plan, train)
    return plan


def dispersed_plan(plan: b.Plan, train: np.ndarray, codes: np.ndarray, seed: int) -> b.Plan:
    """Randomize within slides, preserving exact batch slide labels and boundaries."""
    out = copy.deepcopy(plan)
    for p,(a,z) in enumerate(zip(plan.passes[:-1],plan.passes[1:])):
        start,stop = plan.boundaries[a],plan.boundaries[z]
        rows = out.rows[start:stop]
        rng = np.random.default_rng(seed+60000+p)
        for code in np.unique(codes[train]):
            positions = np.flatnonzero(codes[rows]==code)
            rows[positions] = rng.permutation(train[codes[train]==code])
    # Tile identities of dispersed rows differ; execution is explicitly core-only.
    out.tile_order = np.array([],dtype=np.int64)
    out.tile_boundaries = np.zeros(len(out.boundaries),dtype=np.int64)
    b.verify_plan(out,train)
    np.testing.assert_array_equal(codes[out.rows],codes[plan.rows])
    return out


def admit(deadline: b.Deadline, estimates: list[float], reserve: float = 600.) -> bool:
    """Admit only an entire group with 25 percent timing margin and final reserve."""
    if any(not math.isfinite(x) or x <= 0 for x in estimates):
        raise ValueError('Every runtime estimate must be finite and positive.')
    return deadline.remaining(reserve) >= 1.25*sum(estimates)


def progression(value: dict, baseline: dict) -> bool:
    """Apply the frozen three validation progression criteria."""
    r2 = value.get('positive_cd8_r2')
    return bool(value['cd8'] <= baseline['cd8']-.001 and value['biology'] < baseline['biology']
                and r2 is not None and r2 > 0)


def quality_audit(root: Path, config: dict, bundle: Any, tiles: b.SpatialTileDataset,
                  deadline: b.Deadline) -> np.ndarray:
    """Audit training-only densities and sparse MSI summaries without fitting preprocessing."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    deadline.start()
    if (root/'quality_scope.json').exists():
        return np.load(root/'spectral_summaries.npz')['active_peaks']
    train = bundle.split_indices['train']; m = bundle.metadata
    with h5py.File(config['data']['path'],'r') as f:
        x = f['X']; ptr = x['indptr'][:]; counts = np.diff(ptr)
        tic = np.zeros(len(counts),dtype=np.float64)
        for start in range(0,len(counts),8192):
            deadline.check(600)
            stop = min(start+8192,len(counts))
            local = ptr[start:stop+1]-ptr[start]
            values = x['data'][ptr[start]:ptr[stop]]
            sums = np.concatenate(([0.],np.cumsum(values,dtype=np.float64)))
            tic[start:stop] = sums[local[1:]]-sums[local[:-1]]
    np.savez_compressed(root/'spectral_summaries.npz',active_peaks=counts,total_intensity=tic)
    records = []
    for i,(key,selected) in enumerate(tiles.tiles):
        rows = np.array([r for r,_ in selected]); valid=m.target_valid_mask[rows,0]
        y=m.targets[rows,0]; positive=valid & (y>0)
        records.append(dict(tile=i,slide=str(key[0]),x=int(key[1]),y=int(key[2]),rows=len(rows),
            valid_cd8=int(valid.sum()),positive_cd8=int(positive.sum()),
            positive_fraction=float(positive.sum()/valid.sum()) if valid.any() else None,
            positive_mean=float(y[positive].mean()) if positive.any() else None,
            positive_q50=float(np.median(y[positive])) if positive.any() else None,
            positive_q90=float(np.quantile(y[positive],.9)) if positive.any() else None,
            mean_peaks=float(counts[rows].mean()),mean_tic=float(tic[rows].mean())))
    frame=pd.DataFrame(records); frame.to_csv(root/'training_tile_quality.csv',index=False)
    slides=[]
    for code,name in enumerate(m.batch_names):
        rows=train[m.batch_codes[train]==code]; valid=m.target_valid_mask[rows,0]
        y=m.targets[rows,0]; positive=valid&(y>0)
        slides.append(dict(slide=name,rows=len(rows),valid_cd8=int(valid.sum()),positive_cd8=int(positive.sum()),
            positive_fraction=float(positive.sum()/valid.sum()) if valid.any() else None,
            positive_quantiles=np.quantile(y[positive],[0,.25,.5,.75,.9,1]).tolist() if positive.any() else [],
            peak_quantiles=np.quantile(counts[rows],[0,.25,.5,.75,.9,1]).tolist(),
            tic_quantiles=np.quantile(tic[rows],[0,.25,.5,.75,.9,1]).tolist()))
    b.save_json(root/'training_slide_quality.json',slides)
    # Fixed ordinary regions plus independently flagged low/high signal and density extremes.
    choices=set(np.linspace(0,len(frame)-1,4).astype(int).tolist())
    for column in ('mean_tic','mean_peaks','positive_fraction'):
        choices.update(frame.nsmallest(2,column).tile.tolist()); choices.update(frame.nlargest(2,column).tile.tolist())
    map_dir=root/'quality_maps'; map_dir.mkdir(exist_ok=True)
    for t in sorted(choices):
        key,selected=tiles.tiles[t]; rows=np.array([r for r,_ in selected]); xy=np.array([tiles.keys[int(r)][1:] for r in rows])
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        values=[np.log1p(tic[rows]),counts[rows],np.where(m.target_valid_mask[rows,0],m.targets[rows,0],np.nan)]
        for ax,v,label in zip(axes,values,['log1p total MSI intensity','Active peaks','Normalized CD8 (training only)']):
            sc=ax.scatter(xy[:,0],xy[:,1],c=v,s=9); ax.set_title(label); ax.set_aspect('equal');fig.colorbar(sc,ax=ax)
        fig.suptitle(f'Slide {key[0]}, tile {key[1:]}');fig.tight_layout();fig.savefig(map_dir/f'tile_{t}.png',dpi=120);plt.close(fig)
    b.save_json(root/'quality_scope.json',dict(training_labels_only=True,excluded_rows=0,
        original_registration_images_used=False,alignment_verified=False,
        limitation='MSI/target coordinate maps cannot verify experimental registration or identify causality.',
        selected_tiles=sorted(choices)))
    return counts


def direct_source_check(root: Path, config: dict, bundle: Any, tiles: b.SpatialTileDataset) -> None:
    """Compare extracted cores with independent raw CSR slices and stored target metadata."""
    from dann.data_loader import _matrix_group_path
    if (root/'direct_source_checks.json').exists():
        return
    checked=[]
    with h5py.File(config['data']['path'],'r') as f:
        x=f[_matrix_group_path(config['data']['matrix_key'])]
        for t in np.linspace(0,len(tiles)-1,12).astype(int):
            tiled=b.tile_collate([tiles[int(t)]]); flat=b.core_flat(tiled)
            offset=0
            for j,row in enumerate(flat['row_ids'].tolist()):
                start,stop=x['indptr'][row:row+2]; indices=x['indices'][start:stop]; values=x['data'][start:stop].astype(np.float32)
                assert config['data']['intensity_transform']=='log1p' and not config['data']['nonzero_threshold']
                values=np.log1p(values)
                n=int(flat['peak_counts'][j]);assert n==len(values)
                np.testing.assert_array_equal(flat['peak_indices'][offset:offset+n].numpy(),indices)
                np.testing.assert_allclose(flat['intensities'][offset:offset+n].numpy(),values,rtol=1e-6,atol=0)
                np.testing.assert_array_equal(flat['targets'][j].numpy(),bundle.metadata.targets[row])
                np.testing.assert_array_equal(flat['target_valid_mask'][j].numpy(),bundle.metadata.target_valid_mask[row])
                assert int(flat['batches'][j])==int(bundle.metadata.batch_codes[row])
                offset+=n
            checked.extend(flat['row_ids'].tolist())
    b.save_json(root/'direct_source_checks.json',dict(passed=True,rows=checked,source=config['data']['path']))
    tiles.close()


def plan_summary(root: Path, name: str, plan: b.Plan, bundle: Any, counts: np.ndarray,
                 updates: int) -> None:
    """Persist batch composition and signal summaries without densifying spectra."""
    import pandas as pd
    m=bundle.metadata; records=[]
    for i in range(min(updates,len(plan.boundaries)-1)):
        rows=plan.batch(i); c=np.bincount(m.batch_codes[rows],minlength=len(m.batch_names));p=c/c.sum()
        valid=m.target_valid_mask[rows,0];positive=valid&(m.targets[rows,0]>0)
        records.append(dict(update=i+1,rows=len(rows),slides=int((c>0).sum()),
            slide_entropy=float(-(p[p>0]*np.log(p[p>0])).sum()),max_slide_fraction=float(p.max()),
            valid_cd8=int(valid.sum()),positive_cd8=int(positive.sum()),
            positive_cd8_fraction=float(positive.sum()/valid.sum()) if valid.any() else None,
            core_peaks=int(counts[rows].sum())))
    pd.DataFrame(records).to_csv(root/f'composition_{name}.csv',index=False)


class Study:
    """Sequential group admission and reproducible diagnostic execution."""
    def __init__(self, args: argparse.Namespace):
        self.args=args; self.root=args.output; self.updates=args.updates
        self.config=b.load_config(args.previous_run/'source_config.yaml')
        if self.root.exists() and not args.resume:
            raise FileExistsError('Output exists: use a new root or explicit --resume.')
        self.root.mkdir(parents=True,exist_ok=True)
        b.freeze(self.root,args.previous_run,self.config,b.SEEDS,args.budget_minutes)
        self.deadline=b.Deadline(self.root/'deadline.json',args.budget_minutes)
        payload=torch.load(args.previous_run/'reference.pt',map_location='cpu',weights_only=False)
        assert b.target_contract(self.config)==b.checkpoint_contract(payload)
        self.bundle=b.create_data_bundle(self.config,payload['split_indices'],payload['data_identity'])
        assert tuple(payload['batch_names'])==self.bundle.metadata.batch_names
        assert self.bundle.metadata.num_observations==1698428 and self.config['model']['num_peaks']==5808
        assert [len(self.bundle.split_indices[k]) for k in ('train','validation','test')]==[1358726,169824,169878]
        if not (self.root/'splits.npz').exists():np.savez(self.root/'splits.npz',**self.bundle.split_indices)
        b.save_json(self.root/'data_identity.json',payload['data_identity'])
        self.native=b.SpatialTileDataset(b.geometry_config(self.config),self.bundle.split_indices['train'],self.bundle.metadata)
        self.small=b.SpatialTileDataset(b.geometry_config(self.config,core_size=8),self.bundle.split_indices['train'],self.bundle.metadata,geometry=self.native.geometry)
        self.audit=b.prepare_audit(args.previous_run,self.native,self.bundle.split_indices['train']);self.native.close()
        if not (self.root/'audit_batch.pt').exists():torch.save(self.audit,self.root/'audit_batch.pt')
        self.baseline=json.loads((self.root/'constant_baseline.json').read_text())['validation']
        self.refs={};self.plans={};self.estimates={};self.decisions=[];self.tile_load_cache={}
        for seed in b.SEEDS:
            cfg=copy.deepcopy(self.config);cfg['training']['seed']=seed;b.seed_everything(seed,cfg['training']['deterministic_algorithms'])
            model=b.AdversarialLatentFusion.from_config(cfg,43,4)
            if seed==b.SEEDS[0]: model.load_state_dict(torch.load(args.previous_run/'A_spatial_mlp/initial_state.pt',weights_only=True))
            path=self.root/f'initial_{seed}.pt'
            if path.exists():model.load_state_dict(torch.load(path,weights_only=True))
            else:torch.save(model.state_dict(),path)
            self.refs[seed]=model
            passes=max(2,math.ceil(self.updates/min(math.ceil(len(self.native)/8),math.ceil(len(self.bundle.split_indices['train'])/2048))))
            plans=b.make_plans(self.native,self.bundle.split_indices['train'],self.bundle.metadata.batch_codes,seed,passes=passes)
            plans['Q']=proportional_plan(self.small,self.bundle.split_indices['train'],seed,passes)
            plans['Q-dispersed']=dispersed_plan(plans['Q'],self.bundle.split_indices['train'],self.bundle.metadata.batch_codes,seed)
            self.plans[seed]=plans
            for name,plan in plans.items():plan.save(self.root/f'plan_{seed}_{name}.npz')
        self.counts=None

    def record(self, name: str, **data: Any) -> None:
        """Write immutable decision/admission evidence."""
        path=self.root/f'{name}.json'
        if path.exists():return
        b.save_json(path,data)
        print(json.dumps(dict(event=name,**data)),flush=True)

    def preflight(self, seed: int, plan_name: str, mode: str, geometry: str,
                  dropout: str='unchanged') -> dict:
        """Preflight maximum measured peak workload, never substitute execution settings."""
        key=f'{seed}_{plan_name}_{mode}_{geometry}_{dropout}'
        path=self.root/f'workload_{key}.json'
        if path.exists():return json.loads(path.read_text())
        self.deadline.check(600)
        tiles=self.small if geometry=='small' else self.native;plan=self.plans[seed][plan_name]
        dataset=b.PlannedDataset(tiles,plan,mode in ('X3','X4'))
        loads=np.array([self.counts[plan.batch(i)].sum() for i in range(self.updates)])
        tile_loads={}
        if mode in ('X3','X4'):
            # Geometry/CSR-only enumeration: no dense spectra and no giant preflight batch allocation.
            for t,(key,_) in enumerate(tiles.tiles):
                if geometry in self.tile_load_cache:
                    tile_loads=self.tile_load_cache[geometry];break
                if t%100==0:self.deadline.check(600)
                slide,tx,ty=key; x0=tx*tiles.core_size-tiles.halo;y0=ty*tiles.core_size-tiles.halo
                ids=[tiles.lookup[(slide,x0+x,y0+y)] for y in range(tiles.size) for x in range(tiles.size)
                     if (slide,x0+x,y0+y) in tiles.lookup]
                tile_loads[t]=int(self.counts[ids].sum())
            self.tile_load_cache[geometry]=tile_loads
            loads=np.array([sum(tile_loads[t] for t in dict.fromkeys(dataset.row_tile[int(r)] for r in plan.batch(i)))
                            for i in range(self.updates)])
        index=int(loads.argmax());result=dict(seed=seed,plan=plan.sha256,mode=mode,geometry=geometry,
            dropout=dropout,max_peaks=int(loads[index]),mean_peaks=float(loads.mean()),batch=index)
        # Observed prior spatial throughput is ~10M peaks/s. This is a lower-bound admission gate,
        # not a promise of feasibility. Reserve enough time for a full three-seed comparison.
        optimistic=self.updates*float(loads.mean())/15_000_000+120
        if plan_name=='G-small' and mode=='X4' and not admit(self.deadline,[optimistic]*3,1800):
            result.update(passed=False,reason='Entire three-seed exact-context group exceeds optimistic remaining budget.',optimistic_seconds=optimistic)
            b.save_json(path,result);return result
        cfg=copy.deepcopy(self.config);cfg['training']['seed']=seed
        device=b.resolve_device(cfg['training']['device']); model=b.adapt(self.refs[seed],mode,dropout).to(device).train()
        opt=b.optimizer_for(model,cfg,.001);loss=b.ZILNLoss.from_config(cfg).to(device)
        batch=out=bio=total=None
        start=time.monotonic()
        try:
            batch=b.move_batch_to_device(dataset[index],device)
            torch.manual_seed(seed+1);torch.cuda.reset_peak_memory_stats(device)
            load_seconds=time.monotonic()-start
            times=[]
            for trial_index in range(3):
                self.deadline.check(600);opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
                out,bio,total=b.objective(model,batch,loss,cfg)
                total.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
                times.append(time.monotonic()-start)
                out=bio=total=None
            estimate_step=max(max(times[1:])*1.2, float(loads.mean())/8_000_000)
            result.update(passed=True,load_seconds=load_seconds,trial_seconds=times,gpu_peak_bytes=torch.cuda.max_memory_allocated(device),
                estimate_seconds=max(180,estimate_step*self.updates+250))
        except torch.cuda.OutOfMemoryError as exc:
            result.update(passed=False,reason='OOM; no execution substitution.',failure=str(exc))
        finally:
            del model,opt,loss,batch,out,bio,total;tiles.close();gc.collect();torch.cuda.empty_cache()
        b.save_json(path,result);return result

    def group(self, label: str, specs: list[tuple], reserve: float=600.) -> bool:
        """Preflight and admit a whole specified group before starting any new fit."""
        estimates=[]; pending=[]
        for spec in specs:
            name,seed,plan_name,mode,geometry,*rest=spec
            dropout=rest[0] if rest else 'unchanged'
            directory=self.root/f'{name}_{seed}'
            if (directory/'completion.json').exists():
                contract=json.loads((directory/'settings.json').read_text())
                expected=dict(seed=seed,mode=mode,dropout=dropout,plan=self.plans[seed][plan_name].sha256,updates=self.updates,implementation=b.file_hash(Path(b.__file__)))
                if any(contract[k]!=v for k,v in expected.items()):raise ValueError('Completed run contract mismatch.')
                b.completed_matches(directory,contract)
                continue
            if directory.exists():
                self.record(f'blocked_{label}',reason='Immutable incomplete attempt exists.',run=directory.name);return False
            trial=self.preflight(seed,plan_name,mode,geometry,dropout)
            if not trial['passed']:
                self.record(f'blocked_{label}',reason='A group preflight failed.',trial=trial);return False
            estimate=self.estimates.get((mode,geometry,dropout),trial['estimate_seconds'])
            estimates.append(estimate);pending.append(spec)
        accepted=admit(self.deadline,estimates,reserve) if estimates else True
        self.record(f'admission_{label}',accepted=accepted,estimates=estimates,margin=1.25,
                    available=self.deadline.remaining(reserve),reserve=reserve,specs=specs)
        if not accepted:return False
        for spec in pending:
            name,seed,plan_name,mode,geometry,*rest=spec;dropout=rest[0] if rest else 'unchanged'
            cfg=copy.deepcopy(self.config);cfg['training']['seed']=seed
            tiles=self.small if geometry=='small' else self.native;plan=self.plans[seed][plan_name]
            directory=self.root/f'{name}_{seed}'
            try:
                b.fit(directory,self.refs[seed],cfg,self.bundle,tiles,plan,mode,dropout,self.updates,
                      self.deadline,self.audit,resume=self.args.resume,reserve=reserve)
                seconds=json.loads((directory/'status.json').read_text())['seconds']
                self.estimates[(mode,geometry,dropout)]=seconds
            except (RuntimeError,TimeoutError) as exc:
                self.record(f'failed_{label}',run=directory.name,failure=str(exc));return False
            b.report(self.root)
        return True

    def passed(self, name: str) -> list[bool]:
        """Return per-seed progression, requiring complete fits."""
        result=[]
        for seed in b.SEEDS:
            p=self.root/f'{name}_{seed}'
            result.append((p/'completion.json').exists() and progression(json.loads((p/'history.json').read_text())[-1]['validation'],self.baseline))
        return result

    def prepare(self, seeds: list[int]) -> None:
        """Audit source identities, composition, equivalence and required memory."""
        self.deadline.start()
        self.counts=quality_audit(self.root,self.config,self.bundle,self.native,self.deadline)
        direct_source_check(self.root,self.config,self.bundle,self.native)
        for seed,plans in self.plans.items():
            for name in ('G-small','F-native','Q','Q-dispersed','F-small'):
                plan_summary(self.root,f'{seed}_{name}',plans[name],self.bundle,self.counts,self.updates)
        # Existing strict audits, limited to needed feasible monolithic batches.
        for seed in seeds:
            cfg=copy.deepcopy(self.config);cfg['training']['seed']=seed
            b.preflight(self.root,self.refs[seed],cfg,self.native,{'G-small':self.plans[seed]['G-small']},self.audit,self.deadline)

    def run_selected(self, controls: list[str], seeds: list[int]) -> None:
        """Reproduce only explicitly selected MLP controls from the frozen contract."""
        if len(set(controls))!=len(controls) or not set(controls).issubset(CONTROL_SPECS):
            raise ValueError('Use unique supported controls.')
        if len(set(seeds))!=len(seeds) or not set(seeds).issubset(b.SEEDS):
            raise ValueError('Use unique prescribed seeds.')
        self.prepare(seeds)
        specs=[(name,seed,*CONTROL_SPECS[name]) for seed in seeds for name in controls]
        self.group('selected_controls',specs,600)

    def run(self) -> None:
        """Run references, matched candidate, and evidence-selected complete groups."""
        self.prepare(b.SEEDS)
        refs=[(name,seed,plan,mode,'native') for seed in b.SEEDS for name,plan,mode in [('P','G-small','X0'),('S','F-native','X4')]]
        if not self.group('references',refs,1800):return
        matched=[(name,seed,'Q',mode,'small') for seed in b.SEEDS for name,mode in [('Qflat','X0'),('Qspatial','X4')]]
        if not self.group('matched_small_tiles',matched,1800):return
        flat,spatial,pixel=self.passed('Qflat'),self.passed('Qspatial'),self.passed('P')
        self.record('branch_decision',flat=flat,spatial=spatial,pixel=pixel)
        exact=[('Gspatial',seed,'G-small','X4','native') for seed in b.SEEDS]
        self.group('exact_global_context',exact,1800)
        decisive=None
        if all(flat) and not all(spatial):
            ladder=[(f'Q{mode}',seed,'Q',mode,'small') for mode in ('X1','X2','X3') for seed in b.SEEDS]
            if self.group('execution_ladder',ladder,1800):
                names=['Qflat','QX1','QX2','QX3','Qspatial'];modes=list(b.MODES)
                for left,right,ml,mr in zip(names[:-1],names[1:],modes[:-1],modes[1:]):
                    if all(self.passed(left)) and not any(self.passed(right)) or all(self.passed(right)) and not any(self.passed(left)):
                        decisive=[(left,b.SEEDS[0],'Q',ml,'small'),(right,b.SEEDS[0],'Q',mr,'small')]
                        off=[(f'{n}_off',seed,'Q',mode,'small','off') for seed in b.SEEDS for n,mode in [(left,ml),(right,mr)]]
                        self.group('separating_pair_dropout_off',off,1800);break
        elif not any(flat) and all(pixel):
            specs=[('Qdispersed',seed,'Q-dispersed','X0','small') for seed in b.SEEDS]
            if self.group('within_slide_dispersion',specs,1800) and all(self.passed('Qdispersed')):
                decisive=[('Qflat',b.SEEDS[0],'Q','X0','small'),('Qdispersed',b.SEEDS[0],'Q-dispersed','X0','small')]
        elif all(flat) and all(spatial):
            specs=[('LocalSmall',seed,'F-small','X4','native') for seed in b.SEEDS]
            self.group('local_small_spatial',specs,1800)
            decisive=[('S',b.SEEDS[0],'F-native','X4','native'),('Qspatial',b.SEEDS[0],'Q','X4','small')]
        else:
            # Mixed outcomes: complete the ladder rather than choosing a favorable seed.
            self.group('mixed_execution_ladder',[(f'Q{mode}',seed,'Q',mode,'small') for mode in ('X1','X2','X3') for seed in b.SEEDS],1800)
        if decisive is None:
            decisive=[('P',b.SEEDS[0],'G-small','X0','native'),('S',b.SEEDS[0],'F-native','X4','native')]
        repeats=[(f'{name}_repeat',seed,plan,mode,geometry) for name,seed,plan,mode,geometry in decisive]
        self.group('unchanged_repeat',repeats,600)
        finish_causal_comparisons(self)

def checkpoint_precision_audit(study: Study) -> None:
    """Audit full-size checkpoint precision and RNG without changing initial states."""
    root=study.root
    if (root/'full_batch_checkpoint_precision.json').exists():
        return
    seed=b.SEEDS[0];cfg=copy.deepcopy(study.config);cfg['training']['seed']=seed
    plan=study.plans[seed]['Q'];cpu=b.PlannedDataset(study.small,plan,False)[0]
    device=b.resolve_device(cfg['training']['device'])
    batch=b.move_batch_to_device(cpu,device);ref=copy.deepcopy(study.refs[seed]).to(device)
    strict=b.equivalence(ref,batch,batch,cfg,('X1','X2'),True)
    result={'rows':len(cpu['row_ids']),'peaks':len(cpu['peak_indices']),'plan_sha256':plan.sha256,
                                    'float64_strict':strict,'float32':{}}
    trials=[]
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        for mode in ['X1','X1','X2']:
            m=b.adapt(ref,mode).float().train();opt=b.optimizer_for(m,cfg,.001);loss=b.ZILNLoss.from_config(cfg).to(device)
            torch.manual_seed(seed+1);out,bio,total=b.objective(m,batch,loss,cfg);total.backward()
            grads={k:p.grad.detach().cpu().clone() for k,p in m.named_parameters() if p.grad is not None}
            rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
            norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True));opt.step()
            trials.append(dict(mode=mode,gradients=grads,rng=rng,norm=norm,
                            outputs={k:v.detach().cpu() for k,v in out.items()},state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}))
            del m,opt,loss,out,bio,total;torch.cuda.empty_cache()
    for name,trial in zip(['same_mode_repeat','checkpoint_toggle'],trials[1:]):
        base=trials[0];diff={}
        for category in ['outputs','gradients','state']:
            errors={k:float((base[category][k]-trial[category][k]).abs().max()) for k in base[category]}
            diff[category]={'max_abs':max(errors.values()),'per_parameter_or_output':errors}
        diff['rng_equal']=all(torch.equal(a,z) if a is not None else z is None for a,z in zip(base['rng'],trial['rng']))
        diff['gradient_norms']=[base['norm'],trial['norm']]
        result['float32'][name]=diff
    b.save_json(root/'full_batch_checkpoint_precision.json',result)
    study.small.close();del batch,ref,trials;torch.cuda.empty_cache()
    print(json.dumps({'checkpoint_full_batch_precision':{k:{c:v[c]['max_abs'] for c in ['outputs','gradients','state']} for k,v in result['float32'].items()},'float64_passed':strict['passed']}),flush=True)


def correction_supported(passed: list[bool], improvements: list[float],
                         sensitivities: list[float], repeat_consistent: bool) -> bool:
    """Apply the correction criteria to the tested spatial architecture only."""
    if not (len(passed)==len(improvements)==len(sensitivities)==3):
        return False
    if not np.isfinite(improvements).all() or not np.isfinite(sensitivities).all():
        return False
    return bool(all(passed) and repeat_consistent and np.median(improvements)>=.001
                and min(improvements)>=-.001 and min(sensitivities)>.001)

def finish_causal_comparisons(study: Study) -> None:
    """Complete supported composition contrasts, repeats, and admitted CNN pairs."""
    from dann.verification.discriminator_controls import make_control
    s=study; r=study.root
    s.deadline.check(600)
    checkpoint_precision_audit(s)
    if all(s.passed('Qspatial')):
        s.group('followup_local_small', [('LocalSmall',seed,'F-small','X4','native') for seed in b.SEEDS],1800)
    if all(s.passed('P')) and not all(s.passed('Qflat')):
        s.group('followup_within_slide_dispersion',[('Qdispersed',seed,'Q-dispersed','X0','small') for seed in b.SEEDS],1800)
    repeats=[('S_repeat',b.SEEDS[0],'F-native','X4','native'),('Qspatial_repeat',b.SEEDS[0],'Q','X4','small')]
    if all(s.passed('Qspatial')):
        s.group('followup_correction_repeat',repeats,1200)
    if all(s.passed('Qdispersed')):
        s.group('locality_repeat',[('Qflat_repeat',b.SEEDS[0],'Q','X0','small'),('Qdispersed_repeat',b.SEEDS[0],'Q-dispersed','X0','small')],1200)
    differences=[];sens=[];passed=[]
    for seed in b.SEEDS:
        old=r/f'S_{seed}';new=r/f'Qspatial_{seed}'
        a=json.loads((old/'history.json').read_text())[-1]['validation'];z=json.loads((new/'history.json').read_text())[-1]['validation']
        differences.append(a['cd8']-z['cd8']);passed.append(progression(z,s.baseline))
        sens.append(json.loads((new/'audits.json').read_text())[-1]['eval']['shuffled_mu_rms'])
    repeat_ok=True
    for name,seed,*_ in repeats:
        p=r/f'{name}_{seed}';o=r/f'{name.removesuffix("_repeat")}_{seed}'
        if not (p/'completion.json').exists():repeat_ok=False;continue
        a=json.loads((p/'history.json').read_text())[-1]['validation'];z=json.loads((o/'history.json').read_text())[-1]['validation']
        repeat_ok &= progression(a,s.baseline)==progression(z,s.baseline)
    eligible=correction_supported(passed,differences,sens,repeat_ok)
    s.record('followup_cnn_gate',eligible=eligible,spatial_passed=passed,paired_cd8_improvements=differences,shuffle_mu_rms=sens,unchanged_repeat_consistent=repeat_ok,
                                        criterion='Spatial correction criteria; mixed flat-control outcomes do not veto eligibility.')
    if eligible:
        # Admit the entire prescribed three-seed CNN comparison when feasible; otherwise
        # retain the originally planned one-seed screen. Never select seeds by outcomes.
        per_seed={seed:[json.loads((r/f'{name}_{seed}'/'status.json').read_text())['seconds']*1.2 for name in ['S','Qspatial']] for seed in b.SEEDS}
        cnn_seeds=b.SEEDS if admit(s.deadline,[x for seed in b.SEEDS for x in per_seed[seed]],600) else [b.SEEDS[0]]
        estimates=[x for seed in cnn_seeds for x in per_seed[seed]]
        accepted=admit(s.deadline,estimates,600)
        s.record('followup_cnn_admission',accepted=accepted,seeds=cnn_seeds,estimates=estimates,available=s.deadline.remaining(600),margin=1.25,
                                            reason='Complete all prescribed seeds if the entire group fits; otherwise use the planned first-seed screen. Selection precedes all CNN outcomes.')
        if accepted:
            cnn_refs={};cnn_configs={}
            # Audit both largest actual CNN batches for every admitted seed before any fit.
            for seed in cnn_seeds:
                cfg=copy.deepcopy(s.config);cfg['training']['seed']=seed;cfg['model']['heads']['discriminator']['type']='cnn'
                ref=make_control(s.refs[seed],cfg,cnn=True);cnn_refs[seed]=ref;cnn_configs[seed]=cfg
                for label,plan_name,geometry,tiles in [('original','F-native','native',s.native),('corrected','Q','small',s.small)]:
                    s.deadline.check(600)
                    t=json.loads((r/f'workload_{seed}_{plan_name}_X4_{geometry}_unchanged.json').read_text())
                    m=b.adapt(ref,'X4').cuda().train();opt=b.optimizer_for(m,cfg,.001);loss=b.ZILNLoss.from_config(cfg).cuda()
                    batch=b.move_batch_to_device(b.PlannedDataset(tiles,s.plans[seed][plan_name],True)[t['batch']],torch.device('cuda'))
                    weights=s.bundle.batch_class_weights.cuda() if cfg['loss']['balance_batch_classes'] else None
                    torch.cuda.reset_peak_memory_stats();out,bio,total=b.objective(m,batch,loss,cfg,.01334,True,weights)
                    total.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
                    s.record(f'followup_cnn_preflight_{label}_{seed}',passed=True,batch=t['batch'],gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    del m,opt,loss,batch,weights,out,bio,total;tiles.close();torch.cuda.empty_cache()
            for seed in cnn_seeds:
                for name,plan_name,tiles in [('CNNoriginal','F-native',s.native),('CNNcorrected','Q',s.small)]:
                    b.fit(r/f'{name}_{seed}',cnn_refs[seed],cnn_configs[seed],s.bundle,tiles,s.plans[seed][plan_name],'X4','unchanged',1000,
                                            s.deadline,s.audit,adversarial=True,resume=True,reserve=600)


def finish(root: Path, failure: str | None = None) -> None:
    """Save complete endpoint tables and untruncated per-seed diagnostic figures."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b.report(root)
    records=[]
    baseline=json.loads((root/'constant_baseline.json').read_text())['validation']
    for p in sorted(root.iterdir()):
        if not (p/'status.json').exists():continue
        status=json.loads((p/'status.json').read_text());hist=json.loads((p/'history.json').read_text()) if (p/'history.json').exists() else []
        settings=json.loads((p/'settings.json').read_text());audits=json.loads((p/'audits.json').read_text()) if (p/'audits.json').exists() else []
        for h in hist:
            if h['update'] not in (500,1000):continue
            a=next((x for x in audits if x['update']==h['update']),None)
            records.append(dict(run=p.name,seed=settings['seed'],status=status['status'],update=h['update'],rows=h['rows_seen'],
                cd8=h['validation']['cd8'],biology=h['validation']['biology'],r2=h['validation']['positive_cd8_r2'],
                passed=progression(h['validation'],baseline),shuffle_mu_rms=a['eval']['shuffled_mu_rms'] if a else None))
    pd.DataFrame(records).to_csv(root/'endpoints.csv',index=False)
    for seed in b.SEEDS:
        fig,axes=plt.subplots(2,3,figsize=(18,10))
        for p in sorted(root.iterdir()):
            if not p.name.endswith(str(seed)) or not (p/'history.json').exists():continue
            h=json.loads((p/'history.json').read_text());h=[x for x in h if x['update']>0]
            a=json.loads((p/'audits.json').read_text()) if (p/'audits.json').exists() else []
            u=json.loads((p/'training.json').read_text()) if (p/'training.json').exists() else []
            for ax,key in zip(axes[0,:2],('update','rows_seen')):ax.plot([x[key] for x in h],[x['validation']['cd8'] for x in h],label=p.name)
            axes[0,2].plot([x['update'] for x in a],[x['eval']['shuffled_mu_rms'] for x in a],label=p.name)
            if u:
                axes[1,0].plot([x['update'] for x in u],[x['slides'] for x in u],alpha=.6)
                axes[1,1].plot([x['update'] for x in u],[x.get('positive_cd8_fraction') for x in u],alpha=.6)
                axes[1,2].plot([x['update'] for x in u],[x['gradient_norm'] for x in u],alpha=.6)
        for ax,title in zip(axes.flat,['CD8 loss / updates','CD8 loss / processed rows','MSI shuffle: CD8 mu RMS','Distinct slides','Positive fraction among valid CD8','Gradient norm']):ax.set_title(title)
        for ax in axes[0,:2]:ax.axhline(baseline['cd8'],ls=':',color='black')
        axes[0,2].set_yscale('log');axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(root/f'curves_{seed}.png',dpi=140);plt.close(fig)
    b.save_json(root/'study_status.json',dict(finished=time.time(),failure=failure,completed_controls=sum((p/'completion.json').exists() for p in root.iterdir())))


def main(argv: list[str] | None = None) -> None:
    """Run the authorized extended study; never launch production training."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run',type=Path,default=b.PREVIOUS)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--budget-minutes',type=float,default=240)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--controls',nargs='+',choices=list(CONTROL_SPECS))
    parser.add_argument('--seeds',type=int,nargs='+',choices=b.SEEDS,default=b.SEEDS)
    args=parser.parse_args(argv)
    if len(set(args.seeds))!=len(args.seeds) or (args.controls and len(set(args.controls))!=len(args.controls)):
        parser.error('Seeds and controls must be unique.')
    if args.controls is None and args.seeds!=b.SEEDS:
        parser.error('A seed subset requires explicit --controls; the full study uses all prescribed seeds.')
    if args.updates!=1000 or not math.isfinite(args.budget_minutes) or not 0<args.budget_minutes<=240:
        parser.error('This study requires 1000 updates and a positive budget no larger than 240 minutes.')
    study=Study(args);failure=None
    try:
        if args.controls:study.run_selected(args.controls,args.seeds)
        else:study.run()
    except Exception:
        failure=traceback.format_exc();b.save_json(args.output/'study_failure.json',dict(traceback=failure));print(failure,flush=True)
    finally:
        finish(args.output,failure)
        for dataset in study.bundle.datasets.values():dataset.close()
        study.native.close();study.small.close()
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
