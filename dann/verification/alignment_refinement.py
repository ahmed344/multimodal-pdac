"""Restartable combined-alignment refinement, restricted to training/validation.

Prepare a new directory with --previous PRIOR --output NEW --prepare; then --run.
Every subprocess imports the frozen source snapshot. Historical artifacts remain
read-only. No stage invokes a post-training workflow, tissue inference or export.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from dann.analysis_execution import digest_file, fingerprint
from dann.batch_alignment import AlignmentEvaluationError, NAMES
from dann.checkpoints import checkpoint_config, validate_checkpoint
from dann.config import apply_smoke_overrides, load_config, resolve_execution
from dann.data_loader import create_data_bundle
from dann.train import train_model
from dann.training_artifacts import atomic_json
from dann.verification import alignment_sweep as sweep
from dann.verification.alignment_study_metrics import SEED, choose, eligibility
from dann.verification.dictionary_comparison import close, immutable_json
from dann.verification.pooling_diagnostics import json_clean

FULL_ARMS = dict(soft_nn30=(.1,.1,30.), anova0p3=(.1,.3,10.),
                 both_stronger=(.1,.3,30.), no_nhsic=(0.,.1,10.))
PILOTS = dict(nhsic10_retry=(10.,0.,0.), nhsic100_retry=(100.,0.,0.))
REUSED = ('reference', 'combined')
STAGES = ('smoke', 'pilot', 'pilot_evaluate', 'train', 'evaluate', 'full_umap')


def no_exports(config: dict) -> dict:
    """Disable every downstream workflow switch, including in smoke configs."""
    config['post_training'].update(enabled=False, run_analysis=False, run_inference=False,
        run_spatial_heatmaps=False, overwrite_inference=False)
    return config


def prepare(root: Path, previous: Path) -> None:
    """Reference verified historical checkpoints and freeze the matched protocol."""
    previous = previous.resolve()
    old = sweep.manifest_for(previous)
    recommendation = json.loads((previous/'decisions/recommendation.json').read_text())['selected']
    if not recommendation or recommendation['arm'] != 'combined':
        raise ValueError('Expected the previous combined recommendation.')
    records = {arm: sweep.records_for(previous, arm) for arm in REUSED}
    anchor = next(r for r in records['combined'] if r['id'] == recommendation['id'])
    if anchor != recommendation:
        raise ValueError('Combined recommendation disagrees with completed metrics.')
    source = yaml.safe_load((previous/'source_config.yaml').read_text())
    combined = sweep.config_for(previous, 'combined')
    if tuple(combined['loss']['batch_alignment'][n]['weight'] for n in NAMES) != (.1,.1,10.):
        raise ValueError('Unexpected combined objective weights.')
    expected = sweep.make_config(source, previous, 'combined', (.1,.1,10.))
    if combined != expected or source['training']['epochs'] != 300 or source['training']['seed'] != SEED:
        raise ValueError('Combined training configuration differs from the frozen study.')
    dependencies = {str(previous/'manifest.json'): digest_file(previous/'manifest.json')}
    # Freeze every artifact actually reused, including metrics, colors, histories,
    # panels, full UMAPs and completed-stage receipts. Never copy model checkpoints.
    for arm, values in records.items():
        index = previous/'diagnostics'/f'{arm}_checkpoints.json'
        dependencies[str(index)] = digest_file(index)
        for r in values:
            path = Path(r['checkpoint'])
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            sweep.validate_identity(checkpoint, old)
            if digest_file(path) != r['identity']['checkpoint_sha256']:
                raise ValueError('Historical metric/checkpoint mismatch.')
            dependencies[str(path)] = digest_file(path)
            out = previous/'diagnostics'/r['id']
            if not sweep.reusable(out/'completion.json', r['identity']):
                raise ValueError('Historical evaluation is incomplete.')
            for p in out.rglob('*'):
                if p.is_file() and (p.suffix not in ('.npz', '.npy') or p.name in ('panel.npz', 'umap.npz')):
                    dependencies[str(p)] = digest_file(p)
        directory = Path(values[0]['checkpoint']).parent
        if directory.name == 'archives':
            directory = directory.parent
        for p in directory.glob('*'):
            if p.is_file() and p.suffix in ('.csv', '.png', '.yaml', '.json'):
                dependencies[str(p)] = digest_file(p)
    for relative in ('decisions/recommendation.json', 'configs/combined.yaml'):
        dependencies[str(previous/relative)] = digest_file(previous/relative)
    root.mkdir(parents=True, exist_ok=False)
    for child in ('configs','arms','stages','logs','diagnostics','decisions','source'):
        (root/child).mkdir()
    frozen = ['splits.npz','probe_train_rows.npy','panel_rows.npy','samples.npz','validation_rows.csv']
    for name in frozen:
        shutil.copyfile(previous/name, root/name)
        dependencies[str(previous/name)] = digest_file(previous/name)
    source = no_exports(source)
    (root/'source_config.yaml').write_text(yaml.safe_dump(source, sort_keys=False))
    frozen.append('source_config.yaml')
    sources = {}
    repository = Path(__file__).resolve().parents[2]
    for path in sorted((repository/'dann').rglob('*.py')):
        relative = path.relative_to(repository)
        dest = root/'source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        sources[str(relative)] = digest_file(path)
    m = copy.deepcopy(old)
    m.update(version=2, created_utc=datetime.now(timezone.utc).isoformat(),
        previous=str(previous), dependencies=dependencies, anchor=anchor,
        sources=sources, files={n:digest_file(root/n) for n in frozen},
        refinement=dict(full_arms=FULL_ARMS, pilots=PILOTS, pause_after_epoch=5,
            inference=False, spatial_exports=False, prediction_arrays='memory only',
            continuation='epoch-five numerical/biological eligibility and valid positive-weight objective; mixing improvement not required',
            selection='original nHSIC 0.1 eligibility; improve both mixing excesses over existing combined'))
    immutable_json(root/'manifest.json', json_clean(m))
    for arm, weights in {**FULL_ARMS, **PILOTS}.items():
        sweep.register_arm(root, arm, weights)
    for arm, values in records.items():
        atomic_json(root/'diagnostics'/f'{arm}_checkpoints.json', values)
    print(f'Prepared refinement: {root}', flush=True)


def manifest_for(root: Path) -> dict:
    """Verify frozen inputs/configurations and every historical dependency."""
    m = sweep.manifest_for(root)
    for path, digest in m['dependencies'].items():
        if digest_file(Path(path)) != digest:
            raise ValueError(f'Historical dependency changed: {path}')
    return m


def config_for(root: Path, arm: str) -> dict:
    """Resolve reused arms without assigning any new outputs to old directories."""
    if arm in REUSED:
        weights = (.1,0.,0.) if arm == 'reference' else (.1,.1,10.)
        return no_exports(sweep.make_config(yaml.safe_load((root/'source_config.yaml').read_text()), root, arm, weights))
    return no_exports(load_config(root/'configs'/f'{arm}.yaml'))


def bounded_smoke(root: Path, c: dict) -> dict:
    """Bounded real-data fitting plus validation; no downstream workflow calls."""
    c = copy.deepcopy(c)
    sweep.smoke_inputs(root, c)
    for key in ('path', 'context_path'):
        c['data'][key] = str(root/'smoke_data'/f'{key}.h5ad')
    c = no_exports(apply_smoke_overrides(c))
    c['training'].update(archive_checkpoint_every=1, resume_checkpoint=None)
    c['analysis']['figure_dpi'] = 50
    c['performance']['async_artifacts'] = False
    path = train_model(c)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(checkpoint, c)
    directory = path.parent
    archive = directory/'archives/epoch_0001.pt'
    if not archive.exists() or not checkpoint.get('training_rng_state'):
        raise ValueError('Smoke lacks restartable checkpoint state.')
    history = pd.read_csv(directory/'history.csv')
    for name in ('train_total_loss','validation_total_loss','validation_supervised_objective'):
        if not np.isfinite(history[name]).all():
            raise ValueError(f'Nonfinite smoke {name}.')
    if len(checkpoint['split_indices']['train']) > c['training']['smoke_train_samples']:
        raise ValueError('Smoke exceeded its training bound.')
    return dict(checkpoint=str(path), validation_rows=len(checkpoint['split_indices']['validation']),
                files=[str(path),str(archive),str(directory/'history.csv')], inference=False, spatial_exports=False)


def checkpoint_paths(root: Path, arm: str) -> list[Path]:
    """Enumerate completed epochs only; failure snapshots never enter selection."""
    directory = Path(config_for(root,arm)['training']['output_dir'])
    # Archives provide stable epoch-five provenance across pilot continuation.
    paths = [*sorted((directory/'archives').glob('epoch_*.pt')), directory/'best.pt', directory/'latest.pt']
    seen, result = {}, []
    for path in paths:
        if not path.exists():
            continue
        ck = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint_config(ck)  # Reject an explicitly supplied diagnostic snapshot.
        epoch = ck['epoch'] + 1
        digest = fingerprint({k:fingerprint(v.numpy()) for k,v in ck['model_state'].items()})
        if epoch in seen and seen[epoch] != digest:
            raise ValueError('Completed checkpoints disagree at the same epoch.')
        if epoch not in seen:
            result.append(path)
        seen[epoch] = digest
    return sorted(result, key=lambda p: torch.load(p,map_location='cpu',weights_only=False)['epoch'])


def validate_boundary(root: Path, arm: str, m: dict, *, pilot: bool) -> dict:
    """Check exact identity, epoch coverage, restart state and configured stopping."""
    c = config_for(root,arm)
    directory = Path(c['training']['output_dir'])
    ck = torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
    epoch = ck['epoch']+1
    stopped = ck['early_stopping_state']['stale_epochs'] >= c['training']['early_stopping_patience']
    if not ((pilot and epoch == 5) or stopped or epoch == c['training']['epochs']):
        raise ValueError('Fit has not reached its required completed boundary.')
    for path in checkpoint_paths(root,arm):
        saved = torch.load(path,map_location='cpu',weights_only=False)
        validate_checkpoint(saved,c)
        sweep.validate_identity(saved,m)
        if not saved.get('training_rng_state') or saved['training_progress']['update_in_epoch']:
            raise ValueError('Checkpoint lacks exact completed-epoch restart state.')
        if not all(torch.isfinite(v).all() for v in saved['model_state'].values()):
            raise FloatingPointError('Nonfinite checkpoint weights.')
    archives = {int(p.stem.split('_')[-1]) for p in (directory/'archives').glob('epoch_*.pt')}
    if archives != set(range(5,epoch+1,5)):
        raise ValueError('Archive coverage differs from five-epoch cadence.')
    for e in range(epoch):
        path = directory/'sampling'/f'epoch_{e:04d}_summary.json'
        sample = json.loads(path.read_text())
        if not sample['complete'] or sample['processed_rows'] != m['splits']['train']['rows']:
            raise ValueError('Training did not cover every frozen row.')
        old = Path(m['baseline']['checkpoint']).parent/'sampling'/path.name
        if old.exists():
            historical = json.loads(old.read_text())
            if any(sample[k] != historical[k] for k in ('row_order_sha256','boundaries_sha256')):
                raise ValueError('Training sample order changed.')
    best = torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    return dict(stopped_epoch=epoch, supervised_selected_epoch=best['epoch']+1,
        files=[str(p) for p in checkpoint_paths(root,arm) if not pilot or p.parent.name == 'archives'])


def pilot_gate(candidate: dict, reference: dict, config: dict) -> dict:
    """Apply original eligibility, without using early mixing as a rejection rule."""
    check = eligibility(candidate, reference)
    objectives = []
    for name in NAMES:
        if config['loss']['batch_alignment'][name]['weight'] <= 0:
            continue
        for split in ('train','validation'):
            row = next(iter(candidate.get('training_history', [])), {})
            score = row.get(f'{split}_alignment_{name}')
            count = row.get(f'{split}_alignment_{name}_valid_count', 0)
            objectives.append(dict(name=name, split=split, score=score, valid_count=count,
                passed=bool(score is not None and np.isfinite(score) and count > 0)))
    return dict(extend=bool(candidate['epoch'] == 5 and check['eligible'] and all(v['passed'] for v in objectives)),
        epoch=candidate['epoch'], eligibility=check, positive_objectives=objectives,
        mixing_used_for_gate=False, checkpoint=candidate['checkpoint'])


def execute_stage(root: Path, arm: str, stage: str) -> None:
    """Run a single restartable stage under the frozen source/configuration contract."""
    if stage not in STAGES or arm not in {**FULL_ARMS, **PILOTS}:
        raise ValueError('Unknown refinement arm or stage.')
    m = manifest_for(root)
    c = config_for(root,arm)
    identity = dict(manifest_sha256=digest_file(root/'manifest.json'), arm=arm, stage=stage, config=fingerprint(c))
    if stage in ('evaluate','pilot_evaluate'):
        paths = checkpoint_paths(root,arm)
        if stage == 'pilot_evaluate':
            paths = [p for p in paths if p.name == 'epoch_0005.pt']
            if len(paths) != 1:
                raise ValueError('Pilot requires an archived completed epoch five.')
        identity['checkpoints'] = {str(p):digest_file(p) for p in paths}
    if stage == 'full_umap':
        selected = json.loads((root/'decisions/selected_epochs.json').read_text())[arm]
        identity['checkpoint'] = digest_file(Path(selected['checkpoint']))
    stamp = root/'stages'/f'{arm}-{stage}.json'
    if sweep.reusable(stamp,identity):
        print(f'Reusing {arm} {stage}',flush=True)
        return
    start = time.monotonic()
    atomic_json(stamp,dict(status='running',identity=identity))
    try:
        if stage == 'smoke':
            result = bounded_smoke(root,c)
        elif stage in ('train','pilot'):
            if stage == 'train' and arm in PILOTS:
                gate = json.loads((root/'decisions'/f'{arm}_pilot.json').read_text())
                if not gate['extend']:
                    raise ValueError('Pilot continuation was not eligible.')
            latest = Path(c['training']['output_dir'])/'latest.pt'
            if latest.exists():
                saved = torch.load(latest,map_location='cpu',weights_only=False)
                sweep.validate_identity(saved,m)
                validate_checkpoint(saved,c)
                if not saved.get('training_rng_state'):
                    raise ValueError('Interrupted run lacks exact RNG state.')
                # Resume cannot silently retain different scientific loss settings.
                saved_c = checkpoint_config(saved)
                if saved_c['loss'] != c['loss']:
                    raise ValueError('Resume objective configuration mismatch.')
                c['training']['resume_checkpoint'] = str(latest)
            train_model(c,observer=sweep.training_observer(root,arm,m),stop_after_epoch=5 if stage=='pilot' else None)
            result = validate_boundary(root,arm,m,pilot=stage=='pilot')
        elif stage in ('evaluate','pilot_evaluate'):
            saved = torch.load(paths[0],map_location='cpu',weights_only=False)
            bundle = create_data_bundle(c,saved['split_indices'],saved['data_identity'],requested_splits=('validation',))
            best = torch.load(Path(c['training']['output_dir'])/'best.pt',map_location='cpu',weights_only=False)['epoch']+1
            try:
                records = []
                for path in paths:
                    record = sweep.evaluate_checkpoint(root,arm,path,m,c,bundle)
                    record['supervised_selected'] = record['epoch'] == best
                    records.append(record)
            finally:
                close(bundle)
            destination = root/'diagnostics'/f'{arm}_{"pilot" if stage=="pilot_evaluate" else "checkpoints"}.json'
            atomic_json(destination,records)
            result = dict(files=[str(destination),*[str(root/'diagnostics'/r['id']/'completion.json') for r in records]])
        elif stage == 'full_umap':
            path = Path(selected['checkpoint'])
            checkpoint = torch.load(path,map_location='cpu',weights_only=False)
            from dann.model import AdversarialLatentFusion
            model = AdversarialLatentFusion.from_config(checkpoint_config(checkpoint),len(m['batch_names']),4).cuda().eval()
            model.load_state_dict(checkpoint['model_state'])
            bundle = create_data_bundle(c,checkpoint['split_indices'],checkpoint['data_identity'],requested_splits=('validation',))
            out = root/'diagnostics'/selected['id']/'full_validation'
            out.mkdir(exist_ok=True)
            try:
                data,_ = sweep.extract_rows(model,c,bundle,checkpoint['split_indices']['validation'])
                coords = sweep.fit_umap(data['latent'])
                if not np.isfinite(coords).all():
                    raise FloatingPointError('Nonfinite full-validation UMAP.')
                np.savez_compressed(out/'umap.npz',rows=data['rows'],umap=coords)
                sweep.plot_umap(coords,data['rows'],data,bundle.metadata,m,out/'umap.png',selected['id']+' full validation')
                atomic_json(out/'provenance.json',dict(checkpoint_sha256=digest_file(path),rows=len(data['rows']),
                    rows_sha256=fingerprint(data['rows']),settings=m['protocol']['umap']))
            finally:
                close(bundle)
            result = dict(files=[str(p) for p in out.iterdir() if p.is_file()])
        sweep.complete(stamp,identity,result,start)
    except (FloatingPointError, AlignmentEvaluationError) as error:
        if stage not in ('train','pilot'):
            raise
        directory = Path(c['training']['output_dir'])
        exclusion = root/'diagnostics'/f'{arm}_exclusion.json'
        atomic_json(exclusion,dict(arm=arm,reason=str(error),traceback=traceback.format_exc(),
            failure_snapshots=[str(p) for p in (directory/'failures').rglob('snapshot.pt')],
            resume_checkpoint=str(directory/'latest.pt') if (directory/'latest.pt').exists() else None,
            selectable=False))
        sweep.complete(stamp,identity,dict(excluded=True,files=[str(exclusion)]),start)
    except BaseException as error:
        atomic_json(stamp,dict(status='failed',identity=identity,error=repr(error),traceback=traceback.format_exc()))
        raise


def run_subprocess(root: Path, arm: str, stage: str) -> None:
    """Run exactly one GPU stage at a time using the frozen Python sources."""
    with (root/'logs'/f'{arm}-{stage}.log').open('a') as log:
        print(f'Starting {arm} {stage}',flush=True)
        subprocess.run([sys.executable,'-m','dann.verification.alignment_refinement','--output',str(root),
                        '--arm',arm,'--stage',stage],cwd=root/'source',check=True,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
                                 NUMBA_NUM_THREADS='4',MPLBACKEND='Agg',PYTHONUNBUFFERED='1'))


def selected_epochs(root: Path, reference: dict, anchor: dict) -> tuple[dict, dict]:
    """Keep supervised and mixing selections distinct; only full fits can replace."""
    selected = {'reference':reference, 'combined':anchor}
    candidates = []
    for arm in (*FULL_ARMS,*PILOTS):
        records = sweep.records_for(root,arm)
        if not records:
            continue
        completed = root/'stages'/f'{arm}-train.json'
        full = completed.exists() and json.loads(completed.read_text()).get('status') == 'complete'
        if full and not (root/'diagnostics'/f'{arm}_exclusion.json').exists():
            candidates.extend(records)
        selected[arm] = choose(records,reference,base=anchor) or next((r for r in records if r['supervised_selected']),records[-1])
    replacement = choose(candidates,reference,base=anchor)
    return selected, dict(selected=replacement or anchor, replaced_combined=replacement is not None,
        reason='Eligible improvement over combined in both spaces' if replacement else 'Retained existing combined; no eligible replacement improves both spaces',
        supervised_checkpoints_preserved=True)


def report(root: Path, *, finished: bool = False) -> None:
    """Link compact validation comparisons, common epochs, curves and full UMAPs."""
    m = manifest_for(root)
    previous = Path(m['previous'])
    records = [r for a in (*REUSED,*FULL_ARMS,*PILOTS) for r in sweep.records_for(root,a)]
    reference = sweep.records_for(root,'reference')[0]
    primary, targets, probes, checks, histories, mix, gradients = [],[],[],[],[],[],[]
    per_slide, stratified, spectra = [], [], []
    selected_path = root/'decisions/selected_epochs.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    panels, curves = [], []
    for arm in (*REUSED,*FULL_ARMS,*PILOTS):
        values = [r for r in records if r['arm']==arm]
        if not values:
            continue
        directory = Path(values[-1]['checkpoint']).parent
        if directory.name=='archives':
            directory=directory.parent
        history = pd.read_csv(directory/'history.csv')
        histories.extend(dict(arm=arm,**r) for r in history.to_dict('records'))
        for name in ('loss_curves.png','ihc_curves.png'):
            path = directory/name
            if path.exists():
                curves.append(f'<a href="{os.path.relpath(path,root)}">{arm} {name}</a>')
        if arm in selected:
            r = selected[arm]
            out = (previous if arm in REUSED else root)/'diagnostics'/r['id']
            relative=os.path.relpath(out,root)
            panels.append(f'<h3>{r["id"]}</h3><img src="{relative}/panel.png" width="100%">'
                f'<p><a href="{relative}/full_validation/umap.png">Full-validation UMAP</a> · '
                f'<a href="{relative}/full_validation/umap.npz">UMAP coordinates</a> · '
                f'<a href="{relative}/metrics.json">Metrics and spectra</a></p>')
    for r in records:
        check=eligibility(r,reference)
        row=dict(id=r['id'],arm=r['arm'],epoch=r['epoch'],supervised_selected=r['supervised_selected'],
            umap_selected=selected.get(r['arm'],{}).get('id')==r['id'],
            mixing_selected=(r['id']==m['anchor']['id'] if r['arm']=='combined' else
                (choose([v for v in records if v['arm']==r['arm']],reference,base=m['anchor']) or {}).get('id')==r['id']),
            eligible=check['eligible'],
            **{f'{s}_{k}':v for s,values in r['mixing'].items() for k,v in values.items()},
            cd8_loss=r['distributions'].get('cd8',{}).get('loss'),**{f'he_{k}':v for k,v in r['he'].items()})
        for space in ('raw','normalized'):
            spectrum=r['spectrum'].get(space,{})
            eig=spectrum.get('eigenvalues',[])
            row[space+'_participation_ratio']=spectrum.get('participation_ratio')
            row[space+'_leading_fraction']=eig[0]/sum(eig) if eig and sum(eig)>0 else None
            anchor_eig=m['anchor']['spectrum'][space]['eigenvalues']
            baseline_fraction=anchor_eig[0]/sum(anchor_eig)
            leading=row[space+'_leading_fraction']
            row[space+'_leading_fraction_delta_vs_combined']=leading-baseline_fraction if leading is not None else None
            row[space+'_concentration_increased']=leading>baseline_fraction if leading is not None else None
            spectra.extend(dict(id=r['id'],space=space,component=i+1,eigenvalue=v,
                fraction=v/sum(eig) if sum(eig)>0 else None) for i,v in enumerate(eig))
        primary.append(row)
        checks.extend(dict(id=r['id'],**v) for v in check['checks'])
        history=next(iter(r['training_history']),{})
        for name,values in r['distributions'].items():
            title=name.capitalize() if name!='cd8' else 'CD8'
            targets.append(dict(id=r['id'],target=name,loss=values['loss'],presence_brier=values['presence']['brier'],
                positive_pit_ks=values.get('positive_pit_ks'),
                **{k:v for k,v in history.items() if k.startswith('validation_') and title.lower() in k.lower()},
                **{f'coverage_{level}':v['coverage'] for level,v in values['intervals'].items()},
                **{f'exceedance_{t}_brier':v['brier'] for t,v in values.get('exceedance',{}).items()}))
        probes.extend(dict(id=r['id'],probe=n,**v) for n,v in r['probes'].items())
        stratified.extend(dict(id=r['id'],stratum=q,probe=n,**v) for q,values in r.get('stratified_probes',{}).items() for n,v in values.items())
        gradients.append(dict(id=r['id'],**{k:v for k,v in history.items() if any(w in k for w in ('gradient','alignment'))}))
        out=(previous if r['arm'] in REUSED else root)/'diagnostics'/r['id']
        if (out/'mixing_per_slide.csv').exists():
            per_slide.extend(dict(id=r['id'],**v) for v in pd.read_csv(out/'mixing_per_slide.csv').to_dict('records'))
        if (out/'mixing_samples.csv').exists():
            mix.extend(dict(id=r['id'],**v) for v in pd.read_csv(out/'mixing_samples.csv').to_dict('records'))
    frame=pd.DataFrame(primary)
    common=frame[frame.epoch.isin(frame.groupby('epoch').arm.nunique().loc[lambda s:s>1].index)] if len(frame) else frame
    tables=dict(checkpoint_comparison=frame,common_epochs=common,validation_targets=pd.DataFrame(targets),
        biological_probes=pd.DataFrame(probes),eligibility_limits=pd.DataFrame(checks),
        training_history=pd.DataFrame(histories),mixing_samples=pd.DataFrame(mix),gradient_alignment=pd.DataFrame(gradients),
        mixing_per_slide=pd.DataFrame(per_slide),stratified_probes=pd.DataFrame(stratified),variance_spectra=pd.DataFrame(spectra))
    for name,table in tables.items():
        table.to_csv(root/f'{name}.csv',index=False)
    _plot_comparison(root,tables['training_history'],frame)
    decision_path=root/'decisions/recommendation.json'
    recommendation=json.loads(decision_path.read_text()) if decision_path.exists() else None
    intro=recommendation['reason'] if recommendation else 'Refinement in progress; existing combined remains the recommendation.'
    gates={a:json.loads((root/'decisions'/f'{a}_pilot.json').read_text()) for a in PILOTS if (root/'decisions'/f'{a}_pilot.json').exists()}
    exclusions={a:json.loads((root/'diagnostics'/f'{a}_exclusion.json').read_text()) for a in (*FULL_ARMS,*PILOTS) if (root/'diagnostics'/f'{a}_exclusion.json').exists()}
    links=' · '.join(f'<a href="{n}.csv">{n.replace("_"," ")}</a>' for n in tables)
    concentration=[]
    if recommendation:
        chosen=recommendation['selected']
        for space in ('raw','normalized'):
            current=chosen['spectrum'][space]; old=m['anchor']['spectrum'][space]
            a=current['eigenvalues'][0]/sum(current['eigenvalues']); b=old['eigenvalues'][0]/sum(old['eigenvalues'])
            concentration.append(f'{space}: leading component {a:.2%} (existing combined {b:.2%}); participation ratio {current["participation_ratio"]:.4f} (combined {old["participation_ratio"]:.4f}). '+('Variance concentration increased.' if a>b else 'Variance concentration did not increase.'))
    body=f'''<!doctype html><meta charset="utf-8"><title>Combined alignment refinement</title>
<style>body{{font:15px system-ui;margin:30px;max-width:1700px}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:5px;border:1px solid #ddd}}.scroll{{overflow:auto}}img{{max-width:1600px}}</style>
<h1>Combined alignment refinement</h1><p>{html.escape(intro)} Complete: {finished}.</p>
<p>Training and validation only. No tissue inference or spatial prediction exports. Historical checkpoints and artifacts are referenced unchanged.
All arms use seed {SEED}, the same 90/10 split, initialization, optimizer, sampler, and supervised weights. Full fits retain the 300-epoch maximum and supervised stopping; pilots pause after epoch five.</p>
<p><a href="manifest.json">Frozen protocol and source identities</a> · <a href="decisions/recommendation.json">Recommendation</a> · {links}</p>
<h2>Checkpoint comparison</h2><div class="scroll">{frame.to_html(index=False,float_format=lambda x:f'{x:.5g}')}</div>
<p>Eligibility uses all original biological/CD8 limits against nHSIC 0.1. A replacement must improve both excess measures over combined epoch {m['anchor']['epoch']}.
Latent excess ties within 0.01 use UMAP excess and then CD8 loss. Exact kNN excludes self and subtracts leave-one-out chance.
Five fixed balanced samples and within-stromal-quartile neighbors are reused. Single training seed; diagnostic draws are not training replications. Validation selection is not an unbiased test estimate.</p>
<h2>Variance concentration</h2><p>{' '.join(concentration)} Participation ratio remains a diagnostic, with no new cutoff.
If concentration limits further mixing/biological progress, variance/covariance regularization is the next separate experiment.</p>
<h2>Training and validation curves</h2><img src="comparison_curves.png" width="100%"><p>{' · '.join(curves)}</p>
<h2>High-nHSIC investigation</h2><p>Previous runs failed on unavailable zero-weight soft-NN diagnostics. These now have an unavailable score, zero contribution, and explicit invalid counts/reasons.
Continuation requires epoch-five biological/numerical eligibility and valid positive-weight objectives. Poor early mixing alone does not reject a pilot.</p><pre>{html.escape(json.dumps(json_clean(gates),indent=2))}</pre>
<h2>Numerical failures</h2><pre>{html.escape(json.dumps(exclusions,indent=2))}</pre>
<h2>Selected checkpoint UMAPs</h2>{''.join(panels)}
<p>Full-validation and panel UMAPs use identical rows, slide colors and settings. Predictions and latents remain in memory during new evaluations.
Per-target losses, R², balanced accuracy, calibration, biological and slide probes, gradients/conflicts and raw/normalized spectra are in the linked tables and metrics.</p>'''
    (root/'report.html').write_text(body)
    atomic_json(root/'status.json',dict(complete=finished,recommendation=recommendation,pilot_decisions=gates,
        exclusions=exclusions,updated_utc=datetime.now(timezone.utc).isoformat()))


def _plot_comparison(root: Path, history: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Plot full training histories and matched-checkpoint mixing/concentration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    for ax,key in zip(axes.flat[:3],('validation_supervised_objective','validation_cd8_loss','validation_he_mse')):
        for arm,group in history.groupby('arm'):
            if key in group:
                ax.plot(group.epoch,group[key],label=arm)
        ax.set(title=key,xlabel='Epoch')
    for ax,key in zip(axes.flat[3:],('latent_excess','umap_excess','raw_leading_fraction')):
        for arm,group in frame.groupby('arm'):
            if key in group:
                ax.plot(group.epoch,group[key],'.-',label=arm)
        ax.set(title=key,xlabel='Epoch')
    axes.flat[0].legend(fontsize=8)
    fig.savefig(root/'comparison_curves.png',dpi=140)
    plt.close(fig)


def run_all(root: Path) -> None:
    """Sequential GPU experiments, gated pilot continuation, then final UMAPs."""
    m=manifest_for(root)
    with (root/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another refinement runner owns this directory.') from error
        reference=sweep.records_for(root,'reference')[0]
        if not eligibility(reference,reference)['eligible']:
            raise ValueError('Reference cannot establish eligibility limits.')
        run_subprocess(root,'both_stronger','smoke')
        report(root)
        for arm in FULL_ARMS:
            run_subprocess(root,arm,'train')
            if not (root/'diagnostics'/f'{arm}_exclusion.json').exists():
                run_subprocess(root,arm,'evaluate')
            report(root)
        for arm in PILOTS:
            run_subprocess(root,arm,'pilot')
            if (root/'diagnostics'/f'{arm}_exclusion.json').exists():
                continue
            run_subprocess(root,arm,'pilot_evaluate')
            candidate=json.loads((root/'diagnostics'/f'{arm}_pilot.json').read_text())[0]
            gate=pilot_gate(candidate,reference,config_for(root,arm))
            sweep.decision(root,f'{arm}_pilot',json_clean(gate))
            if gate['extend']:
                run_subprocess(root,arm,'train')
            if not (root/'diagnostics'/f'{arm}_exclusion.json').exists():
                run_subprocess(root,arm,'evaluate')
            report(root)
        selections,recommendation=selected_epochs(root,reference,m['anchor'])
        sweep.decision(root,'selected_epochs',selections)
        sweep.decision(root,'recommendation',recommendation)
        report(root)
        for arm in selections:
            if arm not in REUSED:
                run_subprocess(root,arm,'full_umap')
        report(root,finished=True)


def main() -> None:
    """Dispatch only the explicit training/validation stage allowlist."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--arm',choices=(*FULL_ARMS,*PILOTS))
    parser.add_argument('--stage',choices=STAGES)
    args=parser.parse_args()
    root=args.output.resolve()
    torch.set_num_threads(4)
    if args.prepare:
        if args.previous is None:
            parser.error('--prepare requires --previous')
        prepare(root,args.previous)
    if args.run or args.stage or args.report:
        manifest_for(root)
        snapshot=root/'source/dann/verification/alignment_refinement.py'
        if Path(__file__).resolve()!=snapshot:
            command=[sys.executable,'-m','dann.verification.alignment_refinement','--output',str(root)]
            if args.run: command.append('--run')
            if args.report: command.append('--report')
            if args.stage: command += ['--arm',args.arm,'--stage',args.stage]
            subprocess.run(command,cwd=root/'source',check=True)
            return
    if args.stage:
        if not args.arm: parser.error('--stage requires --arm')
        execute_stage(root,args.arm,args.stage)
    if args.run: run_all(root)
    if args.report:
        status=root/'status.json'
        report(root,finished=json.loads(status.read_text()).get('complete',False) if status.exists() else False)


if __name__=='__main__':
    main()
