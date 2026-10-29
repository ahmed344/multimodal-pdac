"""Common-row, training-frozen diagnostics for the dictionary pooling experiment."""
from __future__ import annotations

import copy
import html
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from scipy.special import expit, ndtr, ndtri
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch

from dann.analysis_execution import digest_file, fingerprint
from dann.checkpoints import checkpoint_config
from dann.config import load_config
from dann.data_loader import load_anndata_metadata, SparseAnnDataDataset, sparse_collate
from dann.execution import ResidentSpectra
from dann.model import AdversarialLatentFusion
from dann.model_components.deep_sets import DeepSetsEncoder
from dann.spatial import _file_identity
from dann.train import move_batch_to_device
from dann.training_artifacts import atomic_json

KEYS = ('peak_indices', 'intensities', 'sample_indices', 'peak_counts')


def json_clean(value):
    """Encode undefined subgroup scores as null, never nonstandard JSON NaN."""
    if isinstance(value, dict):
        return {str(k):json_clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list,np.ndarray)):
        return [json_clean(v) for v in value]
    if isinstance(value, (float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path: Path, value: dict) -> None:
    """Write portable metric records with explicit nulls for undefined scores."""
    atomic_json(path, json_clean(value))


def covariance_metrics(x: np.ndarray, groups: dict[str, np.ndarray]) -> dict:
    """Separate shared offset energy, centered covariance, and group variation."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0); centered = x-mean
    covariance = centered.T @ centered / len(x)
    eig = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    total = np.square(x).mean(); variance = np.square(centered).mean()
    result = dict(rms=np.sqrt(total), centered_rms=np.sqrt(variance),
        centered_to_total_rms=np.sqrt(variance/max(total,1e-300)), shared_offset_energy_fraction=1-variance/max(total,1e-300),
        participation_ratio=eig.sum()**2/max(np.square(eig).sum(),1e-300), eigenvalues=eig)
    for name, labels in groups.items():
        between = sum(np.sum(labels == label)*np.sum(np.square(x[labels == label].mean(0)-mean))
                      for label in np.unique(labels))/len(x)
        result[name] = dict(between_variance=between, within_variance=eig.sum()-between,
                           between_fraction=between/max(eig.sum(),1e-300))
    return result


def associations(x: np.ndarray, covariates: dict[str, np.ndarray]) -> dict:
    """Summarize descriptive latent-coordinate correlations with spectral nuisance variables."""
    centered = np.asarray(x,dtype=np.float64)-x.mean(0)
    norms = np.linalg.norm(centered,axis=0)
    result = {}
    for name, values in covariates.items():
        y = values-values.mean()
        r = (centered.T @ y) / np.maximum(norms*np.linalg.norm(y),1e-30)
        result[name] = dict(mean_squared_correlation=float(np.mean(r*r)), max_abs_correlation=float(np.max(np.abs(r))))
    return result


def metadata_for(c: dict):
    """Read targets using the saved target transformation and feature order."""
    d = c['data']
    return load_anndata_metadata(Path(d['path']), d['target_columns'], d['batch_column'],
                                d['matrix_key'], d['cd8_normalization'])


def dataset_for(c: dict, metadata, rows: np.ndarray, resident=None) -> SparseAnnDataDataset:
    """Gather source rows with frozen training spectral preprocessing."""
    return SparseAnnDataDataset(Path(c['data']['path']), rows, metadata,
        **{k:c['data'][k] for k in ('matrix_key','intensity_transform','intensity_clip_max','nonzero_threshold')}, resident=resident)


def extract(model, c: dict, metadata, rows: np.ndarray, destination: Path, identity: dict, resident) -> dict:
    """Export pooled features and latents in exact requested source-row order."""
    stamp = destination.with_suffix('.json')
    if stamp.exists() and destination.exists():
        record = json.loads(stamp.read_text())
        if record['identity'] == identity and digest_file(destination) == record['sha256']:
            with np.load(destination) as data:
                return {k:data[k] for k in data.files}
    ds = dataset_for(c,metadata,rows,resident)
    collected = {k:[] for k in ('rows','latent','pooled','pi_logits','mu','sigma','count','stored_count','l1','l2')}
    device = next(model.parameters()).device
    started = time.monotonic()
    try:
        with torch.no_grad():
            for start in range(0,len(rows),2048):
                batch = move_batch_to_device(sparse_collate(ds.__getitems__(list(range(start,min(start+2048,len(rows)))))),device)
                pooled = DeepSetsEncoder.forward(model.encoder, *(batch[k] for k in KEYS))
                latent = model.encoder.aggregation_mlp(pooled)
                output = model.biology_predictor(latent)
                sparse = torch.sparse_coo_tensor(torch.stack((batch['sample_indices'],batch['peak_indices'])),
                    batch['intensities'],(len(batch['row_ids']),model.encoder.num_peaks)).coalesce()
                values = sparse.values(); row = sparse.indices()[0]
                l1 = values.new_zeros(len(batch['row_ids'])).index_add_(0,row,values)
                l2 = values.new_zeros(len(batch['row_ids'])).index_add_(0,row,values.square()).sqrt()
                count = torch.bincount(row[values > 0],minlength=len(batch['row_ids']))
                output.update(rows=batch['row_ids'],latent=latent,pooled=pooled,count=count,
                              stored_count=batch['peak_counts'],l1=l1,l2=l2)
                for key in collected:
                    collected[key].append(output[key].cpu().numpy())
    finally:
        ds.close()
    arrays = {key:np.concatenate(parts) for key,parts in collected.items()}
    np.testing.assert_array_equal(arrays['rows'],rows)
    assert len(np.unique(rows)) == len(rows)
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez(destination,**arrays)
    save_json(stamp,dict(identity=identity, sha256=digest_file(destination), rows=len(rows),
                        seconds=time.monotonic()-started, complete_coverage=True, finite=True))
    return arrays


def probability_arrays(data: dict, thresholds: list[float], epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute hurdle presence and fixed-threshold exceedance probabilities."""
    presence = expit(-data['pi_logits'][:,0].astype(np.float64))
    mu,sigma = (data[k][:,0].astype(np.float64) for k in ('mu','sigma'))
    t = np.clip(thresholds,epsilon,1-epsilon)
    exceeds = presence[:,None]*ndtr((mu[:,None]-np.log(t/(1-t)))/sigma[:,None])
    return presence,exceeds


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Return Brier, AUROC and marginal calibration on the supplied rows."""
    return dict(brier=np.mean((p-y)**2), auroc=roc_auc_score(y,p) if len(np.unique(y)) == 2 else np.nan,
                predicted=np.mean(p), observed=np.mean(y))


def reliability(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Use common fixed probability bins, retaining counts for every nonempty bin."""
    bins = np.minimum((p*10).astype(int),9)
    return [dict(bin=i,n=int(np.sum(bins == i)), predicted=np.mean(p[bins == i]),
                 observed=np.mean(y[bins == i])) for i in range(10) if np.any(bins == i)]


def distribution_metrics(data: dict, y: np.ndarray, mask: np.ndarray, thresholds: list[float], epsilon: float) -> dict:
    """Score configured logit-branch likelihood and conditional positive intervals."""
    results = {}
    for j,name in enumerate(('cd8','tumor','collagen','stroma')):
        valid = mask[:,j]; obs = y[valid,j].astype(np.float64); positive = obs>0
        logits,mu,sigma = (data[k][valid,j].astype(np.float64) for k in ('pi_logits','mu','sigma'))
        presence = expit(-logits)
        z = np.log(np.clip(obs,epsilon,1-epsilon)/(1-np.clip(obs,epsilon,1-epsilon)))
        loss = np.logaddexp(0,np.where(positive,logits,-logits)) + positive*(np.log(sigma)+.5*((z-mu)/sigma)**2)
        pit = ndtr((z[positive]-mu[positive])/sigma[positive])
        results[name] = dict(n=len(obs), loss=np.mean(loss), presence=binary_metrics(positive,presence),
            reliability=reliability(positive,presence), positive_pit_histogram=np.histogram(pit,np.linspace(0,1,51))[0],
            positive_pit_mean=np.mean(pit), intervals={})
        for level in (.5,.9):
            a = (1-level)/2
            width = expit(mu[positive]+sigma[positive]*ndtri(1-a))-expit(mu[positive]+sigma[positive]*ndtri(a))
            results[name]['intervals'][str(level)] = dict(coverage=np.mean((pit>=a)&(pit<=1-a)),width=np.mean(width))
        if j == 0:
            results[name]['exceedance'] = {}
            for threshold in thresholds:
                t=np.clip(threshold,epsilon,1-epsilon)
                p=presence*ndtr((mu-np.log(t/(1-t)))/sigma)
                results[name]['exceedance'][str(threshold)] = dict(**binary_metrics(obs>threshold,p), reliability=reliability(obs>threshold,p))
    return results


def fit_probes(train: dict, valid: dict, metadata, thresholds: list[float], epsilon: float, out: Path) -> dict:
    """Identical fixed probes; every scaler and model is fitted on training rows."""
    tr,va = train['rows'],valid['rows']
    scaler=StandardScaler().fit(train['latent'])
    x=scaler.transform(train['latent']).astype(np.float64)
    v=scaler.transform(valid['latent']).astype(np.float64)
    y,m = metadata.targets,metadata.target_valid_mask
    predictions={}; params={'scaler_mean':scaler.mean_, 'scaler_scale':scaler.scale_}
    for j,name in [(1,'tumor'),(2,'collagen'),(3,'stroma')]:
        model=Ridge(alpha=1.).fit(x[m[tr,j]],y[tr,j][m[tr,j]])
        predictions[name]=model.predict(v)
        params[f'{name}_coef']=model.coef_;params[f'{name}_intercept']=model.intercept_
        print(f'Fitted {name} ridge probe on {int(m[tr,j].sum())} training rows',flush=True)
    for name,threshold in [('presence',0.),*[(f'exceedance_{i}',t) for i,t in enumerate(thresholds)]]:
        model=LogisticRegression(C=1.,solver='lbfgs',max_iter=5000,random_state=20260719).fit(
            x[m[tr,0]],(y[tr,0][m[tr,0]]>threshold).astype(int))
        assert np.max(model.n_iter_)<5000, 'Logistic probe did not converge'
        predictions[name]=model.predict_proba(v)[:,1]
        params[f'{name}_coef']=model.coef_;params[f'{name}_intercept']=model.intercept_
        params[f'{name}_classes']=model.classes_
        print(f'Fitted {name} logistic probe in {int(np.max(model.n_iter_))} iterations',flush=True)
    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=5000,random_state=20260719).fit(x,metadata.batch_codes[tr])
    assert np.max(model.n_iter_)<5000, 'Slide probe did not converge'
    predictions['slide']=model.predict(v)
    params['slide_coef']=model.coef_;params['slide_intercept']=model.intercept_
    params['slide_classes']=model.classes_
    print(f'Fitted slide probe in {int(np.max(model.n_iter_))} iterations',flush=True)
    np.savez(out/'probe_parameters.npz',**params)
    np.savez(out/'probe_predictions.npz',**predictions)
    from importlib.metadata import version
    save_json(out/'probe_protocol.json',dict(
        training_rows=len(tr), training_rows_sha256=fingerprint(tr), validation_rows=len(va),
        scaling='StandardScaler fitted on shared training probe rows only',
        ridge=dict(alpha=1.), logistic=dict(C=1.,solver='lbfgs',max_iter=5000,random_state=20260719),
        thresholds=thresholds, target_order=['Density_CD8','Density_Tumor','Density_Collagen','Density_Stroma'],
        software={name:version(name) for name in ('numpy','scipy','scikit-learn')}))
    return predictions


def probe_metrics(predictions: dict, y: np.ndarray, mask: np.ndarray, batches: np.ndarray, thresholds: list[float], selected: np.ndarray) -> dict:
    """Score fixed predictions on all validation rows or a predefined stratum."""
    result={}
    for j,name in [(1,'tumor'),(2,'collagen'),(3,'stroma')]:
        take=selected & mask[:,j]
        result[name]=dict(r2=r2_score(y[take,j],predictions[name][take],force_finite=False) if take.sum()>1 else np.nan,
                         mse=np.mean((y[take,j]-predictions[name][take])**2))
    for name,t in [('presence',0.),*[(f'exceedance_{i}',t) for i,t in enumerate(thresholds)]]:
        take=selected & mask[:,0]
        result[name]=binary_metrics(y[take,0]>t,predictions[name][take]) if take.any() else {}
    result['slide']=dict(accuracy=np.mean(predictions['slide'][selected]==batches[selected]),
        balanced_accuracy=np.mean([np.mean(predictions['slide'][selected & (batches == label)] == label)
            for label in np.unique(batches[selected])]) if selected.any() else np.nan)
    return result


def relative_change(reference: np.ndarray, value: np.ndarray) -> dict:
    """Separate relative scale changes from cosine-direction changes."""
    delta=np.linalg.norm(value-reference,axis=1)
    norm=np.linalg.norm(reference,axis=1)
    cos=np.sum(reference*value,axis=1)/np.maximum(norm*np.linalg.norm(value,axis=1),1e-30)
    cos[(norm == 0) & (np.linalg.norm(value,axis=1) == 0)] = 1.
    centered_reference = reference-reference.mean(0)
    centered_value = value-value.mean(0)
    centered_norm = np.linalg.norm(centered_reference,axis=1)*np.linalg.norm(centered_value,axis=1)
    centered_cos = np.sum(centered_reference*centered_value,axis=1)/np.maximum(centered_norm,1e-30)
    centered_cos[(np.linalg.norm(centered_reference,axis=1) == 0) &
                 (np.linalg.norm(centered_value,axis=1) == 0)] = 1.
    return dict(relative_rms=np.linalg.norm(value-reference)/max(np.linalg.norm(reference),1e-30),
        median_relative_row_change=np.median(delta/np.maximum(norm,1e-12)),mean_cosine_change=np.mean(1-cos),
        relative_centered_rms=np.linalg.norm(centered_value-centered_reference)/max(np.linalg.norm(centered_reference),1e-30),
        mean_centered_cosine_change=np.mean(1-centered_cos))


def evaluate_batch(model, batch: dict, thresholds: list[float], epsilon: float) -> dict:
    """Return pooled, latent and CD8 probability arrays without gradients."""
    with torch.no_grad():
        pooled=DeepSetsEncoder.forward(model.encoder,*(batch[k] for k in KEYS))
        latent=model.encoder.aggregation_mlp(pooled)
        out=model.biology_predictor(latent)
        arrays={k:v.cpu().numpy() for k,v in out.items()}
        p,e=probability_arrays(arrays,thresholds,epsilon)
        return dict(pooled=pooled.cpu().numpy(),latent=latent.cpu().numpy(),presence=p,exceedance=e)


def perturbations(model, batch: dict, thresholds: list[float], epsilon: float, low_cutoff: float) -> list[dict]:
    """Apply shared perturbations to exactly the same measured validation spectra."""
    base=evaluate_batch(model,batch,thresholds,epsilon)
    records=[]
    variants=[('binary_presence',torch.ones_like(batch['intensities']))]
    for scale in (.1,.5,2.,10.):
        variants.append((f'transformed_global_scale_{scale}',batch['intensities']*scale))
        variants.append((f'raw_global_scale_{scale}',torch.log1p(torch.expm1(batch['intensities'].double())*scale).float()))
    def score(name,changed):
        out=evaluate_batch(model,changed,thresholds,epsilon)
        records.append(dict(perturbation=name,**{k:relative_change(base[k],out[k]) for k in ('pooled','latent')},
            presence_mean_absolute_change=np.mean(np.abs(out['presence']-base['presence'])),
            exceedance_mean_absolute_change=np.mean(np.abs(out['exceedance']-base['exceedance']),axis=0)))
    for name,values in variants:
        changed=dict(batch,intensities=values)
        score(name,changed)
    keep=batch['intensities']>low_cutoff
    changed=dict(batch)
    for k in KEYS[:3]: changed[k]=batch[k][keep]
    changed['peak_counts']=torch.bincount(changed['sample_indices'],minlength=len(batch['peak_counts']))
    score('remove_low_intensity_training_q10',changed)
    # Choose one absent peak deterministically in every row.
    rows=len(batch['peak_counts']); present=torch.zeros((rows,model.encoder.num_peaks),device=batch['intensities'].device,dtype=torch.bool)
    present[batch['sample_indices'],batch['peak_indices']]=True
    peak=(~present).int().argmax(1)
    for delta in (0.,1e-8,1e-6,1e-4,.01):
        changed=dict(batch)
        changed['peak_indices']=torch.cat((batch['peak_indices'],peak))
        changed['sample_indices']=torch.cat((batch['sample_indices'],torch.arange(rows,device=peak.device)))
        changed['intensities']=torch.cat((batch['intensities'],batch['intensities'].new_full((rows,),delta)))
        order=torch.argsort(changed['sample_indices'],stable=True)
        for k in KEYS[:3]:changed[k]=changed[k][order]
        changed['peak_counts']=batch['peak_counts']+1
        score(f'add_absent_peak_{delta}',changed)
    order=torch.arange(len(batch['intensities'])-1,-1,-1,device=peak.device)
    changed=dict(batch)
    for k in KEYS[:3]:changed[k]=batch[k][order]
    score('reverse_entry_order',changed)
    changed=dict(batch)
    for k in ('peak_indices','sample_indices'):changed[k]=batch[k].repeat_interleave(2)
    changed['intensities']=batch['intensities'].repeat_interleave(2)/2
    changed['peak_counts']=batch['peak_counts']*2
    score('duplicate_split_after_transform',changed)
    changed=dict(batch)
    for k in KEYS[:3]:changed[k]=batch[k][:0]
    changed['peak_counts']=torch.zeros_like(batch['peak_counts'])
    empty=evaluate_batch(model,changed,thresholds,epsilon)
    assert np.array_equal(empty['pooled'],np.zeros_like(empty['pooled']))
    return records


def intensity_responses(model, output: Path, phase: str, realistic_range: np.ndarray) -> None:
    """Per-peak contribution derivatives, clearly separate from normalized pooling."""
    enc=model.encoder
    peaks=torch.linspace(0,enc.num_peaks-1,32,device=enc.embedding.weight.device).long()
    lo,hi=realistic_range
    levels=np.unique(np.r_[0.,np.geomspace(1e-8,1e2,31),np.linspace(lo,hi,21)])
    records=[]
    with torch.enable_grad():
        for intensity in levels:
            values=enc.embedding.weight.new_full((len(peaks),),float(intensity),requires_grad=True)
            if enc.pooling == 'occurrence_mlp':
                def response(x):
                    value=enc.peak_mlp(enc.embedding(peaks)*x[:,None])
                    if enc.pool_centering == 'zero_input':
                        value=value-enc.peak_mlp(torch.zeros_like(enc.embedding(peaks)))
                    return value
            else:
                dictionary=enc.effective_dictionary()[peaks].detach()
                def response(x): return x[:,None]*dictionary
            value,derivative=torch.autograd.functional.jvp(response,values,torch.ones_like(values))
            for i,peak in enumerate(peaks.tolist()):
                records.append(dict(phase=phase,peak=peak,intensity=intensity,
                    range='realistic' if lo<=intensity<=hi else 'stress',
                    contribution_norm=float(value[i].norm()),derivative_norm=float(derivative[i].norm())))
    pd.DataFrame(records).to_csv(output/f'{phase}_intensity_responses.csv',index=False)
    with torch.no_grad():
        if enc.pooling == 'intensity_weighted_dictionary':
            np.save(output/f'{phase}_effective_dictionary.npy',enc.effective_dictionary().cpu().numpy())
        else:
            for intensity in (lo,(lo+hi)/2,hi):
                response=enc.peak_mlp(enc.embedding.weight*float(intensity))
                if enc.pool_centering == 'zero_input':
                    response-=enc.peak_mlp(enc.embedding.weight.new_zeros(1,enc.embedding_dim))
                np.save(output/f'{phase}_conditioned_response_{intensity:.5g}.npy',response.cpu().numpy())


def pooled_response_curves(model, batch: dict, thresholds: list[float], epsilon: float) -> list[dict]:
    """Finite differences through pooling, latent aggregation and CD8 probabilities.

    Vary the strongest observed peak in each of the first 128 common audit rows.
    The same measured peaks are used for every encoder and phase.
    """
    rows = min(128, len(batch['peak_counts']))
    stop = int(batch['peak_counts'][:rows].sum())
    small = {key: batch[key][:stop] for key in KEYS[:3]}
    small['peak_counts'] = batch['peak_counts'][:rows]
    selected = []
    offset = 0
    for count in small['peak_counts'].tolist():
        if count:
            selected.append(offset + int(small['intensities'][offset:offset+count].argmax()))
        offset += count
    indices = torch.tensor(selected, device=small['intensities'].device, dtype=torch.long)
    reference = evaluate_batch(model, small, thresholds, epsilon)
    def at(factor):
        values = small['intensities'].clone()
        values[indices] *= factor
        return evaluate_batch(model, dict(small, intensities=values), thresholds, epsilon)
    records = []
    for factor in (0., .001, .01, .1, .5, 1., 2., 5., 10.):
        step = 1e-3 * max(1., factor)
        lo, hi = max(0., factor-step), factor+step
        left, right, value = at(lo), at(hi), at(factor)
        records.append(dict(factor=factor, derivative_method='finite difference in intensity multiplier',
            finite_difference_step=step,
            pooled_relative_change=relative_change(reference['pooled'],value['pooled'])['relative_rms'],
            latent_relative_change=relative_change(reference['latent'],value['latent'])['relative_rms'],
            pooled_relative_derivative=np.linalg.norm(right['pooled']-left['pooled'])/(hi-lo)/max(np.linalg.norm(reference['pooled']),1e-30),
            latent_relative_derivative=np.linalg.norm(right['latent']-left['latent'])/(hi-lo)/max(np.linalg.norm(reference['latent']),1e-30),
            presence_mean_absolute_derivative=np.mean(np.abs(right['presence']-left['presence']))/(hi-lo),
            exceedance_mean_absolute_derivative=np.mean(np.abs(right['exceedance']-left['exceedance']),axis=0)/(hi-lo)))
    return records


def benchmark(model, batch: dict) -> dict:
    """Matched resident workloads, excluding transfers and optimizer updates."""
    device=next(model.parameters()).device
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)
    records={}
    for mode in ('forward','inference','forward_backward'):
        torch.manual_seed(20260719)
        model.train(mode=='forward_backward')
        timings=[]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for i in range(12):
            model.zero_grad(set_to_none=True);synchronize();start=time.perf_counter()
            with torch.set_grad_enabled(mode=='forward_backward'):
                out=model(batch, predict_he=mode!='inference')
                if mode=='forward_backward':
                    # Same branches and tensor sizes for each architecture.
                    loss=sum(out[k].square().mean() for k in ('mu','sigma','pi_logits','batch_logits','he_prediction'))
                    loss.backward()
            synchronize()
            if i>=2:timings.append(time.perf_counter()-start)
        records[mode]=dict(median_seconds=np.median(timings),seconds=timings,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None)
    model.eval();model.zero_grad(set_to_none=True)
    dictionary=model.encoder.pooling=='intensity_weighted_dictionary'
    reference_rows=int(not dictionary and model.encoder.pool_centering=='zero_input')
    records.update(rows=len(batch['peak_counts']),nnz=len(batch['intensities']),
        peak_mlp_input_rows=model.encoder.num_peaks if dictionary else len(batch['intensities'])+reference_rows,
        zero_input_reference_rows=reference_rows)
    return records


def centering_comparison(model, batch: dict, thresholds: list[float], epsilon: float) -> dict:
    """Hold weights and spectra fixed while varying the applicable centering rule."""
    result = {}
    original = model.encoder.pool_centering
    modes = ('none', 'zero_input') if model.encoder.pooling == 'occurrence_mlp' else ('none', 'dictionary_mean')
    try:
        for mode in modes:
            model.encoder.pool_centering = mode
            state = evaluate_batch(model, batch, thresholds, epsilon)
            result[mode] = {key: covariance_metrics(state[key], {}) for key in ('pooled', 'latent')}
    finally:
        model.encoder.pool_centering = original
    return result


def run_diagnostics(root: Path, arm: str, manifest: dict) -> dict:
    """Build matched full-validation evidence from one selected checkpoint."""
    run=manifest['runs'][arm];c=load_config(Path(run['config']))
    checkpoint=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)
    saved=checkpoint_config(checkpoint)
    assert all(saved['model'][key]==c['model'][key] for key in ('num_peaks','latent_dim'))
    device=torch.device(c['training']['device'])
    model=AdversarialLatentFusion.from_config(saved,len(checkpoint['batch_names']),4).to(device).eval()
    model.load_state_dict(checkpoint['model_state'])
    out=root/'diagnostics'/arm;out.mkdir(exist_ok=True)
    metadata=metadata_for(c)
    resident=ResidentSpectra(Path(c['data']['path']),c['data'],int(128*2**30))
    identity=dict(extraction_version=3,checkpoint_sha256=digest_file(Path(run['checkpoint'])),input=_file_identity(Path(c['data']['path'])))
    train_rows=np.load(root/'probe_train_rows.npy');valid_rows=checkpoint['split_indices']['validation']
    train=extract(model,c,metadata,train_rows,out/'train_features.npz',dict(identity,rows_sha256=fingerprint(train_rows)),resident)
    valid=extract(model,c,metadata,valid_rows,out/'validation_features.npz',dict(identity,rows_sha256=fingerprint(valid_rows)),resident)
    export_equivalence = {}
    analysis_dir = Path(run['checkpoint']).parent.parent/'analysis'
    if (analysis_dir/'analysis_execution.json').exists():
        execution = json.loads((analysis_dir/'analysis_execution.json').read_text())
        assert execution['source_identity']['checkpoint'] == identity['checkpoint_sha256']
        directory = analysis_dir/'.analysis-cache/extracted-validation'/execution['cache_entries']['extracted-validation']
        arrays = json.loads((directory/'manifest.json').read_text())['arrays']
        reference_rows = np.load(directory/arrays['row_ids']['file'], mmap_mode='r')
        left, right = np.argsort(valid['rows']), np.argsort(reference_rows)
        np.testing.assert_array_equal(valid['rows'][left], reference_rows[right])
        for key in ('latent','mu','sigma','pi'):
            reference = np.load(directory/arrays[key]['file'], mmap_mode='r')[right]
            current = expit(valid['pi_logits'][left]) if key == 'pi' else valid[key][left]
            np.testing.assert_allclose(current,reference,atol=2e-5,rtol=2e-5)
            export_equivalence[key] = dict(max_absolute_difference=float(np.max(np.abs(current-reference))),
                                           rows=len(valid_rows), atol=2e-5, rtol=2e-5)
    from anndata.io import read_elem
    with h5py.File(c['data']['path']) as handle:
        patients=np.asarray(read_elem(handle['obs/patient']),dtype=str)
        x=np.asarray(read_elem(handle['obs/x']));ycoord=np.asarray(read_elem(handle['obs/y']))
    np.savez(out/'row_metadata.npz',rows=valid_rows,patient=patients[valid_rows],slide=metadata.batch_codes[valid_rows],
        targets=metadata.targets[valid_rows],valid=metadata.target_valid_mask[valid_rows],x=x[valid_rows],y=ycoord[valid_rows])
    y=metadata.targets[valid_rows];mask=metadata.target_valid_mask[valid_rows]
    groups=dict(slide=metadata.batch_codes[valid_rows],patient=patients[valid_rows])
    representations={}
    for kind in ('pooled','latent'):
        representations[kind]=covariance_metrics(valid[kind],groups)
        representations[kind]['associations']=associations(valid[kind],{k:valid[k] for k in ('count','stored_count','l1','l2')})
    predictions=fit_probes(train,valid,metadata,manifest['cd8_thresholds'],c['loss']['logit_epsilon'],out)
    selected={'all':np.ones(len(y),bool)}
    selected.update({f'slide_{name}':metadata.batch_codes[valid_rows]==i for i,name in enumerate(metadata.batch_names)})
    quartiles=np.digitize(y[:,3],manifest['stroma_quartiles'])
    selected.update({f'stroma_q{i+1}':(quartiles==i)&mask[:,3] for i in range(4)})
    scores={name:probe_metrics(predictions,y,mask,groups['slide'],manifest['cd8_thresholds'],take) for name,take in selected.items()}
    distributions={name:distribution_metrics({k:v[take] for k,v in valid.items()},y[take],mask[take],manifest['cd8_thresholds'],c['loss']['logit_epsilon']) for name,take in selected.items()}
    audit_rows=np.load(root/'audit_rows.npy')
    ds=dataset_for(c,metadata,audit_rows,resident)
    try:
        batch=move_batch_to_device(sparse_collate(ds.__getitems__(list(range(len(ds))))),device)
    finally:ds.close()
    # Training-only intensity reference, measured on the same probe sample.
    values=np.concatenate([resident.values[resident.indptr[r]:resident.indptr[r+1]] for r in train_rows])
    intensity_quantiles=np.quantile(values[values>0],[.01,.1,.99])
    sensitivity=perturbations(model,batch,manifest['cd8_thresholds'],c['loss']['logit_epsilon'],intensity_quantiles[1])
    response_curves=pooled_response_curves(model,batch,manifest['cd8_thresholds'],c['loss']['logit_epsilon'])
    intensity_responses(model,out,'selected',intensity_quantiles[[0,2]])
    centered=centering_comparison(model,batch,manifest['cd8_thresholds'],c['loss']['logit_epsilon'])
    costs=benchmark(model,batch)
    initial=root/'diagnostics'/('l1' if arm=='baseline' else arm)/'initial_parameters.pt'
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True))
    intensity_responses(model,out,'initial_reconstructed' if arm=='baseline' else 'initial',intensity_quantiles[[0,2]])
    initial_sensitivity=perturbations(model,batch,manifest['cd8_thresholds'],c['loss']['logit_epsilon'],intensity_quantiles[1])
    initial_response_curves=pooled_response_curves(model,batch,manifest['cd8_thresholds'],c['loss']['logit_epsilon'])
    initial_centering=centering_comparison(model,batch,manifest['cd8_thresholds'],c['loss']['logit_epsilon'])
    report=dict(identity=identity,representations=representations,probes=scores,distributions=distributions,
        export_equivalence=export_equivalence,
        sensitivity=sensitivity,initial_sensitivity=initial_sensitivity,centering_controls=centered,benchmark=costs,
        response_curves=response_curves,initial_response_curves=initial_response_curves,
        initial_centering_controls=initial_centering,
        training_intensity_quantiles=intensity_quantiles,
        training_spectral_summary=dict(quantile_levels=[.01,.25,.5,.75,.99],
            stored_entries_per_row=np.quantile(train['stored_count'],[.01,.25,.5,.75,.99]),
            positive_distinct_peaks_per_row=np.quantile(train['count'],[.01,.25,.5,.75,.99]),
            mean_positive_distinct_peaks=float(train['count'].mean()),
            mean_fraction_vocabulary_zero=float(1-train['count'].mean()/model.encoder.num_peaks),
            stored_zero_fraction=float(np.mean(values==0))),
        baseline_early_updates='Unavailable for this historical full run; initial parameters reconstructed from verified common seed' if arm=='baseline' else 'early_diagnostics.json')
    save_json(out/'metrics.json',report)
    # Flat tables are intentionally generated from the exact JSON record.
    pd.json_normalize(json_clean(report),sep='.').to_csv(out/'metrics.csv',index=False)
    return dict(metrics=str(out/'metrics.json'),validation_rows=len(valid_rows),probe_train_rows=len(train_rows),**identity)


def weighted_auc(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> float:
    """Exact weighted AUROC with half credit for ties."""
    order=np.argsort(score,kind='stable');s=score[order];y=y[order];w=weights[order]
    boundaries=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    positives=np.add.reduceat(w*y,boundaries);negatives=np.add.reduceat(w*(1-y),boundaries)
    denominator=positives.sum()*negatives.sum()
    return np.dot(positives,np.cumsum(negatives)-.5*negatives)/denominator if denominator>0 else np.nan


def paired_bootstrap(root: Path, manifest: dict, metrics: dict) -> pd.DataFrame:
    """Patient resampling is paired across models; no training-seed uncertainty."""
    with np.load(root/'diagnostics/baseline/row_metadata.npz') as z:
        y=z['targets'].astype(np.float64);mask=z['valid'];patient=z['patient']
    names,codes=np.unique(patient,return_inverse=True);n=len(names)
    rng=np.random.default_rng(manifest['bootstrap']['seed'])
    draws=np.stack([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(manifest['bootstrap']['replicates'])])
    estimates={}
    epsilon=load_config(Path(manifest['runs']['baseline']['config']))['loss']['logit_epsilon']
    for arm in ('baseline','l1','l2'):
        with np.load(root/'diagnostics'/arm/'validation_features.npz') as z:
            d={k:z[k].astype(np.float64) for k in ('pi_logits','mu','sigma')}
        with np.load(root/'diagnostics'/arm/'probe_predictions.npz') as z:
            probes={k:z[k] for k in z.files}
        p,e=probability_arrays(d,manifest['cd8_thresholds'],epsilon)
        valid=mask[:,0];positive=y[:,0]>0
        z=np.log(np.clip(y[:,0],epsilon,1-epsilon)/(1-np.clip(y[:,0],epsilon,1-epsilon)))
        losses=np.logaddexp(0,np.where(positive,d['pi_logits'][:,0],-d['pi_logits'][:,0]))+positive*(
            np.log(d['sigma'][:,0])+.5*((z-d['mu'][:,0])/d['sigma'][:,0])**2)
        current={}
        def means(key,values,keep):
            count=np.bincount(codes[keep],minlength=n)
            sums=np.bincount(codes[keep],weights=values[keep],minlength=n)
            current[key]=(float(values[keep].mean()),(draws@sums)/(draws@count))
        def aucs(key,labels,pred,keep):
            yy=labels[keep].astype(float);pp=pred[keep];cc=codes[keep]
            order=np.argsort(pp,kind='stable');ordered_y=yy[order];ordered_codes=cc[order]
            boundaries=np.r_[0,np.flatnonzero(np.diff(pp[order])!=0)+1]
            replicates=[]
            for draw in draws:
                weights=draw[ordered_codes]
                pos=np.add.reduceat(weights*ordered_y,boundaries)
                neg=np.add.reduceat(weights*(1-ordered_y),boundaries)
                denominator=pos.sum()*neg.sum()
                replicates.append(np.dot(pos,np.cumsum(neg)-.5*neg)/denominator if denominator>0 else np.nan)
            current[key]=(roc_auc_score(yy,pp),np.asarray(replicates))
        means('prediction_cd8_loss',losses,valid)
        means('prediction_presence_brier',(p-positive)**2,valid)
        aucs('prediction_presence_auroc',positive,p,valid)
        pit=ndtr((z-d['mu'][:,0])/d['sigma'][:,0])
        for level in (.5,.9):
            a=(1-level)/2
            means(f'prediction_positive_coverage_{level}',((pit>=a)&(pit<=1-a)).astype(float),valid&positive)
            width=expit(d['mu'][:,0]+d['sigma'][:,0]*ndtri(1-a))-expit(d['mu'][:,0]+d['sigma'][:,0]*ndtri(a))
            means(f'prediction_positive_width_{level}',width,valid&positive)
        means('probe_presence_brier',(probes['presence']-positive)**2,valid)
        aucs('probe_presence_auroc',positive,probes['presence'],valid)
        for i,t in enumerate(manifest['cd8_thresholds']):
            means(f'prediction_exceedance_{i}_brier',(e[:,i]-(y[:,0]>t))**2,valid)
            means(f'probe_exceedance_{i}_brier',(probes[f'exceedance_{i}']-(y[:,0]>t))**2,valid)
        for j,name in [(1,'tumor'),(2,'collagen'),(3,'stroma')]:
            keep=mask[:,j];obs=y[keep,j];pred=probes[name][keep];cc=codes[keep]
            count=np.bincount(cc,minlength=n);sums=np.bincount(cc,weights=obs,minlength=n)
            sq=np.bincount(cc,weights=obs**2,minlength=n);error=np.bincount(cc,weights=(pred-obs)**2,minlength=n)
            boot=1-(draws@error)/((draws@sq)-(draws@sums)**2/(draws@count))
            current[f'probe_{name}_r2']=(r2_score(obs,pred),boot)
        estimates[arm]=current
    records=[]
    for arm,current in estimates.items():
        for key,(point,boot) in current.items():
            lower,upper=np.nanquantile(boot,[.025,.975])
            records.append(dict(arm=arm,reference='absolute',metric=key,estimate=point,lower=lower,upper=upper,patients=n))
    for arm,reference in [('l1','baseline'),('l2','baseline'),('l2','l1')]:
        for key,(point,boot) in estimates[arm].items():
            delta=boot-estimates[reference][key][1]
            lower,upper=np.nanquantile(delta,[.025,.975])
            records.append(dict(arm=arm,reference=reference,metric=key,estimate=point-estimates[reference][key][0],lower=lower,upper=upper,patients=n))
    frame=pd.DataFrame(records);frame.to_csv(root/'paired_patient_bootstrap.csv',index=False)
    return frame


def cross_model_similarity(root: Path, manifest: dict) -> pd.DataFrame:
    """Compare centered linear CKA and cosine neighborhoods on identical rows."""
    rows=np.load(root/'audit_rows.npy');representations={};records=[]
    for arm in ('baseline','l1','l2'):
        with np.load(root/'diagnostics'/arm/'validation_features.npz') as d:
            lookup={int(r):i for i,r in enumerate(d['rows'])};indices=[lookup[int(r)] for r in rows]
            representations[arm]={kind:d[kind][indices].astype(np.float64) for kind in ('pooled','latent')}
    for kind in ('pooled','latent'):
        neighborhoods={}
        for arm,rep in representations.items():
            values=rep[kind]
            neighborhoods[arm]=NearestNeighbors(n_neighbors=16,metric='cosine',algorithm='brute').fit(values).kneighbors(values,return_distance=False)[:,1:]
        for arm,reference in [('l1','baseline'),('l2','baseline'),('l2','l1')]:
            a=representations[arm][kind];a=a-a.mean(0)
            b=representations[reference][kind];b=b-b.mean(0)
            cka=np.square(a.T@b).sum()/np.sqrt(np.square(a.T@a).sum()*np.square(b.T@b).sum())
            overlap=np.mean([len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(neighborhoods[arm],neighborhoods[reference])])
            records.append(dict(arm=arm,reference=reference,representation=kind,linear_cka=cka,neighbor_jaccard=overlap,rows=len(rows),neighbors=15))
    frame=pd.DataFrame(records);frame.to_csv(root/'cross_model_similarity.csv',index=False)
    return frame


def claim_verdicts(metrics: dict, root: Path) -> pd.DataFrame:
    """Audit substantive claims separately; avoid conflating proof and experiments."""
    confirmed='mathematically confirmed';refuted='mathematically refuted';unresolved='unresolved'
    records=[
        ('Before the MLP, positive intensity changes embedding magnitude but not direction',confirmed,
         'For a fixed nonzero embedding and I > 0, ||I*e|| = I*||e|| and its unit direction is e/||e||. Recovering I from the norm also requires the embedding norm; this is not a claim about the MLP output.'),
        ('The expanded LayerNorm expression includes bias covariance and epsilon',confirmed,
         'Centering I*a+b gives I*(a-mean(a))+(b-mean(b)); its variance is I^2*Var(a)+2*I*Cov(a,b)+Var(b). Epsilon remains inside the square root.'),
        ('LayerNorm completely removes scalar intensity',refuted,
         'Linear bias is generally nonconstant across channels and LayerNorm epsilon is positive. Both retain intensity dependence; equal-width residual paths also matter. Response curves measure the actual extent.'),
        ('Bias is the only possible source of intensity dependence',refuted,
         'Even zero bias leaves the epsilon / intensity^2 term. Approximate invariance can hold when bias and epsilon are negligible; exact invariance does not generally hold.'),
        ('Every occurrence receives the same 1/count coefficient',confirmed,
         'This is the historical aggregation formula. It is not an assertion of equal vector norms or equal predictive influence.'),
        ('Trace and high-intensity peaks have identical norms and 110/120 noise peaks contribute 91.7% of the vector',refuted,
         'The MLP, final linear projection and zero-input subtraction do not impose equal output norms. Vector directions can reinforce or cancel; count fractions are not signal-energy fractions.'),
        ('Adding near-zero peaks is discontinuous under count pooling, including zero-input centering',confirmed,
         'For n nonempty entries with centered sum S, adding an entry whose response tends to zero changes S/n to S/(n+1). Uncentered pooling also adds the finite phi(0) term. Within a fixed stored-entry pattern the response itself is continuous.'),
        ('GELU of a standard normal has mean 1/sqrt(2*pi), approximately .40',refuted,
         'For exact GELU, E[Z Phi(Z)] = E[phi(Z)] = 1/(2 sqrt(pi)), approximately .2821. The stated .3989 is the standard-normal ReLU mean. Learned normalized activations need not be Gaussian.'),
        ('A common offset implies rank-one centered representation collapse',refuted,
         'A constant offset can dominate the uncentered second moment without changing centered covariance. The final projection permits arbitrary signs. Offset energy and centered participation ratio are reported separately.'),
        ('Pixel-to-pixel variation necessarily shrinks as 1/sqrt(number of active peaks)',refuted,
         'That sampling law requires assumptions about independent responses and their covariance. Shared learned peaks and structured biological compositions need not satisfy them; the count alone does not determine centered variance.'),
        ('The offset causes the initial AdamW collapse in all MLP/CNN/GATv2 architectures',unresolved,
         'The present full fits hold residuals, warmup and head initialization fixed and use MLPs. Centering controls isolate immediate forward effects, not a universal causal training claim.'),
        ('Centering alone did not rescue the cited 512-wide three-epoch ablation',unresolved,
         'This is a distinct historical experiment. Its table is not evidence for the selected 256-wide full-fit baseline; no new centering-only full fit was requested.'),
        ('The quoted .7529/.6385 cosines, 8.3 norms and 99.8% offset characterize the cited initialization',unresolved,
         'These depend on initialization, weights, intensity range, rows and centering. Current common-row numerical measurements replace such universal interpretations; original measurement provenance is not established here.'),
        ('A fixed vocabulary forces background peaks to dominate biological peaks',unresolved,
         'Permutation-invariant pooling is well-defined over a finite vocabulary. A projection bottleneck can lose information in either architecture; vocabulary size alone does not establish background dominance. This is evaluated with probes and perturbations.'),
        ('Dictionary normalization preserves 100% of intensity information',refuted,
         'Both L1 and L2 remove positive global transformed-intensity scale above the floor, and projection from 5808 to 256 dimensions is generally noninjective. They preserve relative linear weighting before projection.'),
        ('For the same dictionary peak, intensity 5 gives 500 times the numerator contribution of .01',confirmed,
         'The pre-normalization contribution is I*v[p]. Different peaks can have different dictionary norms; row normalization changes the denominator.'),
        ('Dictionary pooling is continuous at zero including empty spectra',confirmed,
         'The sparse numerator is linear and the positive floored denominator is continuous. Empty spectra give zero. Derivatives at the floor need not be continuous, and near-empty rows can have large slopes.'),
        ('Sum of log1p intensities is physical raw total ion current',refuted,
         'This experiment normalizes the sum of transformed intensities, not the sum of raw ion intensities.'),
        ('Dictionary MLP evaluations fall from nnz to P per invocation',confirmed,
         'The dictionary MLP receives 5808 rows once per invocation; sparse pooling reuses them across spectra and internal chunks. Occurrence pooling additionally evaluates one zero-input reference row when centered. Dropout is shared per peak. Checkpoint backward may recompute the same realization.'),
        ('Dictionary pooling guarantees greater than 40-fold end-to-end time and memory speedup',unresolved,
         'MLP row-count reduction is not an end-to-end speedup. Matched measured timings, peak memory, full validation and inference costs are reported separately.'),
        ('Dictionary pooling resolves biological dilution or prevents training collapse',unresolved,
         'Biological utility must be supported by held-out stromal probes and CD8 distributions. Higher rank or variance alone is insufficient; one seed cannot establish universal prevention.'),
        ('An intensity-gated nonlinear response guarantees continuity',confirmed,
         'For bounded continuous h near zero and a positive denominator floor, I*h(I) tends to zero. This is conditional; the gated alternative was not fitted here.'),
        ('Concatenating scalar intensity guarantees that every learned gated MLP preserves it',refuted,
         'Concatenation removes the specific scalar-rescaling symmetry of I*e, but learned weights can ignore the scalar channel or project distinct intensities to the same output. It provides access to intensity, not an information-preservation guarantee.'),
        ('Every low absolute-intensity peak is negligible under intensity weighting or attention',refuted,
         'Relative intensity, dictionary/value norms, total competing mass and attention scores determine influence. A low absolute-intensity peak can dominate a nearly empty row. For a fixed nonempty reference row, bounded responses and a positive denominator floor give vanishing influence as its intensity tends to zero.'),
        ('softmax(score + log(I+epsilon)) equals I*exp(score)/(sum(I*exp(score))+epsilon)',refuted,
         'The left expression weights I+epsilon and sums those terms in its denominator. At I=0 it gives positive mass; it is not the stated floored intensity prior and does not give exact sparse-insertion continuity.'),
        ('Multiple attention heads guarantee selective biological peaks and remove superposition',unresolved,
         'Learned attention can select different peaks but provides no such guarantee. No attention pooling fit was part of this experiment.'),
        ('Segment-wise attention can operate in O(nnz) without dense vocabulary padding',confirmed,
         'With fixed numbers of heads and channels, segment reductions over occurrences are linear in nnz; feature/value tensors still consume occurrence-dependent memory.'),
        ('Deep Sets history and equivalence to scVI/scGPT/PointNet/Set Transformer standard practice',unresolved,
         'Literature claims were not established by these code and data experiments. No conclusion here depends on those attributions.'),
    ]
    spectrum=metrics['baseline'].get('training_spectral_summary')
    if spectrum:
        median=spectrum['positive_distinct_peaks_per_row'][2]
        records.append(('This dataset typically has 80–160 positive peaks per spectrum',
            'empirically supported in this experiment' if 80<=median<=160 else 'empirically contradicted in this experiment',
            f'The shared training probe sample has median {median:g} positive distinct peaks; '
            f'1/25/50/75/99 percentiles are {spectrum["positive_distinct_peaks_per_row"]}. '
            f'The mean zero fraction of the 5808-bin vocabulary is {spectrum["mean_fraction_vocabulary_zero"]:.4%}. '
            f'Stored-entry zero fraction is {spectrum["stored_zero_fraction"]:.4%}; stored counts are reported separately.'))
    cost=metrics['baseline']['benchmark']['forward_backward']['median_seconds']
    memory=metrics['baseline']['benchmark']['forward_backward']['peak_allocated_gib']
    memory_reductions=[memory/metrics[a]['benchmark']['forward_backward']['peak_allocated_gib'] for a in ('l1','l2')] if memory else []
    speedups=[cost/metrics[a]['benchmark']['forward_backward']['median_seconds'] for a in ('l1','l2')]
    for i,(claim,verdict,reason) in enumerate(records):
        if claim.startswith('Dictionary pooling guarantees greater'):
            records[i]=(claim,'empirically supported in this experiment' if min(speedups)>40 and memory_reductions and min(memory_reductions)>40 else 'empirically contradicted in this experiment',
                reason+f' Matched forward/backward speedups: L1 {speedups[0]:.2f}x, L2 {speedups[1]:.2f}x. Peak-memory reductions: {memory_reductions}.')
    archive=Path('data/PDAC/Results/dann/diagnostics/2026-10-28-stabilization/comparison_all_epochs.csv')
    if archive.exists():
        historical=pd.read_csv(archive)
        expected={
            'baseline':([1.2489,1.2483,1.0944],[1.01,1.10,2.69]),
            'centering':([1.2490,1.2482,1.1444],[1.08,1.14,1.92]),
            'residuals':([1.1287,1.0453,1.0172],[2.32,3.85,4.75]),
            'initialization':([1.0845,1.0487,1.0300],[2.95,4.10,4.59]),
            'warmup':([1.0944,1.0302,1.0102],[3.78,6.84,8.52]),
            'combined':([1.0078,.9802,.9742],[9.55,13.83,15.16]),
        }
        for name,(loss,pr) in expected.items():
            values=historical[(historical.arm==f'mlp_512_{name}_lr0.001')&(historical.epoch>0)].sort_values('epoch')
            matches=(len(values)==3 and np.allclose(values.cd8_loss,loss,atol=5.01e-5,rtol=0)
                     and np.allclose(values.participation_ratio,pr,atol=.00501,rtol=0))
            records.append((f'The cited three-epoch {name} ablation has the tabulated losses and ranks',
                'empirically supported in this experiment' if matches else 'empirically contradicted in this experiment',
                'Archive audit: values were compared at the stated rounding precision in the original 512-wide setup. This verifies historical observations; it does not replicate that ablation in the current 256-wide fits.'))
        historical.to_csv(root/'historical_stabilization.csv',index=False)
        save_json(root/'historical_stabilization_provenance.json',dict(source=str(archive.resolve()),sha256=digest_file(archive)))
        for i,(claim,verdict,reason) in enumerate(records):
            if claim.startswith('Centering alone did not rescue'):
                records[i]=(claim,'empirically supported in this experiment',
                    'Archive audit confirms participation ratios near one in epochs 1–2 for the cited 512-wide centering-only arm. The causal mechanism and transfer of this result to the current 256-wide setup remain unresolved.')
    frame=pd.DataFrame(records,columns=['claim','verdict','evidence_or_limit']);frame.to_csv(root/'claim_verdicts.csv',index=False)
    return frame


def figures(root: Path, metrics: dict) -> list[str]:
    """Publish matching raster and vector figures for the offline report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=root/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(directory/f'{name}.png',dpi=180);fig.savefig(directory/f'{name}.pdf');plt.close(fig)
    manifest=json.loads((root/'manifest.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm in metrics:
        history=pd.read_csv(Path(manifest['runs'][arm]['checkpoint']).parent/'history.csv')
        for ax,key in zip(axes,('validation_cd8_loss','validation_he_mse','validation_latent_participation_ratio')):
            ax.plot(history.epoch,history[key],label=arm)
    for ax,label in zip(axes,('Configured CD8 loss','H&E MSE','Centered latent participation ratio')):
        ax.set(xlabel='Completed epoch',ylabel=label);ax.legend()
    save(fig,'training_histories')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,('pooled','latent')):
        for arm,m in metrics.items():
            e=np.asarray(m['representations'][kind]['eigenvalues']);ax.semilogy(np.arange(1,len(e)+1),e/max(e.sum(),1e-30),label=arm)
        ax.set(xlabel='Centered component',ylabel='Fraction of centered variance',title=kind);ax.legend()
    save(fig,'covariance_spectra')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,(arm,m) in enumerate(metrics.items()):
        axes[0].bar(np.arange(3)+(i-1)*.25,[m['probes']['all'][k]['r2'] for k in ('stroma','collagen','tumor')],width=.25,label=arm)
        rel=m['distributions']['all']['cd8']['reliability']
        axes[1].plot([r['predicted'] for r in rel],[r['observed'] for r in rel],'.-',label=arm)
    axes[0].set(xticks=np.arange(3),xticklabels=['Stroma','Collagen','Tumor'],ylabel='Held-out fixed ridge R²');axes[0].legend()
    axes[1].plot([0,1],[0,1],'k--',linewidth=.8);axes[1].set(xlabel='Predicted presence',ylabel='Observed presence');axes[1].legend()
    save(fig,'probes_and_reliability')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for i,threshold in enumerate(manifest['cd8_thresholds']):
        ax=axes[0,i]
        for arm,m in metrics.items():
            reliability_values=m['distributions']['all']['cd8']['exceedance'][str(threshold)]['reliability']
            ax.plot([r['predicted'] for r in reliability_values],
                    [r['observed'] for r in reliability_values],'.-',label=arm)
        ax.plot([0,1],[0,1],'k--',linewidth=.8)
        ax.set(title=f'CD8 density > {threshold:.4g}',xlabel='Predicted probability',ylabel='Observed fraction')
        ax.legend()
    for i,(arm,m) in enumerate(metrics.items()):
        cd8=m['distributions']['all']['cd8']
        histogram=np.asarray(cd8['positive_pit_histogram'])
        axes[1,0].plot((np.arange(50)+.5)/50,histogram/histogram.sum()*50,label=arm)
        levels=[.5,.9]
        axes[1,1].plot(levels,[cd8['intervals'][str(level)]['coverage'] for level in levels],'.-',label=arm)
        axes[1,2].bar(np.arange(2)+(i-1)*.25,[cd8['intervals'][str(level)]['width'] for level in levels],width=.25,label=arm)
    axes[1,0].axhline(1,color='k',linestyle='--',linewidth=.8)
    axes[1,0].set(xlabel='Positive-branch PIT',ylabel='Histogram density')
    axes[1,1].plot([.5,.9],[.5,.9],'k--',linewidth=.8)
    axes[1,1].set(xlabel='Nominal positive-branch coverage',ylabel='Observed coverage')
    axes[1,2].set(xticks=np.arange(2),xticklabels=['50%','90%'],ylabel='Mean positive-branch interval width')
    for ax in axes[1]:ax.legend()
    save(fig,'cd8_distribution_calibration')
    chosen=['binary_presence','remove_low_intensity_training_q10','add_absent_peak_1e-06','transformed_global_scale_2.0']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,(arm,m) in enumerate(metrics.items()):
        lookup={r['perturbation']:r for r in m['sensitivity']}
        axes[0].bar(np.arange(len(chosen))+(i-1)*.25,[lookup[k]['latent']['relative_rms'] for k in chosen],width=.25,label=arm)
        axes[1].bar(np.arange(len(chosen))+(i-1)*.25,[lookup[k]['presence_mean_absolute_change'] for k in chosen],width=.25,label=arm)
    for ax,ylabel in zip(axes,('Relative latent RMS change','Mean absolute CD8 presence change')):
        ax.set(xticks=np.arange(len(chosen)),xticklabels=['Binary','Drop low','Add near-zero','Scale ×2'],ylabel=ylabel);ax.legend()
    save(fig,'sensitivity')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for row,phase in enumerate(('initial','selected')):
        for arm in metrics:
            prefix='initial_reconstructed' if phase=='initial' and arm=='baseline' else phase
            frame=pd.read_csv(root/'diagnostics'/arm/f'{prefix}_intensity_responses.csv')
            summary=frame.groupby('intensity')[['contribution_norm','derivative_norm']].median()
            for ax,key in zip(axes[row],summary.columns):ax.plot(summary.index,summary[key],label=arm)
        for ax,label in zip(axes[row],('Contribution norm','Derivative norm')):
            ax.set_xscale('symlog',linthresh=1e-7);ax.set_yscale('symlog',linthresh=1e-5)
            lo,_,hi=metrics['baseline']['training_intensity_quantiles']
            ax.axvspan(lo,hi,color='gray',alpha=.15,label='Training 1–99% range')
            ax.set(xlabel='Transformed intensity',ylabel=label,title=phase);ax.legend()
    save(fig,'intensity_response')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in ('l1','l2'):
        path=root/'diagnostics'/arm/'early_diagnostics.json'
        if path.exists():
            observations=json.loads(path.read_text())
            steps=[r['update'] for r in observations]
            color={'l1':'tab:orange','l2':'tab:green'}[arm]
            axes[0].plot(steps,[r['latent_participation_ratio'] for r in observations],'.-',label=arm,color=color)
            axes[1].plot(steps,[r['latent']['centered_over_rms'] for r in observations],'.-',label=arm,color=color)
    initial=metrics['baseline']['initial_centering_controls']['zero_input']['latent']
    for ax,value,label in zip(axes,
            (initial['participation_ratio'],initial['centered_to_total_rms']),
            ('Centered latent participation ratio','Centered / total latent RMS')):
        ax.scatter([0],[value],marker='x',color='black',label='Baseline initialization, reconstructed')
        ax.set(xlabel='Completed optimizer updates',ylabel=label)
        ax.legend(fontsize=8)
    save(fig,'early_observations')
    # Common validation positions, same slide and axes, independent PCA per model.
    with np.load(root/'diagnostics/baseline/row_metadata.npz') as meta:
        slide=meta['slide'];x=meta['x'];y=meta['y']
        common=slide==np.bincount(slide).argmax()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    fig.suptitle('Same validation rows; PC1 signs and color scales are independent across models',fontsize=11)
    for ax,arm in zip(axes,metrics):
        with np.load(root/'diagnostics'/arm/'validation_features.npz') as z:
            values=z['latent'][common].astype(np.float64);values-=values.mean(0)
        _,_,basis=np.linalg.svd(values,full_matrices=False)
        scores=values@basis[0]
        ax.scatter(x[common],y[common],c=scores,s=.5,cmap='coolwarm',rasterized=True)
        ax.set(title=f'{arm}: latent PC1',aspect='equal');ax.invert_yaxis()
    save(fig,'common_slide_latents')
    # Preserve existing raw-embedding displays and add model-appropriate transforms.
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,arm in zip(axes,metrics):
        out=root/'diagnostics'/arm
        path=out/'selected_effective_dictionary.npy'
        if not path.exists():path=sorted(out.glob('selected_conditioned_response_*.npy'))[1]
        values=np.load(path).astype(np.float64);values-=values.mean(0)
        _,_,basis=np.linalg.svd(values,full_matrices=False);points=values@basis[:2].T
        ax.scatter(points[:,0],points[:,1],c=np.arange(len(points)),s=1,cmap='viridis',rasterized=True)
        label=f'response at I={path.stem.split("_")[-1]}' if arm=='baseline' else 'effective dictionary'
        ax.set(title=f'{arm}: {label}',xlabel='PC1',ylabel='PC2')
    save(fig,'transformed_peak_responses')
    return sorted(p.name for p in directory.glob('*.png'))


def variance_component_counts(eigenvalues: np.ndarray) -> dict[str, int]:
    """Count centered principal components needed for 90% and 99% variance."""
    values=np.asarray(eigenvalues,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Covariance eigenvalues must be a nonempty finite nonnegative vector.')
    if values.sum()==0:
        return dict(pcs_90=0,pcs_99=0)
    cumulative=np.cumsum(np.sort(values)[::-1])/values.sum()
    return {f'pcs_{percent}':min(len(values),int(np.searchsorted(cumulative,percent/100))+1)
            for percent in (90,99)}


def checkpoint_summary(history: pd.DataFrame, selected_epoch: int, eigenvalues: np.ndarray) -> dict:
    """Summarize selected validation losses and the minimum over completed epochs.

    IHC uses the target-weighted biology loss before the global biology multiplier;
    H&E uses MSE before its global multiplier. Discriminator loss is cross-entropy.
    """
    chosen=history.loc[history.epoch==selected_epoch]
    if len(chosen)!=1:
        raise ValueError('Selected checkpoint must identify exactly one history epoch.')
    selected=chosen.iloc[0]
    lowest=history.sort_values('epoch').loc[lambda d:np.isfinite(d.validation_cd8_loss)]
    if lowest.empty:
        raise ValueError('Training history has no finite validation CD8 losses.')
    lowest=lowest.loc[lowest.validation_cd8_loss.idxmin()]
    counts=variance_component_counts(eigenvalues)
    return dict(selected_epoch=int(selected_epoch),stopped_epoch=int(history.epoch.max()),
        latent_pcs_90=counts['pcs_90'],latent_pcs_99=counts['pcs_99'],
        cd8_loss=float(selected.validation_cd8_loss),lowest_cd8_loss=float(lowest.validation_cd8_loss),
        lowest_cd8_loss_epoch=int(lowest.epoch),discriminator_loss=float(selected.validation_batch_loss),
        ihc_loss=float(selected.validation_biology_loss),he_mse=float(selected.validation_he_mse))


def compare(root: Path) -> dict:
    """Assemble paired evidence, individual claim verdicts and an offline report."""
    manifest=json.loads((root/'manifest.json').read_text())
    metrics={a:json.loads((root/'diagnostics'/a/'metrics.json').read_text()) for a in ('baseline','l1','l2')}
    intervals=paired_bootstrap(root,manifest,metrics)
    similarity=cross_model_similarity(root,manifest)
    claims=claim_verdicts(metrics,root)
    plots=figures(root,metrics)
    primary=[];efficiency=[];summary_sources={}
    for arm,m in metrics.items():
        run=manifest['runs'][arm]
        history=pd.read_csv(Path(run['checkpoint']).parent/'history.csv')
        checkpoint=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)
        selected=history.loc[history.epoch==checkpoint['epoch']+1].iloc[0]
        summary=checkpoint_summary(history,checkpoint['epoch']+1,m['representations']['latent']['eigenvalues'])
        tissue_path=Path(run['directory'])/'analysis/latent_variance/latent_variance_summary.csv'
        tissue=pd.read_csv(tissue_path)
        if len(tissue)!=1 or not np.isclose(tissue.iloc[0].inference_coverage_fraction,1.):
            raise ValueError('Full-tissue latent variance summary must cover the complete tissue export.')
        summary.update(tissue_latent_pcs_90=int(tissue.iloc[0].pcs_90),
                       tissue_latent_pcs_99=int(tissue.iloc[0].pcs_99))
        summary_sources[arm]=dict(history_sha256=digest_file(Path(run['checkpoint']).parent/'history.csv'),
            checkpoint_sha256=digest_file(Path(run['checkpoint'])),
            tissue_variance_summary=str(tissue_path),tissue_variance_summary_sha256=digest_file(tissue_path))
        primary.append(dict(arm=arm,**summary,collagen_r2=m['probes']['all']['collagen']['r2'],
            tumor_r2=m['probes']['all']['tumor']['r2'],slide_accuracy=m['probes']['all']['slide']['accuracy'],
            cd8_presence_brier=m['distributions']['all']['cd8']['presence']['brier'],
            cd8_presence_auroc=m['distributions']['all']['cd8']['presence']['auroc'],
            ihc_weighted_loss=float(selected['validation_biology_weighted_loss']),
            he_weighted_loss=float(selected['validation_he_weighted_loss']),
            supervised_objective=selected['validation_supervised_objective'],
            batch_aware_objective=selected['validation_batch_aware_objective']))
        efficiency.append(dict(arm=arm,train_seconds=float(history.train_seconds.sum()),
            median_train_epoch_seconds=float(history.loc[history.epoch>1,'train_seconds'].median()),
            median_validation_seconds=float(history.validation_seconds.median()),
            forward_seconds=m['benchmark']['forward']['median_seconds'],
            inference_seconds=m['benchmark']['inference']['median_seconds'],
            forward_backward_seconds=m['benchmark']['forward_backward']['median_seconds'],
            peak_memory_gib=m['benchmark']['forward_backward']['peak_allocated_gib'],
            peak_mlp_rows=m['benchmark']['peak_mlp_input_rows']))
        provenance=Path(run['checkpoint']).parent.parent/'analysis/spatial/spatial_predictions.provenance.json'
        if provenance.exists():
            execution=json.loads(provenance.read_text())['execution']
            efficiency[-1].update(full_tissue_inference_seconds=execution['total_seconds'],
                full_tissue_inference_peak_memory_gib=execution['peak_allocated_vram_gib'])
    frame=pd.DataFrame(primary);frame.to_csv(root/'primary_metrics.csv',index=False)
    cost=pd.DataFrame(efficiency);cost.to_csv(root/'efficiency.csv',index=False)
    strata=[];representation=[];sensitivity=[];controls=[]
    for arm,m in metrics.items():
        for stratum,probes in m['probes'].items():
            cd8=m['distributions'][stratum]['cd8']
            strata.append(dict(arm=arm,stratum=stratum,cd8_valid_rows=cd8['n'],
                stroma_r2=probes['stroma']['r2'],collagen_r2=probes['collagen']['r2'],
                tumor_r2=probes['tumor']['r2'],cd8_loss=cd8['loss'],
                presence_brier=cd8['presence']['brier'],presence_auroc=cd8['presence']['auroc']))
        for kind,values in m['representations'].items():
            representation.append(dict(arm=arm,representation=kind,
                **variance_component_counts(values['eigenvalues']),
                participation_ratio=values['participation_ratio'],centered_to_total_rms=values['centered_to_total_rms'],
                offset_energy_fraction=values['shared_offset_energy_fraction'],
                between_slide_fraction=values['slide']['between_fraction'],
                between_patient_fraction=values['patient']['between_fraction'],
                **{f'{key}_mean_squared_coordinate_correlation':value['mean_squared_correlation']
                   for key,value in values['associations'].items()}))
        for item in m['sensitivity']:
            sensitivity.append(dict(arm=arm,perturbation=item['perturbation'],
                pooled_relative_rms=item['pooled']['relative_rms'],latent_relative_rms=item['latent']['relative_rms'],
                latent_relative_centered_rms=item['latent']['relative_centered_rms'],
                latent_mean_cosine_change=item['latent']['mean_cosine_change'],
                presence_mean_absolute_change=item['presence_mean_absolute_change'],
                **{f'exceedance_{i}_mean_absolute_change':v for i,v in enumerate(item['exceedance_mean_absolute_change'])}))
        for phase,key in [('initial','initial_centering_controls'),('selected','centering_controls')]:
            for mode,representations in m[key].items():
                for kind,value in representations.items():
                    controls.append(dict(arm=arm,phase=phase,centering=mode,representation=kind,
                        participation_ratio=value['participation_ratio'],
                        centered_to_total_rms=value['centered_to_total_rms'],
                        shared_offset_energy_fraction=value['shared_offset_energy_fraction']))
    strata=pd.DataFrame(strata);strata.to_csv(root/'stratified_metrics.csv',index=False)
    representation=pd.DataFrame(representation);representation.to_csv(root/'representation_metrics.csv',index=False)
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(root/'sensitivity_metrics.csv',index=False)
    controls=pd.DataFrame(controls);controls.to_csv(root/'centering_controls.csv',index=False)
    # A recommendation must prioritize biological probes, then calibrated CD8.
    winner=max(('baseline','l1','l2'),key=lambda a:metrics[a]['probes']['all']['stroma']['r2'])
    recommendation=f'{winner} has the highest observed stromal probe R² in this seed. '
    if winner=='baseline':
        recommendation+='Retain occurrence_mlp/count unless the reported robustness and compute gains justify a representation tradeoff.'
    else:
        delta=intervals[(intervals.arm==winner)&(intervals.reference=='baseline')&(intervals.metric=='probe_stroma_r2')].iloc[0]
        if delta.lower>0:
            recommendation+=f'Its paired patient interval for the stromal R² difference versus baseline is [{delta.lower:.4g}, {delta.upper:.4g}]. Consider it alongside collagen, tumor, nuisance and CD8 results; this is evidence from one training seed.'
        else:
            recommendation+='The paired patient interval does not resolve a stromal improvement over baseline. The configuration comparison remains inconclusive.'
    findings=[]
    stromal_deltas=intervals[(intervals.reference=='baseline')&(intervals.metric=='probe_stroma_r2')]
    if winner=='baseline' and len(stromal_deltas)==2 and (stromal_deltas.upper<0).all():
        recommendation=('Retain occurrence_mlp/count with zero_input centering for stromal analysis in this study. '
            'Both dictionary fits have lower held-out stromal probe R², with paired patient intervals entirely below zero. '
            'The current default remains unchanged; dictionary pooling is a compute and sparse-insertion robustness tradeoff.')
    for arm in ('l1','l2'):
        delta=stromal_deltas[stromal_deltas.arm==arm].iloc[0]
        findings.append(f'{arm.upper()} minus baseline stromal R²: {delta.estimate:.5f}, '
            f'95% paired patient interval [{delta.lower:.5f}, {delta.upper:.5f}]. '
            f'Collagen R² is {metrics[arm]["probes"]["all"]["collagen"]["r2"]:.4f} '
            f'and tumor R² is {metrics[arm]["probes"]["all"]["tumor"]["r2"]:.4f}. '
            f'Slide-probe accuracy remains {metrics[arm]["probes"]["all"]["slide"]["accuracy"]:.1%}; '
            'these representations still contain substantial slide information.')
    collagen_deltas=intervals[(intervals.reference=='baseline')&(intervals.metric=='probe_collagen_r2')]
    if ((collagen_deltas.lower<=0)&(collagen_deltas.upper>=0)).all():
        findings.append('Neither dictionary-versus-baseline collagen R² difference is resolved by the paired patient intervals.')
    cd8=intervals[(intervals.reference=='baseline')&intervals.metric.isin(
        ['prediction_cd8_loss','prediction_presence_brier'])]
    if len(cd8)==4 and (cd8.lower>0).all():
        findings.append('Both dictionary arms have worse CD8 loss and presence Brier scores than baseline, '
            'with paired intervals excluding zero. The full tables include AUROC, fixed-threshold exceedance '
            'calibration, positive-branch PIT, coverage and width; interval coverage alone does not establish better distributions.')
    relative_l1=intervals[(intervals.arm=='l2')&(intervals.reference=='l1')]
    l1_better_cd8=relative_l1[relative_l1.metric.isin(['prediction_cd8_loss','prediction_presence_brier'])]
    l1_better_collagen=relative_l1[relative_l1.metric=='probe_collagen_r2']
    stromal_pair=relative_l1[relative_l1.metric=='probe_stroma_r2'].iloc[0]
    if (l1_better_cd8.lower>0).all() and (l1_better_collagen.upper<0).all():
        findings.append('If dictionary pooling is chosen for compute or sparse-insertion behavior, prefer L1 in this experiment: '
            'it has better CD8 loss, presence Brier and collagen probe R² than L2 with paired intervals excluding zero. '
            f'The L2-minus-L1 stromal R² interval [{stromal_pair.lower:.5f}, {stromal_pair.upper:.5f}] '
            'does not resolve their stromal difference.' if stromal_pair.lower<=0<=stromal_pair.upper else
            'L1 has better CD8 loss, presence Brier and collagen probe R² than L2 with paired intervals excluding zero; '
            'consider the stromal comparison separately.')
    changes={arm:{v['perturbation']:v for v in m['sensitivity']} for arm,m in metrics.items()}
    for arm in ('l1','l2'):
        speed=metrics['baseline']['benchmark']['forward_backward']['median_seconds']/metrics[arm]['benchmark']['forward_backward']['median_seconds']
        base_memory=metrics['baseline']['benchmark']['forward_backward']['peak_allocated_gib']
        arm_memory=metrics[arm]['benchmark']['forward_backward']['peak_allocated_gib']
        memory_text=(f'{base_memory/arm_memory:.2f}× lower peak allocated memory' if base_memory and arm_memory
                     else 'GPU peak memory unavailable for this workload')
        near=changes[arm]['add_absent_peak_1e-06']['presence_mean_absolute_change']
        low=changes[arm]['remove_low_intensity_training_q10']['presence_mean_absolute_change']
        findings.append(f'{arm.upper()} gives a {speed:.2f}× matched forward/backward speedup; {memory_text}. '
            f'Adding one intensity-10⁻⁶ absent peak changes CD8 presence by {near:.3g} on average; '
            f'removing training-bottom-decile intensities changes it by {low:.4f}. '
            'These are separate perturbations, not a uniform robustness score.')
    findings.append('Production timings show smaller gains than the isolated benchmark: median training epochs take '
        + ', '.join(f'{item["arm"]} {item["median_train_epoch_seconds"]:.2f}s' for item in efficiency)
        + '. Full-tissue export times are '
        + ', '.join(f'{item["arm"]} {item["full_tissue_inference_seconds"]:.1f}s' for item in efficiency
                    if 'full_tissue_inference_seconds' in item)
        + '. Baseline production times are historical; the fresh common-row benchmark is the controlled workload comparison.')
    report=dict(primary=primary,efficiency=efficiency,recommendation=recommendation,
        primary_sources=summary_sources,
        primary_definitions=dict(loss_split='validation',lowest_cd8_loss='Minimum validation CD8 loss over every completed epoch; earliest epoch breaks ties.',
            ihc_loss='Configured target-weighted biology loss before the global biology multiplier.',
            he_mse='H&E MSE before the global H&E multiplier.',discriminator_loss='Selected-checkpoint discriminator cross-entropy.',
            latent_pcs='Number of centered principal components explaining at least 90% or 99% of total latent variance; validation and full-tissue counts are separate.'),
        findings=findings,
        scope='Single selected training seed. Patient bootstrap conditions on fitted models and does not cover training-seed uncertainty. H&E-to-Visium transfer is separate.',
        thresholds=manifest['cd8_thresholds'],stroma_quartiles=manifest['stroma_quartiles'])
    peak_section = ''
    peak_completion = root/'peak_comparison/completion.json'
    if peak_completion.exists() and json.loads(peak_completion.read_text())['status']=='complete':
        peak_summary = json.loads((root/'peak_comparison/summary.json').read_text())
        report['peak_comparison'] = dict(report='peak_comparison/report.html', summary=peak_summary,
            completion_sha256=digest_file(peak_completion))
        peak_section = ('<section><h2>Peak-vector quality and family correlations</h2>'
            '<p>The extended comparison covers raw and effective vectors, matched family memberships, '
            'held-out IHC associations with abundance controls, biological neighborhood coherence, '
            'rank, cluster separation and neighbor hubness.</p>'
            '<p><a href="peak_comparison/report.html">Open the peak-vector and family comparison</a></p></section>')
    save_json(root/'comparison.json',report)
    def table(df):return df.to_html(index=False,float_format=lambda x:f'{x:.5g}',escape=True,border=0)
    loss_table=frame[['arm','selected_epoch','stopped_epoch','cd8_loss','lowest_cd8_loss',
        'lowest_cd8_loss_epoch','discriminator_loss','ihc_loss','he_mse']].rename(columns={
        'selected_epoch':'Chosen epoch','stopped_epoch':'Stopped epoch','cd8_loss':'Chosen CD8 loss',
        'lowest_cd8_loss':'Lowest CD8 loss','lowest_cd8_loss_epoch':'Lowest CD8 epoch',
        'discriminator_loss':'Chosen discriminator loss','ihc_loss':'Chosen IHC loss','he_mse':'Chosen H&E loss (MSE)'})
    variance_table=frame[['arm','latent_pcs_90','latent_pcs_99','tissue_latent_pcs_90','tissue_latent_pcs_99']].rename(columns={
        'latent_pcs_90':'Validation PCs 90%','latent_pcs_99':'Validation PCs 99%',
        'tissue_latent_pcs_90':'Full-tissue PCs 90%','tissue_latent_pcs_99':'Full-tissue PCs 99%'})
    supporting_table=frame[['arm','collagen_r2','tumor_r2','slide_accuracy','cd8_presence_brier',
        'cd8_presence_auroc','ihc_weighted_loss','he_weighted_loss','supervised_objective','batch_aware_objective']]
    links=[]
    verification_link = ('<li><a href="validation/verification.json">Implementation verification evidence</a></li>'
                         if (root/'validation/verification.json').exists() else '')
    for arm in metrics:
        run=manifest['runs'][arm]
        analysis=Path(run['checkpoint']).parent.parent/'analysis'
        import os
        relative=html.escape(os.path.relpath(analysis,root))
        run_relative=html.escape(os.path.relpath(run['directory'],root))
        checkpoint_relative=html.escape(os.path.relpath(run['checkpoint'],root))
        links.append(f'<li>{arm}: <a href="{relative}/latent/sample_umap.html">interactive latent view</a>; '
                     f'<a href="{relative}/peaks/umap/peak_umap.html">raw peak-embedding view</a>; '
                     f'<a href="{relative}">all configured downstream artifacts</a>; '
                     f'<a href="{run_relative}/analysis/latent_variance">full-tissue latent variance</a>; '
                     f'<a href="{checkpoint_relative}">selected checkpoint</a>; '
                     f'<a href="configs/{arm}.yaml">run configuration</a>; '
                     f'<a href="diagnostics/{arm}/metrics.json">all metrics and strata</a>; '
                     +(f'<a href="diagnostics/{arm}/early_diagnostics.json">early observations</a>' if arm!='baseline'
                       else 'historical early-update observations unavailable')+'</li>')
    body=''.join(f'<figure><img src="figures/{p}" alt="{p}"><figcaption><a href="figures/{p[:-4]}.pdf">PDF</a></figcaption></figure>' for p in plots)
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>PDAC dictionary pooling comparison</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:2em auto;padding:0 1em;color:#17212b}}table{{border-collapse:collapse;display:block;overflow-x:auto}}td,th{{padding:.5em;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}a{{color:#145a96}}section{{margin:2em 0}}</style>
<h1>Dictionary pooling: L1, L2 and historical occurrence baseline</h1>
<p>{html.escape(recommendation)}</p><p>{html.escape(report['scope'])}</p>
{''.join(f'<p>{html.escape(finding)}</p>' for finding in findings)}
<p>Training: 1,528,567 rows; validation: all 169,861 rows. Probes fit on the same 65,536 proportionally slide-stratified training rows. All scaling, CD8 quartile thresholds and stromal quartiles are frozen from training. No abundance channel was added.</p>
<p>CD8 thresholds on the normalized density scale: {html.escape(str(manifest['cd8_thresholds']))}. Interval coverage and width refer to the conditional positive branch.</p>
<section><h2>Chosen and lowest training-history losses</h2><p>Every loss below is a validation metric. Chosen means the selected checkpoint; lowest CD8 means the minimum over all completed epochs, which need not be the checkpoint selected by the batch-aware objective. IHC is the configured target-weighted biology loss before its global multiplier; H&E is MSE before its global multiplier. Discriminator loss is cross-entropy (larger loss is harder domain classification, not necessarily poorer representation).</p>{table(loss_table)}</section>
<section><h2>Latent variance: 90% and 99%</h2><p>Entries are the number of centered principal components needed to explain at least 90% or 99% of total variance in the 256-dimensional exported latent representation at the chosen checkpoint. Validation uses all 169,861 rows; full tissue uses all 5,893,291 rows. Larger counts alone do not establish biological improvement.</p>{table(variance_table)}</section>
<section><h2>Supporting representation and prediction metrics</h2>{table(supporting_table)}</section>
<section><h2>Representation variation and nuisance associations</h2><p>Offset energy, centered rank and group variation measure different properties. Coordinate correlations are descriptive and basis-dependent; the slide probe above measures decodable slide information.</p>{table(representation)}</section>
<section><h2>Controlled centering comparisons</h2><p>Weights and input spectra are held fixed within each comparison. These forward controls do not identify the causal effect of centering on training. Above the normalization floor, L1 dictionary centering subtracts the same dictionary mean from each nonempty pooled row; L2 subtracts that mean multiplied by the row-specific L1/L2 intensity-norm ratio. The L2 control therefore changes more than a shared constant offset.</p>{table(controls)}</section>
<section><h2>Robustness on common rows</h2><p>All changes use the same 2,048 validation rows. Intensity scaling is distinguished before and after log1p; low-intensity removal uses the training tenth percentile. Relative changes and cosine changes avoid interpreting L2 scale alone as sensitivity. Binary replacement and global scaling measure dependence on intensity; smaller changes are not automatically preferable. Low-intensity measured peaks have not been established to be noise.</p>{table(sensitivity)}</section>
<section><h2>Slide and stromal strata</h2><p>Stromal quartiles are defined from training targets. Stratum R² may be small or negative when its target variance is narrow, and is undefined for constant targets.</p>{table(strata)}</section>
<section><h2>Paired patient intervals</h2><p>Differences are arm minus reference, with 95% percentile intervals from 1,000 paired patient resamples. Larger R²/AUROC and smaller losses/Brier are preferable. These are conditional validation comparisons, not independent external validation.</p>{table(intervals)}</section>
<section><h2>Efficiency</h2><p>Matched 2,048-row resident workloads; forward/backward includes all heads and no optimizer update. Full epoch medians omit the first epoch with observational diagnostics. Baseline epoch and full-tissue times come from historical logs; the common-row benchmarks are measured afresh for all models. MLP row-count reductions are distinct from measured speedups.</p>{table(cost)}</section>
<section><h2>Cross-model neighborhood structure</h2>{table(similarity)}</section>{peak_section}{body}
<section><h2>Individual claim verdicts</h2><p>Mathematical properties, measured behavior and untested causal claims have separate verdicts.</p>{table(claims)}</section>
<section><h2>Provenance and artifacts</h2><ul><li><a href="manifest.json">Immutable experiment manifest</a></li>{verification_link}<li><a href="primary_metrics.csv">Primary metrics CSV</a></li><li><a href="stratified_metrics.csv">Stratified metrics CSV</a></li><li><a href="representation_metrics.csv">Representation metrics CSV</a></li><li><a href="sensitivity_metrics.csv">Sensitivity metrics CSV</a></li><li><a href="centering_controls.csv">Centering controls CSV</a></li><li><a href="paired_patient_bootstrap.csv">Paired intervals CSV</a></li><li><a href="claim_verdicts.csv">Claim verdicts CSV</a></li>{''.join(links)}</ul></section></html>'''
    (root/'report.html').write_text(document)
    save_json(root/'stages/comparison.json',dict(status='complete',report=str(root/'report.html'),
        metrics_sha256={a:digest_file(root/'diagnostics'/a/'metrics.json') for a in metrics}))
    return report
