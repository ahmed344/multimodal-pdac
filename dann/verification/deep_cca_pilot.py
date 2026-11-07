"""Budgeted, matched Deep CCA pilots; outputs never overwrite production runs.

Run with --output NEW_DIRECTORY. Training plus validation fit time is debited
against the cumulative budget; extraction/probes are within the overall window.
The common 160/320-update horizons, subsets and scientific settings are frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import ndtr
from scipy.stats import kstest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dann.batch_alignment import sample_rows
from dann.checkpoints import checkpoint_config, validate_checkpoint
from dann.config import load_config, resolve_execution, seed_everything
from dann.data_loader import create_data_bundle, stratified_split_indices, stratified_cap_indices
from dann.deep_cca import evaluate_cca
from dann.execution import PreparedLoader, device_batches
from dann.initialization import initialize_supervised_heads
from dann.losses import ZILNLoss
from dann.lr_scheduler import UpdateWarmup
from dann.model import AdversarialLatentFusion
from dann.targets import CD8
from dann.train import (TrainingProgress, capture_training_rng, restore_training_rng,
                        checkpoint_metadata, checkpoint_payload, run_epoch)
from dann.training_artifacts import cpu_snapshot, training_state, atomic_json
from dann.verification.alignment_study_metrics import LIMITS, variance_spectrum, geometry
from dann.verification.dictionary_comparison import close, parameter_hash
from dann.verification.pooling_diagnostics import binary_metrics, distribution_metrics, json_clean

ARMS = [('no_he', None, 0.), ('raw_mse', None, None)] + [
    (f'cca_d{d}_w{str(w).replace(".", "p")}', d, w)
    for d in (16, 32) for w in (.015, .05, .15)]
PROBES = [('tumor', 'mse'), ('collagen', 'mse'), ('stroma', 'mse')] + [
    (n, 'brier') for n in ('presence', 'exceedance_0', 'exceedance_1', 'exceedance_2')]


def save(path: Path, value) -> None:
    """Write portable JSON atomically, preserving unavailable results as null."""
    atomic_json(path, json_clean(value))


def arm_config(source: dict, root: Path, arm: tuple, seed: int) -> dict:
    """Hold scientific settings fixed except the requested H&E objective."""
    c = copy.deepcopy(source)
    name, d, weight = arm
    c['data'].update(max_train_samples=65536, max_validation_samples=16384)
    c['model']['heads']['he'].update(enabled=name != 'no_he', objective='deep_cca' if d else 'raw_mse')
    if d:
        c['model']['heads']['he']['deep_cca'] = dict(dimension=d, msi_hidden_dim=128,
            he_hidden_dim=256, biology_only_epochs=1, ramp_epochs=2)
    if weight is not None:
        c['loss']['he_weight'] = weight
    c['training'].update(epochs=20, model_seed=seed, early_stopping_enabled=False,
                         resume_checkpoint=None, archive_checkpoint_every=10)
    c['training']['lr_scheduler']['enabled'] = False
    c['post_training']['enabled'] = False
    c['analysis']['export_he_predictions'] = False
    c['results'].update(root=str(root/'runs'), run_name=f'{name}_seed{seed}', smoke=False)
    c['monitoring'].update(enabled=True, gradient_updates_per_epoch=2)
    return resolve_execution(c)


def probe_scores(predictions: dict, y, mask, thresholds, selected) -> dict:
    result = {}
    for j, name in ((1, 'tumor'), (2, 'collagen'), (3, 'stroma')):
        take = mask[:, j] & selected
        result[name] = dict(mse=float(np.mean((predictions[name][take] - y[take, j])**2)))
    for name, threshold in [('presence', 0.), *[(f'exceedance_{i}', t) for i, t in enumerate(thresholds)]]:
        take = mask[:, 0] & selected
        result[name] = binary_metrics(y[take, 0] > threshold, predictions[name][take])
    return result


def primary_score(candidate: dict, control: dict) -> float:
    """Mean of seven error ratios in the training-defined upper stromal half."""
    return float(np.mean([candidate['stromal_probes'][name][metric] /
                         max(control['stromal_probes'][name][metric], 1e-8) for name, metric in PROBES]))


def preservation(candidate: dict, control: dict) -> dict:
    """Apply the existing biological/CD8 bounds and stromal neighbor constraint."""
    checks = [dict(metric='noncollapsed', passed=candidate['spectrum']['noncollapsed'])]
    def bound(name, value, ref, allowance):
        checks.append(dict(metric=name, value=value, reference=ref, allowance=allowance,
            passed=bool(value is not None and ref is not None and np.isfinite(value) and
                        np.isfinite(ref) and value <= ref + allowance + 1e-12)))
    for scope in ('probes', 'stromal_probes'):
        for name, metric in PROBES:
            ref = control[scope][name][metric]
            allowance = LIMITS['probe_mse_relative'] * abs(ref) if metric == 'mse' else LIMITS['probe_brier_absolute']
            bound(f'{scope}_{name}', candidate[scope][name][metric], ref, allowance)
    c, r = candidate['distributions']['cd8'], control['distributions']['cd8']
    bound('cd8_loss', c['loss'], r['loss'], LIMITS['cd8_loss_relative']*abs(r['loss']))
    bound('presence_brier', c['presence']['brier'], r['presence']['brier'], LIMITS['brier_absolute'])
    for threshold in r['exceedance']:
        bound('exceedance_' + threshold, c['exceedance'][threshold]['brier'], r['exceedance'][threshold]['brier'], LIMITS['brier_absolute'])
    bound('pit_ks', c['positive_pit_ks'], r['positive_pit_ks'], LIMITS['pit_ks_absolute'])
    for level in (.5, .9):
        bound(f'coverage_error_{level}', abs(c['intervals'][str(level)]['coverage'] - level),
              abs(r['intervals'][str(level)]['coverage'] - level), LIMITS['coverage_error_absolute'])
    bound('stromal_neighbor_excess', candidate['stromal_neighbors']['excess'],
          control['stromal_neighbors']['excess'], .01)
    return dict(eligible=all(x['passed'] for x in checks), checks=checks)


def rank(records: list[dict], control: dict) -> list[dict]:
    """Resolve primary-score ties within .01 by CD8 loss, weight, dimension."""
    remaining = [r for r in records if r['dimension'] and preservation(r, control)['eligible']]
    ordered = []
    while remaining:
        minimum = min(primary_score(r, control) for r in remaining)
        tied = [r for r in remaining if primary_score(r, control) <= minimum + .01 + 1e-12]
        winner = min(tied, key=lambda r: (r['distributions']['cd8']['loss'], r['weight'], r['dimension']))
        ordered.append(winner)
        remaining = [r for r in remaining if r['id'] != winner['id']]
    return ordered


class Pilot:
    """Own shared immutable data, frozen subsets, cumulative timing and fit state."""
    def __init__(self, root: Path, source: Path, budget_seconds: float, overall_seconds: float):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.overall_deadline = self.started + overall_seconds
        self.budget, self.spent = budget_seconds, 0.
        self.ledger = []
        self.source = load_config(source)
        if (self.source['model']['latent_dim'] != 256 or self.source['loss']['biology_weight'] != 1.
                or self.source['loss']['batch_alignment']['nhsic']['weight'] != .01
                or self.source['training']['batch_size'] != 4096
                or self.source['training']['validation_batch_size'] != 8192
                or self.source['training']['learning_rate'] != .001
                or self.source['training']['early_stopping_metric'] != 'total_loss'):
            raise ValueError('Pilot source must match the validated 256-width baseline.')
        self.seed = self.source['training']['seed']
        for folder in ('configs', 'checkpoints', 'metrics', 'source'):
            (self.root/folder).mkdir()
        (self.root/'source_config.yaml').write_text(yaml.safe_dump(self.source, sort_keys=False))
        sources = {}
        for path in sorted(Path('dann').rglob('*.py')):
            dest = self.root/'source'/path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        c = arm_config(self.source, self.root, ARMS[3], self.seed)
        self.bundle = create_data_bundle(c, requested_splits=('train', 'validation'))
        self.initial_rng = capture_training_rng(self.bundle.loaders)
        self.steps = len(self.bundle.loaders['train'])
        if self.steps != 16:
            raise ValueError('The preregistered 160/320 horizons require 65,536 available training rows.')
        splits = self.bundle.split_indices
        np.savez_compressed(self.root/'splits.npz', **splits)
        m = self.bundle.metadata
        full = stratified_split_indices(m.batch_codes, self.source['data']['train_fraction'],
            self.source['data']['validation_fraction'], self.source['data']['test_fraction'], self.seed)
        tr = full['train']
        self.thresholds = np.quantile(m.targets[tr, 0][m.target_valid_mask[tr, 0] & (m.targets[tr, 0] > 0)], [.25, .5, .75]).tolist()
        self.median = float(np.median(m.targets[tr, 3][m.target_valid_mask[tr, 3]]))
        va = splits['validation']
        stromal = va[m.target_valid_mask[va, 3] & (m.targets[va, 3] >= self.median)]
        self.neighbor_rows = np.sort(stratified_cap_indices(stromal, m.batch_codes, 4096, self.seed + 999))
        np.save(self.root/'stromal_neighbor_rows.npy', self.neighbor_rows)
        self.manifest = dict(training_budget_seconds=budget_seconds, overall_seconds=overall_seconds,
            seed=self.seed, confirmation_seed=self.seed + 1, horizons=[160, 320],
            training_rows=len(splits['train']), validation_rows=len(va),
            split_hashes={k: hashlib.sha256(v.astype('<i8').tobytes()).hexdigest() for k, v in splits.items()},
            full_split_hashes={k: hashlib.sha256(v.astype('<i8').tobytes()).hexdigest() for k, v in full.items()},
            cd8_thresholds=self.thresholds, stromal_median=self.median,
            primary='mean seven stromal probe error ratios to no_he; denominator floor 1e-8',
            preservation_limits=LIMITS, stromal_neighbor_allowance=.01,
            raw_mse_weight=self.source['loss']['he_weight'], sources=sources,
            timing='train and validation epochs; startup, artifacts and probes excluded',
            initialization='shared encoder/biology hashes within seed; separately seeded CCA projectors',
            probe_protocol='training StandardScaler; Ridge alpha=1; logistic C=1, lbfgs max_iter=5000')
        save(self.root/'manifest.json', self.manifest)
        self.hashes = {}
        self.records = []

    def config(self, arm: tuple, seed: int) -> dict:
        c = arm_config(self.source, self.root, arm, seed)
        (self.root/'configs'/f'{arm[0]}_seed{seed}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
        return c

    def fit(self, arm: tuple, seed: int, horizon: int) -> Path | None:
        c = self.config(arm, seed)
        identifier = f'{arm[0]}_seed{seed}'
        directory = self.root/'checkpoints'/identifier
        directory.mkdir(exist_ok=True)
        latest = directory/'latest.pt'
        restore_training_rng(self.initial_rng, self.bundle.loaders)
        seed_everything(seed)
        model = AdversarialLatentFusion.from_config(c, len(self.bundle.metadata.batch_names), 4).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=c['training']['learning_rate'],
            weight_decay=c['training']['weight_decay'], betas=(c['training']['beta1'], c['training']['beta2']))
        warmup = UpdateWarmup(opt, c, self.steps)
        progress = TrainingProgress()
        history = []
        if latest.exists():
            checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            validate_checkpoint(checkpoint, c, model, optimizer=True)
            model.load_state_dict(checkpoint['model_state'])
            opt.load_state_dict(checkpoint['optimizer_state'])
            warmup.load_state_dict(checkpoint['warmup_state'])
            progress = TrainingProgress(**checkpoint['training_progress'])
            restore_training_rng(checkpoint['training_rng_state'], self.bundle.loaders)
            history = pd.read_csv(directory/'history.csv').to_dict('records')
            initialization = checkpoint['head_initialization']
        else:
            initialization = initialize_supervised_heads(model, self.bundle, c)
            hashes = dict(encoder=parameter_hash(model.encoder), biology=parameter_hash(model.biology_predictor))
            if seed in self.hashes and self.hashes[seed] != hashes:
                raise ValueError('Shared initialization mismatch.')
            self.hashes[seed] = hashes
            save(self.root/f'initialization_seed{seed}.json', hashes)
            seed_everything(seed)  # Identical initial dropout stream, independent of head allocation.
        metadata = checkpoint_metadata(model, c, self.bundle)
        metadata['head_initialization'] = initialization
        rows = self.bundle.split_indices['validation']
        selected = sample_rows(rows, self.bundle.metadata.batch_codes[rows],
            c['loss']['batch_alignment']['validation_per_slide'], self.seed)
        loss = ZILNLoss.from_config(c).cuda()
        common = dict(model=model, ziln_loss=loss, device=torch.device('cuda'),
            biology_weight=c['loss']['biology_weight'], batch_weight=c['loss']['batch_weight'],
            class_weights=self.bundle.batch_class_weights.cuda() if c['loss']['balance_batch_classes'] else None,
            total_epochs=20, grl_schedule=c['training']['grl_schedule'],
            grl_max_lambda=c['training']['grl_max_lambda'], grl_gamma=c['training']['grl_gamma'],
            target_columns=c['data']['target_columns'], cd8_index=c['data']['target_columns'].index(CD8),
            he_weight=c['loss']['he_weight'], monitoring=c['monitoring'],
            batch_alignment=c['loss']['batch_alignment'], alignment_seed=self.seed,
            validation_alignment_rows=selected)
        for epoch in range(progress.epoch, horizon // self.steps):
            remaining = min(self.budget - self.spent, self.overall_deadline - time.perf_counter())
            previous = max([r['seconds'] for r in self.ledger if r['arm'] == identifier], default=60.)
            if remaining < max(10., 1.25 * previous):
                print(f'Budget stopped {identifier} before epoch {epoch + 1}', flush=True)
                return None
            started = time.perf_counter()
            try:
                train = run_epoch(loader=self.bundle.loaders['train'], optimizer=opt, epoch_index=epoch,
                    gradient_clip_norm=c['training']['gradient_clip_norm'], progress_state=progress,
                    warmup=warmup, deadline=started + remaining - min(10., remaining/4), **common)
                if train['optimizer_updates'] != self.steps or progress.update_in_epoch:
                    return None  # Never publish a partial horizon as complete.
                valid = run_epoch(loader=self.bundle.loaders['validation'], optimizer=None, epoch_index=epoch,
                    gradient_clip_norm=None, **common)
            finally:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                self.spent += elapsed
                self.ledger.append(dict(arm=identifier, epoch=epoch+1, seconds=elapsed))
                save(self.root/'budget.json', dict(budget_seconds=self.budget, spent_seconds=self.spent, epochs=self.ledger))
            history.append(dict(epoch=epoch+1, **{f'train_{k}': v for k,v in train.items()},
                                **{f'validation_{k}': v for k,v in valid.items()}))
            pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
            print(f'{identifier} epoch={epoch+1} updates={progress.optimizer_updates} '
                  f'fit_seconds={elapsed:.2f} budget={self.spent:.1f}/{self.budget:.1f} '
                  f'validation_total={valid["total_loss"]:.6f}', flush=True)
            if (epoch+1) * self.steps in (160, 320):
                state = cpu_snapshot(training_state(model, opt, None, warmup))
                payload = checkpoint_payload(metadata, state, epoch, valid['total_loss'],
                    best_epoch=epoch+1, progress_state=progress)
                payload['training_rng_state'] = capture_training_rng(self.bundle.loaders)
                path = directory/f'update_{progress.optimizer_updates:04d}.pt'
                torch.save(payload, path)
                torch.save(payload, latest)
        del model, opt
        torch.cuda.empty_cache()
        return directory/f'update_{horizon:04d}.pt'

    def extract(self, model, c, split: str, paired: bool = False) -> dict:
        source = self.bundle.loaders[split]
        dataset = self.bundle.datasets[split]
        loader = DataLoader(dataset, batch_size=8192, shuffle=False, collate_fn=source.collate_fn,
                            generator=torch.Generator().manual_seed(self.seed))
        loader = PreparedLoader(loader, c['performance'], c['model'], validation=True)
        arrays = {k: [] for k in ('rows', 'latent', 'pi_logits', 'mu', 'sigma')}
        with torch.inference_mode():
            for cpu, batch in device_batches(loader, torch.device('cuda')):
                output = model(batch, grl_strength=0., predict_he=paired)
                arrays['rows'].append(batch['row_ids'].cpu().numpy())
                for key in ('latent', 'pi_logits', 'mu', 'sigma', 'cca_msi', 'cca_he'):
                    if key in output:
                        arrays.setdefault(key, []).append(output[key].cpu().numpy())
        values = {k: np.concatenate(v) for k, v in arrays.items()}
        order = np.argsort(values['rows'])
        return {k: v[order] for k, v in values.items()}

    def evaluate(self, arm: tuple, seed: int, horizon: int, path: Path) -> dict:
        started = time.perf_counter()
        identifier = f'{arm[0]}_seed{seed}_u{horizon}'
        out = self.root/'metrics'/identifier
        out.mkdir(exist_ok=True)
        cp = torch.load(path, map_location='cpu', weights_only=False)
        c = checkpoint_config(cp)
        model = AdversarialLatentFusion.from_config(c, len(self.bundle.metadata.batch_names), 4).cuda().eval()
        model.load_state_dict(cp['model_state'])
        valid = self.extract(model, c, 'validation', paired=bool(arm[1]))
        train = self.extract(model, c, 'train')
        del model
        m = self.bundle.metadata
        tr, va = train['rows'], valid['rows']
        y, mask = m.targets[va], m.target_valid_mask[va]
        scaler = StandardScaler().fit(train['latent'])
        x, v = (scaler.transform(a['latent']).astype(np.float64) for a in (train, valid))
        predictions = {}
        params = dict(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        for j, name in ((1, 'tumor'), (2, 'collagen'), (3, 'stroma')):
            fitted = Ridge(alpha=1.).fit(x[m.target_valid_mask[tr, j]], m.targets[tr, j][m.target_valid_mask[tr, j]])
            predictions[name] = fitted.predict(v)
            params[name+'_coef'], params[name+'_intercept'] = fitted.coef_, fitted.intercept_
        for name, threshold in [('presence', 0.), *[(f'exceedance_{i}', t) for i,t in enumerate(self.thresholds)]]:
            take = m.target_valid_mask[tr, 0]
            fitted = LogisticRegression(C=1., solver='lbfgs', max_iter=5000, random_state=self.seed).fit(x[take], m.targets[tr, 0][take] > threshold)
            if fitted.n_iter_.max() >= 5000:
                raise ValueError('Probe did not converge.')
            predictions[name] = fitted.predict_proba(v)[:, 1]
            params[name+'_coef'], params[name+'_intercept'] = fitted.coef_, fitted.intercept_
        np.savez_compressed(out/'probe_parameters.npz', **params)
        np.savez_compressed(out/'probe_predictions.npz', rows=va, **predictions)
        distributions = distribution_metrics(valid, y, mask, self.thresholds, c['loss']['logit_epsilon'])
        positive = mask[:, 0] & (y[:, 0] > 0)
        obs = np.clip(y[positive, 0].astype(float), c['loss']['logit_epsilon'], 1-c['loss']['logit_epsilon'])
        pit = ndtr((np.log(obs/(1-obs))-valid['mu'][positive, 0])/valid['sigma'][positive, 0])
        distributions['cd8']['positive_pit_ks'] = float(kstest(pit, 'uniform').statistic)
        positions = np.searchsorted(va, self.neighbor_rows)
        np.testing.assert_array_equal(va[positions], self.neighbor_rows)
        neighbors, per_row = geometry(valid['latent'][positions], m.batch_codes[self.neighbor_rows])
        per_row.to_csv(out/'stromal_neighbors.csv', index=False)
        record = dict(id=identifier, arm=arm[0], dimension=arm[1], weight=c['loss']['he_weight'],
            seed=seed, horizon=horizon, checkpoint=str(path),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            spectrum=variance_spectrum(valid['latent']), distributions=distributions,
            probes=probe_scores(predictions, y, mask, self.thresholds, np.ones(len(va), bool)),
            stromal_probes=probe_scores(predictions, y, mask, self.thresholds, mask[:, 3] & (y[:, 3] >= self.median)),
            stromal_neighbors=neighbors, evaluation_seconds=time.perf_counter()-started)
        if arm[1]:
            np.savez_compressed(out/'cca_projections.npz', rows=va, slides=m.batch_codes[va],
                                msi=valid['cca_msi'], he=valid['cca_he'])
            record['cca'] = evaluate_cca(torch.tensor(valid['cca_msi']), torch.tensor(valid['cca_he']),
                                         torch.tensor(m.batch_codes[va])).diagnostics
        save(out/'metrics.json', record)
        self.records.append(record)
        self.report()
        print(f'Evaluated {identifier} in {record["evaluation_seconds"]:.1f}s', flush=True)
        return record

    def report(self, decision: dict | None = None) -> None:
        rows = []
        for r in self.records:
            controls = [c for c in self.records if c['arm'] == 'no_he' and c['seed'] == r['seed'] and c['horizon'] == r['horizon']]
            row = dict(id=r['id'], arm=r['arm'], seed=r['seed'], updates=r['horizon'],
                       dimension=r['dimension'], weight=r['weight'], cd8_loss=r['distributions']['cd8']['loss'],
                       stromal_neighbor_excess=r['stromal_neighbors']['excess'],
                       cca_correlation=r.get('cca', {}).get('correlation'))
            if controls:
                row.update(probe_error_ratio=primary_score(r, controls[0]), eligible=preservation(r, controls[0])['eligible'])
                save(self.root/'metrics'/r['id']/'preservation.json', preservation(r, controls[0]))
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(self.root/'comparison.csv', index=False)
        if decision is not None:
            save(self.root/'decision.json', decision)
        text = '<html><head><meta charset="utf-8"><title>Deep CCA pilots</title></head><body><h1>Deep CCA pilots</h1>'
        text += f'<p>Fit time: {self.spent:.1f} / {self.budget:.1f} seconds. Elapsed: {time.perf_counter()-self.started:.1f} seconds.</p>'
        text += '<p>Matched 65,536/16,384 rows; compare only common completed update horizons. Primary: mean seven probe error ratios in the training-defined upper stromal half. Lower is better. This bounded study does not establish global optimality.</p>'
        text += frame.to_html(index=False, float_format=lambda x: f'{x:.6g}')
        if decision:
            import html
            text += '<pre>' + html.escape(json.dumps(json_clean(decision), indent=2)) + '</pre>'
        text += '</body></html>'
        (self.root/'comparison.html').write_text(text)

    def permutations(self, winner: dict) -> dict:
        data = np.load(self.root/'metrics'/winner['id']/'cca_projections.npz')
        u, v, slides = (torch.tensor(data[k]) for k in ('msi', 'he', 'slides'))
        observed = evaluate_cca(u, v, slides).diagnostics['correlation']
        scores = []
        for index in range(20):
            generator = torch.Generator().manual_seed(self.seed + 10000 + index)
            shuffled = v.clone()
            for slide in slides.unique():
                positions = torch.where(slides == slide)[0]
                shuffled[positions] = v[positions[torch.randperm(len(positions), generator=generator)]]
            scores.append(evaluate_cca(u, shuffled, slides).diagnostics['correlation'])
        result = dict(id=winner['id'], observed=observed, permuted=scores,
            empirical_p=(1+sum(s >= observed for s in scores))/21,
            seeds=list(range(self.seed+10000, self.seed+10020)))
        save(self.root/'pairing_permutations.json', result)
        return result

    def run(self) -> None:
        completed = {}
        for horizon in (160, 320):
            for arm in ARMS:
                path = self.fit(arm, self.seed, horizon)
                if path is None:
                    break
                completed[(arm[0], horizon)] = path
            if not all((a[0], horizon) in completed for a in ARMS):
                break
        common = max((h for h in (160, 320) if all((a[0], h) in completed for a in ARMS)), default=None)
        if common is None:
            self.report(dict(promote=False, reason='No common completed horizon across all eight arms.', incomplete=True))
            return
        # Keep both horizons, with selection based on the greatest common horizon.
        for horizon in (160, 320):
            if horizon > common:
                break
            for arm in ARMS:
                self.evaluate(arm, self.seed, horizon, completed[(arm[0], horizon)])
        records = [r for r in self.records if r['horizon'] == common]
        control = next(r for r in records if r['arm'] == 'no_he')
        ranked = rank(records, control)
        confirmations = []
        if ranked:
            confirmation_arms = [ARMS[0]] + [next(a for a in ARMS if a[0] == r['arm']) for r in ranked[:2]]
            for arm in confirmation_arms:
                # Fits retain both archive horizons even when called directly for 320.
                path = self.fit(arm, self.seed + 1, common)
                if path is None:
                    break
                confirmations.append(self.evaluate(arm, self.seed + 1, common, path))
            self.permutations(ranked[0])
        confirmed = []
        if len(confirmations) > 1:
            ref = confirmations[0]
            for r in confirmations[1:]:
                first = next(x for x in ranked if x['arm'] == r['arm'])
                if preservation(r, ref)['eligible']:
                    confirmed.append((.5*(primary_score(first, control)+primary_score(r, ref)), first, r))
        selected = None
        if confirmed:
            best = min(x[0] for x in confirmed)
            selected = min((x for x in confirmed if x[0] <= best + .01 + 1e-12),
                key=lambda x: (.5*(x[1]['distributions']['cd8']['loss']+x[2]['distributions']['cd8']['loss']), x[1]['weight'], x[1]['dimension']))
        promote = bool(selected and selected[0] <= .99 and primary_score(selected[1], control) <= .99
                       and len(confirmations) == min(2, len(ranked)) + 1)
        winner = selected[1] if selected else (ranked[0] if ranked else None)
        if winner and (not ranked or winner['id'] != ranked[0]['id']):
            self.permutations(winner)
        decision = dict(promote=promote, common_updates=common,
            leading_candidate=winner['arm'] if winner else None,
            initial_eligible=[r['arm'] for r in ranked], confirmation_completed=[r['arm'] for r in confirmations],
            confirmed_mean_probe_error_ratio=selected[0] if selected else None,
            cumulative_fit_seconds=self.spent, elapsed_seconds=time.perf_counter()-self.started,
            reason='Preservation and >=1% improvement passed with confirmation.' if promote else
                   'Retain current default: preservation, improvement or confirmation requirements were not all met.',
            incomplete_comparisons=[f'{a[0]}@{h}' for a in ARMS for h in (160, 320) if (a[0], h) not in completed])
        if winner:
            chosen = self.config(next(a for a in ARMS if a[0] == winner['arm']), self.seed)
            (self.root/'selected_candidate.yaml').write_text(yaml.safe_dump(chosen, sort_keys=False))
        self.report(decision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--training-minutes', type=float, default=90.)
    parser.add_argument('--overall-hours', type=float, default=10.)
    args = parser.parse_args()
    pilot = Pilot(args.output, args.config, args.training_minutes*60, args.overall_hours*3600)
    try:
        pilot.run()
        from dann.verification.deep_cca_report import render_report
        render_report(pilot.root)
    finally:
        close(pilot.bundle)


if __name__ == '__main__':
    main()
