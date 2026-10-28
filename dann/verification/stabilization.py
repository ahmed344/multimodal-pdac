"""Matched three-epoch stabilization ablations with a fixed 300-epoch GRL horizon.

Create configurations with --prepare-only, then execute selected arms with --run
or all arms with --run-all. Existing arm histories are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr
import torch
import yaml

from dann.config import load_config, resolve_execution, resolve_device, seed_everything
from dann.data_loader import create_data_bundle
from dann.initialization import initialize_supervised_heads
from dann.losses import ZILNLoss
from dann.lr_scheduler import UpdateWarmup, create_lr_scheduler, step_lr_scheduler
from dann.model import AdversarialLatentFusion
from dann.monitoring import StreamingCovariance, freeze_monitoring
from dann.targets import CD8
from dann.train import TrainingProgress, run_epoch, move_batch_to_device, save_checkpoint
from dann.training_artifacts import atomic_json
from dann.verification.core_batch_pilot import Diagnostics
from dann.verification.fit_diagnosis import variation


ARMS = [*(('mlp', 512, ablation, .001) for ablation in
          ('baseline', 'residuals', 'centering', 'initialization', 'warmup', 'combined')),
        ('mlp', 1024, 'combined', .001), ('mlp', 1024, 'combined', .0005),
        ('cnn', 512, 'baseline', .001), ('cnn', 512, 'combined', .001),
        ('gatv2', 512, 'baseline', .001), ('gatv2', 512, 'combined', .001)]


def arm_config(source: dict, kind: str, width: int, ablation: str, lr: float, root: Path) -> dict:
    """Explicit experiment settings; never change production files or sample caps."""
    c = copy.deepcopy(source)
    name = f'{kind}_{width}_{ablation}_lr{lr:g}'
    residual = ablation in ('combined', 'residuals')
    spectral = c['model']['spectral_encoder']['deep_sets']
    spectral.update(peak_hidden_dims=[width, width], peak_output_dim=width,
                    peak_mlp_residual=residual,
                    pool_centering='zero_input' if ablation in ('combined', 'centering') else 'none')
    if kind == 'mlp' and width >= 1024:
        c['model'].update(pixel_spectral_microbatching=True, spectral_peak_budget=262144,
                          spectral_checkpointing=True)
    c['model']['latent_dim'] = width
    c['model']['aggregation']['mlp']['hidden_dims'] = [width, width]
    c['model']['aggregation']['gatv2']['depth'] = 4
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = kind
        group['mlp']['residual'] = residual
        group['gatv2']['residual'] = residual  # Historical GATv2 baseline: no residuals.
        # CNN flags retain their source values in both arms.
    for mode in ('pixel', 'spatial'):
        c['training'][mode]['learning_rate'] = lr
    c['training'].update(epochs=300, early_stopping_enabled=False, resume_checkpoint=None,
                         grl_horizon_updates=None,
                         head_initialization=dict(enabled=ablation in ('combined', 'initialization'),
                                                  output_weight_scale=.01, min_positive_sigma=.1),
                         warmup=dict(epochs=int(ablation in ('combined', 'warmup')), start_factor=.1))
    c['post_training']['enabled'] = False
    c['analysis'].update(latent_interactive=False, export_he_predictions=False)
    c['analysis']['peaks']['enabled'] = False
    c['performance']['async_artifacts'] = False
    c['results'].update(root=str(root/'runs'), run_name=name, smoke=False)
    return resolve_execution(c)


class DistributionDiagnostics(Diagnostics):
    """Validation distributions at frozen training-positive quartiles."""
    def __init__(self, cd8: int, thresholds: list[float], epsilon: float):
        super().__init__(cd8)
        self.thresholds, self.epsilon = thresholds, epsilon

    def finish(self) -> dict:
        result = super().finish()
        y, logits, mu, sigma = np.concatenate(self.cd8_values).astype(np.float64).T
        presence = expit(-logits)
        positive = y > 0
        p = np.clip(y[positive], self.epsilon, 1-self.epsilon)
        pit = ndtr((np.log(p)-np.log1p(-p)-mu[positive])/sigma[positive])
        result['cd8']['positive_coverage'] = {
            str(level): float(np.mean((pit >= (1-level)/2) & (pit <= (1+level)/2))) for level in (.5, .9)}
        result['cd8']['prediction_spread'] = {
            'presence_sd': float(presence.std()), 'mu_sd': float(mu.std()), 'sigma_sd': float(sigma.std())}
        result['cd8']['exceedance'] = {}
        for quartile, threshold in zip(('q25', 'q50', 'q75'), self.thresholds):
            t = np.clip(threshold, self.epsilon, 1-self.epsilon)
            predicted = presence * ndtr((mu-(np.log(t)-np.log1p(-t)))/sigma)
            result['cd8']['exceedance'][quartile] = dict(threshold=threshold,
                brier=float(np.mean((predicted-(y > threshold))**2)),
                predicted=float(predicted.mean()), observed=float(np.mean(y > threshold)))
        return result


@torch.no_grad()
def stage_probe(model, batch) -> dict:
    """Probe the same held-out rows without dropout or training RNG consumption."""
    training = model.training
    model.eval()
    try:
        out = model(batch, grl_strength=0.)
        pooled = model.encoder.microbatched(*(batch[k] for k in
            ('peak_indices', 'intensities', 'sample_indices', 'peak_counts')),
            peak_budget=model.peak_budget, checkpointing=False)
        if model.spatial:
            lookup = torch.full((batch['occupancy'].numel(),), -1, device=pooled.device, dtype=torch.long)
            lookup[batch['spatial_positions']] = torch.arange(len(pooled), device=pooled.device)
            pooled = pooled[lookup[batch['core_positions']]]
        cov = StreamingCovariance(); cov.update(out['latent'])
        return dict(pooled=variation(pooled), latent=variation(out['latent']), **cov.finish(),
                    predictions={k: variation(out[k]) for k in ('pi_logits', 'mu', 'sigma')})
    finally:
        model.train(training)


def component_parameters(model):
    """Named groups for actual AdamW update sizes, distinct from gradient norms."""
    return {name: list(module.parameters()) for name, module in
            [('embedding', model.encoder.embedding), ('peak_mlp', model.encoder.peak_mlp),
             ('aggregation', model.encoder.aggregation_mlp), ('biology', model.biology_predictor),
             ('he', model.he_predictor), ('discriminator', model.batch_discriminator)] if module is not None}


def run_arm(root: Path, name: str) -> None:
    """Fit one fresh arm for three full epochs, recording initial and early variation."""
    c = load_config(root/'configs'/f'{name}.yaml')
    directory = root/'arms'/name
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    t = c['training']
    seed_everything(t['seed'], t['deterministic_algorithms'])
    bundle = create_data_bundle(c, requested_splits=('train', 'validation'))
    try:
        split_file = root/'splits.npz'
        if split_file.exists():
            with np.load(split_file) as splits:
                for key, value in bundle.split_indices.items():
                    np.testing.assert_array_equal(value, splits[key])
        else:
            np.savez(split_file, **bundle.split_indices)
        logical = getattr(bundle.loaders['train'].sampler, 'logical_updates', len(bundle.loaders['train']))
        t['grl_horizon_updates'] = 300 * logical
        c = freeze_monitoring(c)
        (directory/'resolved_config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
        device = resolve_device(t['device'])
        model = AdversarialLatentFusion.from_config(c, len(bundle.metadata.batch_names), len(c['data']['target_columns'])).to(device)
        initial_hash = hashlib.sha256()
        for p in model.parameters():
            initial_hash.update(p.detach().cpu().numpy().tobytes())
        initialization = initialize_supervised_heads(model, bundle, c)
        atomic_json(directory/'head_initialization.json', initialization)
        optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'],
            weight_decay=t['weight_decay'], betas=(t['beta1'], t['beta2']))
        scheduler = create_lr_scheduler(optimizer, c)
        warmup = UpdateWarmup(optimizer, c, logical)
        progress = TrainingProgress()
        cd8 = c['data']['target_columns'].index(CD8)
        rows = bundle.split_indices['train']
        y = bundle.metadata.targets[rows, cd8]
        positive = y[bundle.metadata.target_valid_mask[rows, cd8] & (y > 0)]
        thresholds = np.quantile(positive, [.25, .5, .75]).tolist()
        atomic_json(directory/'manifest.json', dict(initial_parameters_sha256=initial_hash.hexdigest(),
            threshold_fit_split='train', thresholds=thresholds, logical_updates_per_epoch=logical,
            grl_horizon_updates=t['grl_horizon_updates'], epoch_limit=3,
            training_rows=len(rows), validation_rows=len(bundle.split_indices['validation'])))
        with torch.random.fork_rng():
            iterator = iter(bundle.loaders['validation'])
            audit_cpu = next(iterator)
            del iterator
            audit = move_batch_to_device(audit_cpu, device)
        torch.save(audit_cpu, directory/'probe_batch.pt')
        probes = [dict(update=0, **stage_probe(model, audit))]
        updates, history = [], []
        loss = ZILNLoss.from_config(c).to(device)
        kwargs = dict(model=model, ziln_loss=loss, device=device,
            biology_weight=c['loss']['biology_weight'], batch_weight=c['loss']['batch_weight'],
            he_weight=c['loss']['he_weight'], class_weights=bundle.batch_class_weights.to(device)
            if c['loss']['balance_batch_classes'] else None,
            total_epochs=300, grl_schedule=t['grl_schedule'], grl_max_lambda=t['grl_max_lambda'],
            grl_gamma=t['grl_gamma'], grl_horizon_updates=t['grl_horizon_updates'],
            cd8_index=cd8, target_columns=c['data']['target_columns'], monitoring=c['monitoring'])
        groups = component_parameters(model)
        before, update_record = {}, {}
        def pre_step(opt, args, kw):
            if progress.optimizer_updates < 10:
                before.update({k: [p.detach().clone() for p in ps] for k,ps in groups.items()})
                update_record.update({k: dict(gradient_l2=float(torch.stack([
                    p.grad.detach().square().sum() for p in ps if p.grad is not None]).sum().sqrt()))
                    for k,ps in groups.items()})
        def post_step(opt, args, kw):
            if before:
                for k,ps in groups.items():
                    delta = float(torch.stack([(p.detach()-q).square().sum() for p,q in zip(ps,before[k])]).sum().sqrt())
                    norm = float(torch.stack([q.square().sum() for q in before[k]]).sum().sqrt())
                    update_record[k].update(update_l2=delta, relative_update=delta/max(norm,1e-30))
                updates.append(dict(update=progress.optimizer_updates+1,
                                    learning_rate=opt.param_groups[0]['lr'], components=copy.deepcopy(update_record)))
                before.clear(); update_record.clear()
        optimizer.register_step_pre_hook(pre_step)
        optimizer.register_step_post_hook(post_step)
        def after_update(record):
            count = record['optimizer_updates']
            if count in (1, 5, 10):
                with torch.random.fork_rng():
                    probes.append(dict(update=count, **stage_probe(model, audit)))
                atomic_json(directory/'early_probes.json', probes)
                atomic_json(directory/'early_updates.json', updates)
            if count % 100 == 0:
                print(json.dumps(dict(arm=name, updates=count, elapsed_seconds=time.monotonic()-started)), flush=True)
        for epoch_index in range(4):
            row = dict(epoch=epoch_index)
            if epoch_index:
                row['train'] = run_epoch(**kwargs, loader=bundle.loaders['train'], optimizer=optimizer,
                    epoch_index=epoch_index-1, gradient_clip_norm=t['gradient_clip_norm'],
                    progress_state=progress, warmup=warmup, update_callback=after_update,
                    sampling_audit_dir=directory/'sampling')
            with torch.random.fork_rng():
                diagnostic = DistributionDiagnostics(cd8, thresholds, c['loss']['logit_epsilon'])
                row['validation'] = run_epoch(**kwargs, loader=bundle.loaders['validation'], optimizer=None,
                    epoch_index=max(0, epoch_index-1), gradient_clip_norm=None, evaluation_callback=diagnostic)
                row['diagnostics'] = diagnostic.finish()
                row['probe'] = stage_probe(model, audit)
            if epoch_index and not row['train']['warmup_updates']:
                step_lr_scheduler(scheduler, row['validation'])
            row.update(next_learning_rate=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic()-started)
            history.append(row)
            atomic_json(directory/'history.json', history)
            pd.json_normalize(history, sep='_').to_csv(directory/'history.csv', index=False)
            print(json.dumps(dict(arm=name, epoch=epoch_index, cd8_loss=row['validation']['cd8_loss'],
                variance=row['validation']['latent_total_variance'], elapsed_seconds=row['elapsed_seconds'])), flush=True)
        atomic_json(directory/'early_updates.json', updates)
        atomic_json(directory/'early_probes.json', probes)
        save_checkpoint(directory/'epoch_3.pt', model, optimizer, 2, c, bundle,
                        history[-1]['validation'][t['early_stopping_metric']],
                        progress_state=progress, lr_scheduler=scheduler, warmup=warmup)
        atomic_json(directory/'completion.json', dict(epochs=3, updates=progress.optimizer_updates,
                                                     elapsed_seconds=time.monotonic()-started))
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


def summarize(root: Path) -> None:
    """Write compact comparison tables and shareable plots for completed histories."""
    records = []
    for path in sorted((root/'arms').glob('*/history.json')):
        for row in json.loads(path.read_text()):
            d = row['diagnostics']; v = row['validation']
            records.append(dict(arm=path.parent.name, epoch=row['epoch'], cd8_loss=v['cd8_loss'],
                latent_variance=v['latent_total_variance'], participation_ratio=v['latent_participation_ratio'],
                presence_brier=d['cd8']['presence_brier'], coverage_50=d['cd8']['positive_coverage']['0.5'],
                coverage_90=d['cd8']['positive_coverage']['0.9'],
                **{f'exceedance_brier_{q}':x['brier'] for q,x in d['cd8']['exceedance'].items()},
                between_slide_fraction=d['latent']['between_slide_fraction'], batch_loss=v['batch_loss'],
                batch_accuracy=v['batch_accuracy'], pooled_centered_rms=row['probe']['pooled']['centered_rms'],
                **d['cd8']['prediction_spread']))
    if not records:
        return
    table = pd.DataFrame(records)
    table.to_csv(root/'comparison_all_epochs.csv', index=False)
    table[table.epoch == 3].to_csv(root/'comparison_epoch_3.csv', index=False)
    probes, updates = [], []
    for directory in sorted((root/'arms').iterdir()):
        probe_path, update_path = directory/'early_probes.json', directory/'early_updates.json'
        if probe_path.exists():
            probes.extend(dict(arm=directory.name, **row) for row in json.loads(probe_path.read_text()))
        if update_path.exists():
            for row in json.loads(update_path.read_text()):
                updates.extend(dict(arm=directory.name, update=row['update'],
                                    learning_rate=row['learning_rate'], component=name, **values)
                               for name, values in row['components'].items())
    if probes:
        pd.json_normalize(probes, sep='_').to_csv(root/'early_variation.csv', index=False)
    if updates:
        pd.DataFrame(updates).to_csv(root/'early_optimizer_updates.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selections = {
        'comparison': table,
        'mlp_ablations': table[table.arm.str.startswith('mlp_512_')],
        'width_learning_rates': table[table.arm.str.startswith('mlp_') & table.arm.str.contains('_combined_')],
        'spatial_ablations': table[~table.arm.str.startswith('mlp_')],
    }
    for name, selected in selections.items():
        if selected.empty:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
        for ax, metric in zip(axes.flat, ('cd8_loss','latent_variance','participation_ratio','presence_brier','coverage_90','between_slide_fraction')):
            for arm, values in selected.groupby('arm', sort=False):
                ax.plot(values.epoch, values[metric], marker='.', label=arm)
            ax.set(xlabel='Completed epochs', title=metric.replace('_',' '), xticks=[0,1,2,3])
            if metric == 'latent_variance':
                ax.set_yscale('log')
            if metric == 'coverage_90':
                ax.axhline(.9, color='gray', ls='--')
        axes[0,0].legend(fontsize=6 if name != 'comparison' else 5)
        fig.savefig(root/f'{name}.png', dpi=180)
        fig.savefig(root/f'{name}.pdf')
        plt.close(fig)
    final = table[table.epoch == 3].set_index('arm')
    deltas = []
    for arm, row in final.iterrows():
        kind, width, *_ = arm.split('_')
        baseline = f'{kind}_{width}_baseline_lr0.001'
        if arm != baseline and baseline in final.index:
            delta = row - final.loc[baseline]
            deltas.append(dict(arm=arm, reference=baseline, **delta.drop('epoch').to_dict()))
    if deltas:
        pd.DataFrame(deltas).to_csv(root/'deltas_from_baseline.csv', index=False)
    hashes = {}
    for path in (root/'arms').glob('*/manifest.json'):
        key = '_'.join(path.parent.name.split('_')[:2])
        hashes.setdefault(key, {})[path.parent.name] = json.loads(path.read_text())['initial_parameters_sha256']
    atomic_json(root/'matched_initial_draws.json', {
        key: dict(matched=len(set(values.values())) == 1, arms=values) for key, values in hashes.items()})



def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.prepare_only:
        root.mkdir(parents=True, exist_ok=False)
        (root/'configs').mkdir(); (root/'arms').mkdir()
        shutil.copyfile(args.config, root/'source_config.yaml')
        for path in Path('dann').rglob('*.py'):
            dest = root/'source'/path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        source = load_config(args.config)
        names = []
        for arm in ARMS:
            c = arm_config(source, *arm, root)
            name = c['results']['run_name']; names.append(name)
            (root/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
        atomic_json(root/'manifest.json', dict(arms=names, epoch_limit=3, grl_schedule_epochs=300,
            limitations='Three epochs assess early training, not convergence or biological utility.'))
    elif args.run:
        run_arm(root, args.run)
        summarize(root)
    elif args.run_all:
        for name in json.loads((root/'manifest.json').read_text())['arms']:
            if not (root/'arms'/name/'completion.json').exists():
                run_arm(root, name)
                summarize(root)
    elif args.summarize:
        summarize(root)
    else:
        parser.error('Choose --prepare-only, --run, --run-all, or --summarize.')


if __name__ == '__main__':
    main()
