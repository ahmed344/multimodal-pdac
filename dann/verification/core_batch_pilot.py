"""Budgeted paired all-GATv2 pilot using production accumulation and saved splits.

Run ``python -m dann.verification.core_batch_pilot --help`` for invocation.
The source checkpoint supplies row identity and splits only, never fitted weights.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import resource
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit, ndtr

from dann.config import load_config, resolve_execution, seed_everything
from dann.data_loader import create_data_bundle
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.targets import CD8
from dann.train import TrainingProgress, checkpoint_metadata, checkpoint_payload, run_epoch
from dann.training_artifacts import atomic_json, atomic_output, cpu_snapshot


def audit_sampling(root: Path) -> dict:
    """Replay CPU-only requests against hashes of consumed rows and epoch boundaries."""
    from dann.sampling import ProportionalSlideTileSampler
    from dann.tiles import SpatialTileDataset
    manifest = json.loads((root/'manifest.json').read_text())
    with np.load(root/'splits.npz') as saved:
        train_rows = saved['train']
    result, geometry = {}, None
    for name, directory in manifest['arm_directories'].items():
        directory = Path(directory)
        config = yaml.safe_load((directory/'resolved_config.yaml').read_text())
        # No spectra or H&E values are read: this audit only needs grid identities.
        tiles = SpatialTileDataset(config, train_rows, input_path=config['data']['path'], geometry=geometry)
        geometry = tiles.geometry
        sampler = ProportionalSlideTileSampler(tiles, config['training']['seed'],
                                               config['training']['supervised_rows_per_batch'])
        path = directory/'updates.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        epochs, expected_epoch, total_rows, step = [], 0, 0, 0
        requests = iter(sampler.logical_requests())
        visited = np.zeros(len(tiles.keys), dtype=bool)
        try:
            for update, record in enumerate(records, 1):
                request = next(requests)
                rows = np.asarray(request.row_ids, dtype='<i8')
                if (record['epoch'] != expected_epoch or record['step'] != step
                        or record['optimizer_updates'] != update
                        or record['row_ids_sha256'] != hashlib.sha256(rows.tobytes()).hexdigest()
                        or record['supervised_rows'] != len(rows)
                        or record['unique_tiles'] != len(request.tile_ids)):
                    raise ValueError(f'{name}: consumed request differs at update {update}.')
                if visited[rows].any() or len(np.unique(rows)) != len(rows):
                    raise ValueError(f'{name}: duplicate supervision in epoch {expected_epoch}.')
                visited[rows] = True
                total_rows += len(rows)
                step += 1
                if record['cumulative_supervised_rows'] != total_rows:
                    raise ValueError(f'{name}: row counter differs at update {update}.')
                if step == sampler.logical_updates:
                    if not visited[train_rows].all() or visited.sum() != len(train_rows):
                        raise ValueError(f'{name}: incomplete epoch coverage.')
                    epochs.append(dict(epoch=expected_epoch, rows=int(visited.sum()), complete=True))
                    expected_epoch += 1
                    sampler.set_epoch(expected_epoch)
                    requests = iter(sampler.logical_requests())
                    visited.fill(False)
                    step = 0
            if step:
                epochs.append(dict(epoch=expected_epoch, rows=int(visited.sum()), complete=False))
            result[name] = dict(passed=True, updates=len(records), supervised_rows=total_rows,
                                logical_updates_per_epoch=sampler.logical_updates, epochs=epochs)
        finally:
            tiles.close()
    atomic_json(root/'sampling_verification.json', result)
    return result


def pilot_config(base: dict, name: str, root: Path) -> dict:
    """Resolve the two combined-change arms without editing production defaults."""
    config = copy.deepcopy(base)
    candidate = name == 'candidate'
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = 'gatv2'
        group['gatv2'].update(neighbor_radius=1.5, attention_dropout=.1 if candidate else 0.)
    config['model']['heads']['he']['enabled'] = True
    config['training']['spatial'].update(
        core_size=32 if candidate else 8, supervised_rows_per_batch=32768 if candidate else 2048,
        microbatch_rows=4096 if candidate else None, tiles_per_batch=8, learning_rate=.0001)
    config['training'].update(grl_horizon_updates=500, grl_max_lambda=.25, grl_gamma=10.,
                              early_stopping_enabled=False, resume_checkpoint=None)
    config['performance'].update(validation_tiles=8, async_artifacts=False)
    config['post_training'] = {key: False for key in config['post_training']}
    config['results'].update(root=str(root), run_name=name)
    resolve_execution(config)
    config['analysis']['checkpoint'] = str(Path(config['training']['output_dir'])/'latest.pt')
    return config


def rng_state() -> dict:
    """Capture each arm's independent dropout stream at an optimizer boundary."""
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


class Diagnostics:
    """Held-out CD8 distribution calibration and centered latent variation."""
    def __init__(self, cd8: int):
        self.cd8 = cd8
        self.cd8_values = []
        self.latent_groups = {}
        self.rank_sample = []
        self.class_counts = None
        self.class_correct = None

    def __call__(self, batch, outputs):
        selected = batch['target_valid_mask'][:, self.cd8]
        self.cd8_values.append(torch.stack([
            batch['targets'][:, self.cd8][selected],
            outputs['pi_logits'][:, self.cd8][selected],
            outputs['mu'][:, self.cd8][selected],
            outputs['sigma'][:, self.cd8][selected]], 1).detach().cpu().numpy())
        z = outputs['latent'].detach().cpu().numpy().astype(np.float64)
        slides = batch['batches'].cpu().numpy()
        rows = batch['row_ids'].cpu().numpy()
        self.rank_sample.append(z[rows % 23 == 0])
        for slide in np.unique(slides):
            values = z[slides == slide]
            n, mean, m2 = self.latent_groups.get(int(slide), (0, np.zeros(z.shape[1]), np.zeros(z.shape[1])))
            k = len(values)
            center = values.mean(0)
            delta = center - mean
            m2 = m2 + ((values-center)**2).sum(0) + delta**2 * n * k / (n+k)
            self.latent_groups[int(slide)] = (n+k, mean + delta*k/(n+k), m2)
        predicted = outputs['batch_logits'].argmax(1).cpu().numpy()
        classes = outputs['batch_logits'].shape[1]
        if self.class_counts is None:
            self.class_counts = np.zeros(classes, dtype=np.int64)
            self.class_correct = self.class_counts.copy()
        self.class_counts += np.bincount(slides, minlength=classes)
        self.class_correct += np.bincount(slides[predicted == slides], minlength=classes)

    def finish(self) -> dict:
        y, logits, mu, sigma = np.concatenate(self.cd8_values).astype(np.float64).T
        presence = expit(-logits)
        positive = y > 0
        pit = ndtr((np.log(np.clip(y[positive], 1e-5, 1-1e-5) /
                             (1-np.clip(y[positive], 1e-5, 1-1e-5))) - mu[positive]) / sigma[positive])
        bins = np.minimum((presence * 10).astype(int), 9)
        calibration = [dict(count=int((bins == i).sum()),
                            predicted=float(presence[bins == i].mean()),
                            observed=float(positive[bins == i].mean()))
                       for i in range(10) if np.any(bins == i)]
        cd8 = dict(rows=len(y), positive_rows=int(positive.sum()),
                   presence_brier=float(np.mean((presence-positive)**2)),
                   predicted_presence=float(presence.mean()), observed_presence=float(positive.mean()),
                   presence_bins=calibration, positive_pit_mean=float(pit.mean()),
                   positive_pit_histogram=np.histogram(pit, bins=np.linspace(0, 1, 11))[0].tolist(),
                   positive_coverage={str(level): float(np.mean((pit >= (1-level)/2) & (pit <= (1+level)/2)))
                                      for level in (.5, .9)})
        cd8['exceedance'] = {}
        for threshold in (.01, .05, .1):
            p = presence * ndtr((mu - np.log(threshold/(1-threshold))) / sigma)
            cd8['exceedance'][str(threshold)] = dict(
                predicted=float(p.mean()), observed=float((y > threshold).mean()),
                brier=float(np.mean((p-(y > threshold))**2)))
        groups = list(self.latent_groups.values())
        n = sum(g[0] for g in groups)
        mean = sum(k*m for k, m, _ in groups) / n
        within = sum(m2 for _, _, m2 in groups)
        between = sum(k*(m-mean)**2 for k, m, _ in groups)
        total = within + between
        sample = np.concatenate(self.rank_sample)
        sample -= sample.mean(0)
        eigenvalues = np.maximum(np.linalg.eigvalsh(sample.T @ sample / max(len(sample)-1, 1)), 0.)
        proportions = eigenvalues / max(eigenvalues.sum(), 1e-300)
        nonzero = proportions > 0
        latent = dict(rows=n, variance_trace=float(total.sum()/max(n-1, 1)),
                      within_slide_variance_trace=float(within.sum()/max(n-1, 1)),
                      between_slide_fraction=float(between.sum()/max(total.sum(), 1e-300)),
                      mean_dimension_sd=float(np.sqrt(total/max(n-1, 1)).mean()),
                      rank_sample_rows=len(sample),
                      sampled_effective_rank=float(np.exp(-(proportions[nonzero]*np.log(proportions[nonzero])).sum())),
                      sampled_top_pc_fraction=float(proportions.max()))
        present = self.class_counts > 0
        return dict(cd8=cd8, latent=latent, discriminator_balanced_accuracy=float(
            np.mean(self.class_correct[present]/self.class_counts[present])))


class Arm:
    def __init__(self, name: str, config: dict, source: dict, root: Path):
        self.name, self.config = name, config
        self.directory = Path(config['training']['output_dir'])
        self.directory.mkdir(parents=True)
        self.bundle = create_data_bundle(config, source['split_indices'], source['data_identity'],
                                         requested_splits=('train', 'validation'))
        seed_everything(config['training']['seed'], config['training']['deterministic_algorithms'])
        self.model = AdversarialLatentFusion.from_config(config, len(self.bundle.metadata.batch_names),
                                                        len(config['data']['target_columns'])).cuda()
        self.initial_hash = hashlib.sha256(b''.join(
            p.detach().cpu().numpy().tobytes() for p in self.model.parameters())).hexdigest()
        t = config['training']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=t['learning_rate'],
            weight_decay=t['weight_decay'], betas=(t['beta1'], t['beta2']))
        self.loss = ZILNLoss.from_config(config).cuda()
        self.weights = self.bundle.batch_class_weights.cuda() if config['loss']['balance_batch_classes'] else None
        self.progress = TrainingProgress()
        self.rng = rng_state()
        self.metadata = checkpoint_metadata(self.model, config, self.bundle)
        self.updates, self.evaluations, self.fallbacks = [], [], []
        self.write_config()

    def write_config(self):
        (self.directory / 'resolved_config.yaml').write_text(yaml.safe_dump(self.config, sort_keys=False))

    def arguments(self) -> dict:
        t, loss = self.config['training'], self.config['loss']
        return dict(model=self.model, ziln_loss=self.loss, device=torch.device('cuda'),
            biology_weight=loss['biology_weight'], batch_weight=loss['batch_weight'],
            he_weight=loss['he_weight'], class_weights=self.weights, total_epochs=t['epochs'],
            epoch_index=self.progress.epoch, grl_schedule=t['grl_schedule'], grl_max_lambda=t['grl_max_lambda'],
            grl_gamma=t['grl_gamma'], grl_horizon_updates=t['grl_horizon_updates'],
            cd8_index=self.config['data']['target_columns'].index(CD8),
            target_columns=self.config['data']['target_columns'])

    def checkpoint(self, name: str):
        payload = checkpoint_payload(self.metadata, cpu_snapshot(dict(
            model_state=self.model.state_dict(), optimizer_state=self.optimizer.state_dict())),
            self.progress.epoch-1, float('inf'), progress_state=self.progress)
        payload.update(pilot=True, rng_state=self.rng)
        with atomic_output(self.directory / name) as path:
            torch.save(payload, path)

    def evaluate(self, started: float) -> dict:
        print(f'Validating {self.name} at update {self.progress.optimizer_updates}', flush=True)
        restore_rng(self.rng)
        diagnostic = Diagnostics(self.config['data']['target_columns'].index(CD8))
        metrics = run_epoch(**self.arguments(), loader=self.bundle.loaders['validation'],
                            optimizer=None, gradient_clip_norm=None, evaluation_callback=diagnostic)
        result = dict(update=self.progress.optimizer_updates, rows_processed=self.progress.supervised_rows,
                      elapsed_seconds=perf_counter()-started, **metrics, diagnostics=diagnostic.finish())
        self.evaluations.append(result)
        atomic_json(self.directory/'evaluations.json', self.evaluations)
        print(json.dumps(dict(arm=self.name, validation=result)), flush=True)
        return result

    def train_to(self, target: int, deadline: float, started: float):
        restore_rng(self.rng)

        def completed(record):
            record.update(elapsed_seconds=perf_counter()-started,
                          microbatch_rows=self.config['training']['microbatch_rows'],
                          rows_per_second=record['supervised_rows']/max(record['seconds'], 1e-9),
                          peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                          peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/2**30)
            self.updates.append(record)
            self.rng = rng_state()
            with (self.directory/'updates.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            if record['optimizer_updates'] % 5 == 0 or record['optimizer_updates'] == 1:
                print(json.dumps(dict(arm=self.name, update=record)), flush=True)

        while self.progress.optimizer_updates < target and perf_counter() < deadline:
            failure = None
            before = self.progress.optimizer_updates
            try:
                metrics = run_epoch(**self.arguments(), loader=self.bundle.loaders['train'],
                    optimizer=self.optimizer, gradient_clip_norm=self.config['training']['gradient_clip_norm'],
                    progress_state=self.progress, max_updates=target-self.progress.optimizer_updates,
                    deadline=deadline, update_callback=completed)
                print(json.dumps(dict(arm=self.name, training=metrics)), flush=True)
            except (torch.cuda.OutOfMemoryError, MemoryError) as error:
                failure = str(error)
            if failure is not None:
                micro = self.config['training']['microbatch_rows']
                if self.name != 'candidate' or micro <= 1024:
                    raise MemoryError(failure)
                micro //= 2
                self.optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                restore_rng(self.rng)
                self.config['training']['spatial']['microbatch_rows'] = micro
                resolve_execution(self.config)
                self.config['analysis']['checkpoint'] = str(self.directory/'latest.pt')
                self.bundle.loaders['train'].sampler.microbatch_rows = micro
                self.metadata = checkpoint_metadata(self.model, self.config, self.bundle)
                self.fallbacks.append(dict(update=self.progress.optimizer_updates, microbatch_rows=micro, reason=failure))
                atomic_json(self.directory/'fallbacks.json', self.fallbacks)
                self.write_config()
                continue
            if self.progress.optimizer_updates == before:
                break
        self.checkpoint('latest.pt')


def run(args) -> dict:
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    deadline = started + args.minutes * 60
    source = torch.load(args.split_checkpoint, map_location='cpu', weights_only=False)
    # Frozen splits are checked against the current row/feature identity by the data loader.
    if len(source['split_indices']['test']) or not .89 < len(source['split_indices']['train']) / sum(
            len(v) for v in source['split_indices'].values()) < .91:
        raise ValueError('Pilot requires a saved 90/10 split with no test rows.')
    source = {k: source[k] for k in ('split_indices', 'data_identity')}
    np.savez_compressed(root/'splits.npz', **source['split_indices'])
    base = load_config(args.config)
    arms = []
    try:
        for name in ('reference', 'candidate'):
            print(f'Preparing {name}', flush=True)
            arms.append(Arm(name, pilot_config(base, name, root), source, root))
        assert arms[0].initial_hash == arms[1].initial_hash
        manifest = dict(split_checkpoint=str(args.split_checkpoint), split_sizes={
            k: len(v) for k, v in source['split_indices'].items()},
            data_identity=source['data_identity'], initial_parameter_sha256=arms[0].initial_hash,
            budget_seconds=args.minutes*60, max_updates=args.updates, block_updates=50,
            grl_horizon_updates=500, initialization='fresh identical parameters',
            arm_directories={a.name: str(a.directory) for a in arms},
            paired_update=0, preparation_seconds=perf_counter()-started)
        atomic_json(root/'manifest.json', manifest)
        for arm in arms:
            arm.checkpoint('update_0000.pt')
            arm.evaluate(started)
        manifest['initial_validation_absolute_differences'] = {
            key: abs(arms[0].evaluations[0][key] - arms[1].evaluations[0][key])
            for key in ('supervised_objective', 'cd8_loss', 'he_mse')}
        atomic_json(root/'manifest.json', manifest)
        paired = 0
        reason = 'maximum_updates'
        for target in range(50, args.updates+1, 50):
            validation_seconds = sum(a.evaluations[-1]['seconds'] for a in arms)
            reserve = validation_seconds * 1.25 + 30.
            # After one paired block, avoid starting a pair unlikely to finish.
            predicted = sum(np.mean([r['seconds'] for r in a.updates[-50:]])*50
                            for a in arms if a.updates)
            if perf_counter() + reserve + predicted >= deadline:
                reason = 'insufficient_time_for_another_paired_block'
                break
            for arm in arms:
                arm.train_to(target, deadline-reserve, started)
                if arm.progress.optimizer_updates < target:
                    reason = 'budget_reached_inside_block'
                    break
            if any(a.progress.optimizer_updates != target for a in arms):
                break
            for arm in arms:
                arm.checkpoint(f'update_{target:04d}.pt')
                arm.evaluate(started)
            paired = target
            manifest.update(paired_update=paired, elapsed_seconds=perf_counter()-started)
            atomic_json(root/'manifest.json', manifest)
        result = dict(manifest, paired_update=paired, stopping_reason=reason,
                      elapsed_seconds=perf_counter()-started,
                      arms={a.name: dict(progress=asdict(a.progress), evaluations=a.evaluations,
                                         fallbacks=a.fallbacks) for a in arms})
        atomic_json(root/'results.json', result)
        rows = [dict(arm=a.name, **{k: v for k, v in e.items() if k != 'diagnostics'},
                     **{f'latent_{k}': v for k, v in e['diagnostics']['latent'].items()},
                     cd8_presence_brier=e['diagnostics']['cd8']['presence_brier'])
                for a in arms for e in a.evaluations]
        pd.DataFrame(rows).to_csv(root/'comparison.csv', index=False)
        audit_sampling(root)
        return result
    finally:
        for arm in arms:
            for dataset in arm.bundle.datasets.values():
                dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--split-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minutes', type=float, default=60.)
    parser.add_argument('--updates', type=int, default=500)
    args = parser.parse_args()
    if not 0 < args.minutes <= 60 or not 50 <= args.updates <= 500 or args.updates % 50:
        parser.error('Use 0 < minutes <= 60 and a multiple of 50 updates, at most 500.')
    result = run(args)
    print(json.dumps(dict(paired_update=result['paired_update'],
                          elapsed_seconds=result['elapsed_seconds'],
                          stopping_reason=result['stopping_reason'])), flush=True)


if __name__ == '__main__':
    main()
