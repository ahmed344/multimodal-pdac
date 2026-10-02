"""Four-hour, three-complete-epoch biology diagnosis, using archived production B.

python -m dann.verification.cnn_biology_diagnosis --output NEW_DIRECTORY
Only training and full validation are executed. Output directories are never reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

from dann.config import load_config, seed_everything
from dann.data_loader import create_data_bundle
from dann.losses import ZILNLoss
from dann.tiles import tile_collate
from dann.train import move_batch_to_device, run_epoch
from dann.verification.batch_execution_diagnosis import Deadline
from dann.verification.biology_variants import (make_variant, optimizer_for, audit,
                                               preserve_state)
from dann.verification.epoch_comparison import file_hash
from dann.verification.extended_batch_diagnosis import admit, direct_source_check
from dann.verification.production_fits import (save_json, digest, close_bundle, assess,
                                             fit_baseline, sampler_evidence, gpu_snapshot)

ARCHIVE = Path('data/PDAC/Results/dann/production/2026-10-01-two-architectures')
SEEDS = (20260719, 20260720, 20260721)
EPOCHS = 3
HORIZON = 300
REPORT_RESERVE = 20 * 60
FORMAT = 'cnn-biology-diagnostic-v1'


def classify(history: list[dict], independent: dict | None, baseline: dict) -> dict:
    """Only a complete epoch-three checkpoint can pass, regardless of earlier scores."""
    if [row['epoch'] for row in history] != [1, 2, 3] or independent is None:
        return dict(passed=False, state='incomplete')
    result = assess(independent, baseline)
    return dict(state='completed', **result)


def followups(pointwise_passed: bool) -> tuple[str, ...]:
    """Return the fixed, independent intervention table selected by P."""
    return ('shallow', 'residual', 'pointwise_init') if pointwise_passed else ('no_norm', 'no_dropout', 'low_lr')


def select_candidate(results: dict, names: tuple[str, ...]) -> str | None:
    """Stable table order is the final tie-breaker."""
    passing = [name for name in names if results[name]['assessment']['passed']]
    return min(passing, key=lambda name: (results[name]['validation']['cd8_loss'],
               results[name]['validation']['biology_loss'], names.index(name))) if passing else None


class CheckedLoader:
    """Check deadlines per batch and verify the exact consumed row/boundary stream."""
    def __init__(self, loader, deadline):
        self.loader, self.deadline = loader, deadline
        self.sampler, self.dataset = loader.sampler, loader.dataset
        self.rows, self.boundaries = [], [0]

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.deadline.check(REPORT_RESERVE)
            self.rows.extend(batch['row_ids'].tolist())
            self.boundaries.append(len(self.rows))
            yield batch

    def verify(self, expected_rows, plan=None):
        np.testing.assert_array_equal(np.sort(self.rows), np.sort(expected_rows))
        result = dict(rows=len(self.rows), batches=len(self.boundaries)-1,
                      row_order_sha256=digest(self.rows), boundaries_sha256=digest(self.boundaries))
        if plan is not None:
            for key in ('row_order_sha256', 'boundaries_sha256'):
                assert result[key] == plan[key]
            assert result['batches'] == plan['batches']
        return result


def epoch_arguments(config: dict, bundle: Any, model: torch.nn.Module,
                    loss: ZILNLoss, adversarial: bool) -> dict:
    """The three-epoch fit deliberately retains the production 300-epoch schedule."""
    t, weights = config['training'], config['loss']
    device = next(model.parameters()).device
    return dict(model=model, ziln_loss=loss, device=device,
        biology_weight=weights['biology_weight'], batch_weight=weights['batch_weight'] if adversarial else 0.,
        class_weights=bundle.batch_class_weights.to(device) if weights['balance_batch_classes'] else None,
        total_epochs=HORIZON, grl_schedule=t['grl_schedule'],
        grl_max_lambda=t['grl_max_lambda'] if adversarial else 0., grl_gamma=t['grl_gamma'],
        gradient_clip_norm=t['gradient_clip_norm'], target_columns=config['data']['target_columns'])


def run_fit(root: Path, config: dict, bundle: Any, plans: dict, audit_cpu: dict,
            baseline: dict, deadline: Deadline, seed: int, variant: str,
            adversarial: bool = False) -> dict:
    """One immutable fit, three checkpoints, independent rescore, and failure artifacts."""
    directory = root / f'{seed}_{variant}' / ('adversarial' if adversarial else 'disabled')
    directory.mkdir(parents=True, exist_ok=False)
    started = time.time()
    model = optimizer = None
    history, audits = [], []
    result = dict(seed=seed, variant=variant, adversarial=adversarial, directory=str(directory),
                  status='running', assessment=dict(passed=False, state='incomplete'))
    save_json(directory/'status.json', result)
    try:
        deadline.check(REPORT_RESERVE)
        seed_everything(seed, config['training']['deterministic_algorithms'])
        model, initialization = make_variant(config, len(bundle.metadata.batch_names), seed, variant, adversarial)
        torch.save(model.state_dict(), directory/'initial.pt')
        settings = dict(format=FORMAT, config=config, initialization=initialization,
                        fit_epochs=EPOCHS, grl_horizon=HORIZON, early_stopping=False,
                        sampling=plans[seed], audit_split='train', precision='float32')
        save_json(directory/'settings.json', settings)
        device = torch.device(config['training']['device'])
        # Explicitly retain the study's common derived CUDA dropout stream.
        # Model construction itself only touches/restores the CPU generator.
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(initialization['cuda_dropout_seed'])
        model.to(device)
        optimizer = optimizer_for(model, config, variant)
        loss = ZILNLoss.from_config(config).to(device)
        audit_batch = move_batch_to_device(audit_cpu, device)
        bundle.loaders['train'].sampler.seed = seed
        bundle.loaders['train'].generator.manual_seed(seed)
        # Fixed independent worker RNG means validation cannot shift training dropout.
        bundle.loaders['validation'].generator = torch.Generator().manual_seed(seed+1)
        common = epoch_arguments(config, bundle, model, loss, adversarial)
        torch.cuda.reset_peak_memory_stats()
        audits.append(dict(epoch=0, **audit(model, audit_batch, loss, config['loss']['biology_weight'])))
        save_json(directory/'audits.json', audits)
        for epoch in range(1, EPOCHS+1):
            deadline.check(REPORT_RESERVE)
            before = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
            loader = CheckedLoader(bundle.loaders['train'], deadline)
            tick = time.time()
            training = run_epoch(loader=loader, optimizer=optimizer, epoch_index=epoch-1,
                                 sampling_audit_dir=directory/'sampling', **common)
            coverage = loader.verify(bundle.split_indices['train'], plans[seed][epoch-1])
            train_seconds = time.time()-tick
            updates = {k: float((v.detach().cpu()-before[k]).norm()) for k, v in model.named_parameters()}
            del before
            # Save immediately after a complete epoch, even if its validation later fails.
            checkpoint = dict(format=FORMAT, epoch=epoch, updates=epoch*len(loader), settings=settings,
                model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                split_indices=bundle.split_indices, batch_names=bundle.metadata.batch_names,
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all())
            torch.save(checkpoint, directory/f'epoch_{epoch}.pt')
            with preserve_state(model):
                tick = time.time()
                validation_loader = CheckedLoader(bundle.loaders['validation'], deadline)
                validation = run_epoch(loader=validation_loader, optimizer=None, epoch_index=epoch-1, **common)
                validation_coverage = validation_loader.verify(bundle.split_indices['validation'])
                validation_seconds = time.time()-tick
                audit_result = audit(model, audit_batch, loss, config['loss']['biology_weight'])
            row = dict(epoch=epoch, updates=epoch*len(loader), train=training, validation=validation,
                       train_seconds=train_seconds, validation_seconds=validation_seconds,
                       sampling=coverage, validation_coverage=validation_coverage,
                       parameter_epoch_update_l2=updates)
            history.append(row)
            audits.append(dict(epoch=epoch, **audit_result))
            with (directory/'fit.log').open('a') as log:
                log.write(json.dumps(dict(epoch=epoch, train=training, validation=validation,
                                          train_seconds=train_seconds, validation_seconds=validation_seconds))+'\n')
            if device.type == 'cuda':
                with (directory/'gpu.jsonl').open('a') as log:
                    log.write(json.dumps(dict(epoch=epoch, **gpu_snapshot()))+'\n')
            save_json(directory/'history.json', history)
            save_json(directory/'audits.json', audits)
            print(json.dumps(dict(run=directory.name, variant=variant, seed=seed, epoch=epoch,
                                  train_seconds=train_seconds, validation=validation)), flush=True)
        # Reload epoch 3 into a fresh instance; do not trust live weights or select best epoch.
        saved = torch.load(directory/'epoch_3.pt', map_location='cpu', weights_only=False)
        assert saved['format'] == FORMAT and saved['epoch'] == EPOCHS
        restored, metadata = make_variant(config, len(bundle.metadata.batch_names), seed, variant, adversarial)
        assert metadata == initialization
        restored.load_state_dict(saved['model_state'], strict=True)
        restored.to(device)
        common['model'] = restored
        loader = CheckedLoader(bundle.loaders['validation'], deadline)
        with preserve_state(restored):
            independent = run_epoch(loader=loader, optimizer=None, epoch_index=2, **common)
        coverage = loader.verify(bundle.split_indices['validation'])
        for key in ('cd8_loss', 'biology_loss', 'mean_r2_cd8'):
            np.testing.assert_allclose(independent[key], history[-1]['validation'][key], atol=1e-6, rtol=1e-6)
        result.update(status='completed', completed_epochs=3, updates=3*len(bundle.loaders['train']),
            validation=independent, validation_coverage=coverage,
            assessment=classify(history, independent, baseline['validation']),
            checkpoint_sha256=file_hash(directory/'epoch_3.pt'))
        del restored, saved, checkpoint
    except Exception as error:
        result.update(status='deadline_interrupted' if isinstance(error, TimeoutError) else
                      'oom' if isinstance(error, torch.OutOfMemoryError) else
                      'numerical_failure' if isinstance(error, FloatingPointError) else 'failed',
                      error=traceback.format_exc(), completed_epochs=len(history))
        if model is not None:
            torch.save(dict(format=FORMAT, completed_epochs=len(history), model_state=model.state_dict(),
                            optimizer_state=optimizer.state_dict() if optimizer else None,
                            error=result['error']), directory/'failure.pt')
        (directory/'failure.log').write_text(result['error'])
        print(result['error'], flush=True)
    finally:
        result.update(seconds=time.time()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        save_json(directory/'status.json', result)
        # Do not carry model/optimizer or gradients between fits.
        del model, optimizer
        torch.cuda.empty_cache()
    return result


def route(group: Callable, get: Callable) -> str:
    """Execute predeclared branches; failed infrastructure never counts as a negative fit."""
    seed = SEEDS[0]
    if not group('screen', [(seed, v, False) for v in ('M', 'C', 'P')], reserve_fits=2):
        return 'Screen incomplete; no architecture conclusion.'
    if not get(seed, 'M')['assessment']['passed']:
        group('positive_control_confirmation', [(s, 'M', False) for s in SEEDS[1:]])
        return 'Positive control was not reliably established under the three-epoch protocol; architecture selection stopped.'
    if get(seed, 'C')['assessment']['passed']:
        if not group('current_cnn_restoration_screen', [(seed, 'C', True)], reserve_fits=4):
            return 'Current CNN passed; adversarial comparison incomplete.'
        group('current_cnn_confirmation', [(s, 'C', a) for s in SEEDS[1:] for a in (False, True)])
        return 'Current CNN learned without adversarial loss; inspect matched restoration and seed outcomes.'
    pointwise = get(seed, 'P')['assessment']['passed']
    names = followups(pointwise)
    selected = None
    if group('variants', [(seed, v, False) for v in names], reserve_fits=7):
        selected = select_candidate({v: get(seed, v) for v in names}, names)
    control = selected or ('P' if pointwise else 'M')
    if not group('decisive_confirmation', [(s, v, False) for s in SEEDS[1:] for v in ('C', control)],
                 reserve_fits=3 if selected else 0):
        return 'Screen completed; decisive seed confirmation incomplete.'
    if selected and all(get(s, selected)['assessment']['passed'] for s in SEEDS):
        if group('restoration', [(s, selected, True) for s in SEEDS]):
            return f'Spatial candidate {selected} passed all three seeds without adversarial loss; restoration completed.'
        return f'Spatial candidate {selected} passed all three seeds; restoration incomplete.'
    if selected:
        return f'Spatial candidate {selected} had mixed seed outcomes; adversarial restoration skipped.'
    return ('No tested spatial rescue established. Pointwise success is not a spatial-CNN rescue.' if pointwise else
            'No tested spatial rescue established; inspect matched C versus M confirmation.')


class Study:
    """Frozen evidence, sequential GPU ownership and measured group admission."""
    def __init__(self, root: Path, archive: Path) -> None:
        root.mkdir(parents=True, exist_ok=False)
        self.root, self.archive = root, archive
        self.deadline = Deadline(root/'deadline.json', 240)
        self.results, self.events = {}, []
        self.bundle = None
        self.source_hash = file_hash(Path('dann/config.yaml'))
        self.archive_hashes = {str(p): file_hash(p) for p in archive.iterdir() if p.is_file()}
        shutil.copyfile(archive/'B.yaml', root/'source_B.yaml')
        for source in Path('dann').rglob('*.py'):
            destination = root/'source'/source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        raw = yaml.safe_load((archive/'B.yaml').read_text())
        raw['model']['heads']['biology']['cnn']['residual'] = False
        raw['results'].update(root=str(root.resolve()/'unused_production_paths'), run_name='diagnostic_only')
        config_path = root/'geometry.yaml'
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        self.config = load_config(config_path)
        c, t = self.config, self.config['training']
        assert c['model']['aggregation']['type'] == c['model']['heads']['discriminator']['type'] == 'mlp'
        assert c['model']['heads']['biology']['type'] == 'cnn'
        assert c['model']['heads']['biology']['cnn']['depth'] == 3
        assert (t['core_size'], t['supervised_rows_per_batch'], t['learning_rate'], t['epochs']) == (8,2048,.001,300)
        assert c['execution']['halo'] == 3 and t['sampling_strategy'] == 'proportional_slide_tiles'
        assert all(c['data'].get('max_'+s+'_samples') is None for s in ('train','validation','test'))
        assert t['resume_checkpoint'] is None
        save_json(root/'resolved_geometry.json', c)

    def prepare(self) -> None:
        # First real-data diagnostic operation starts the persistent wall-clock budget.
        self.deadline.start()
        assert torch.cuda.is_available(), 'CUDA required; no CPU substitution.'
        while gpu_snapshot()['compute_pids']:
            self.deadline.check(REPORT_RESERVE)
            save_json(self.root/'gpu_wait.json', gpu_snapshot())
            time.sleep(30)
        save_json(self.root/'gpu_preflight.json', gpu_snapshot())
        self.input_stats = {key: dict(path=self.config['data'][key],
            size=Path(self.config['data'][key]).stat().st_size,
            mtime_ns=Path(self.config['data'][key]).stat().st_mtime_ns) for key in ('path', 'context_path')}
        identity = json.loads((self.archive/'data_identity.json').read_text())
        with np.load(self.archive/'splits.npz') as archive_splits:
            splits = {k: archive_splits[k] for k in archive_splits.files}
        self.bundle = create_data_bundle(self.config, splits, identity)
        b = self.bundle
        assert len(b.split_indices['train']) == 1358726 and len(b.split_indices['validation']) == 169824
        assert len(b.loaders['train']) == 664
        for k in splits:
            np.testing.assert_array_equal(splits[k], b.split_indices[k])
        save_json(self.root/'data_identity.json', identity)
        np.savez_compressed(self.root/'splits.npz', **splits)
        self.baseline = fit_baseline(b, self.config)
        old = json.loads((self.archive/'constant_baseline.json').read_text())
        for k in old['parameters']:
            np.testing.assert_allclose(self.baseline['parameters'][k], old['parameters'][k], atol=2e-6, rtol=1e-6)
        for k in old['validation']:
            np.testing.assert_allclose(self.baseline['validation'][k], old['validation'][k], atol=2e-6, rtol=1e-6)
        save_json(self.root/'constant_baseline.json', self.baseline)
        self.plans = {}
        for seed in SEEDS:
            b.loaders['train'].sampler.seed = seed
            self.plans[seed] = []
            for epoch in range(3):
                self.deadline.check(REPORT_RESERVE)
                b.loaders['train'].sampler.set_epoch(epoch)
                self.plans[seed].append(sampler_evidence(b))
        manifest = json.loads((self.archive/'manifest.json').read_text())
        assert self.plans[SEEDS[0]][0] == next(r['sampling'] for r in manifest['runs'] if r['name']=='B')
        save_json(self.root/'sampling_plans.json', self.plans)
        direct_source_check(self.root, self.config, b, b.datasets['train'])
        self.grid_check(b.datasets['train'])
        tiles = b.datasets['train']
        self.audit_cpu = tile_collate([tiles[int(i)] for i in np.linspace(0, len(tiles)-1, 4).astype(int)])
        assert set(self.audit_cpu['row_ids'].tolist()) <= set(splits['train'].tolist())
        torch.save(self.audit_cpu, self.root/'training_audit.pt')
        tiles.close()
        # Initializations and shared hashes are checked before any admitted fit.
        initializations = {}
        from dann.verification.biology_variants import VARIANTS
        for seed in SEEDS:
            for variant in VARIANTS:
                model, meta = make_variant(self.config, len(b.metadata.batch_names), seed, variant)
                initializations[f'{seed}_{variant}'] = meta
                assert meta['shared_sha256'] == initializations[f'{seed}_M']['shared_sha256']
                del model
        save_json(self.root/'initializations.json', initializations)
        save_json(self.root/'rng_streams.json', dict(streams={str(seed): dict(
            cpu_dropout=seed, cuda_dropout=seed+1000, shape_initialization=seed+1000)
            for seed in SEEDS}, executed_source_snapshot='source/dann/verification'))
        save_json(self.root/'preflight.json', dict(passed=True, input_stats=self.input_stats,
            source_config_sha256=self.source_hash, archive_hashes=self.archive_hashes,
            seconds=240*60-self.deadline.remaining(), free_disk_bytes=shutil.disk_usage(self.root).free,
            torch_version=torch.__version__, device=torch.cuda.get_device_name()))

    def grid_check(self, tiles: Any) -> None:
        """Independently reconstruct every occupied coordinate in sampled footprints."""
        checked = []
        slides = np.asarray(tiles.keys.get_level_values(0))
        xs = np.asarray(tiles.keys.get_level_values(1))
        ys = np.asarray(tiles.keys.get_level_values(2))
        for index in np.linspace(0, len(tiles)-1, 12).astype(int):
            tile = tiles[int(index)]
            (slide, tx, ty), selected = tiles.tiles[index]
            size, halo = tiles.size, tiles.halo
            expected = np.zeros((size, size), dtype=np.float32)
            rows = np.flatnonzero((slides == slide) & (xs >= tx*8-halo) & (xs < tx*8-halo+size)
                                  & (ys >= ty*8-halo) & (ys < ty*8-halo+size))
            expected[ys[rows]-(ty*8-halo), xs[rows]-(tx*8-halo)] = 1
            np.testing.assert_array_equal(expected, tile['occupancy'])
            np.testing.assert_array_equal(np.flatnonzero(expected), tile['spatial_positions'])
            for row, position in zip(tile['row_ids'], tile['core_positions']):
                s, x, y = tiles.keys[row]
                assert s == slide and position == (y-(ty*8-halo))*size+x-(tx*8-halo)
            checked.append(dict(tile=int(index), occupied=len(rows), supervised=len(selected)))
        save_json(self.root/'grid_checks.json', dict(passed=True, tiles=checked))
        tiles.close()

    def get(self, seed: int, variant: str, adversarial: bool = False) -> dict:
        return self.results[f'{seed}_{variant}_{adversarial}']

    def group(self, name: str, specs: list[tuple], reserve_fits: int = 0) -> bool:
        completed = [r['seconds'] for r in self.results.values() if r['status']=='completed']
        estimate = max(completed, default=780.)
        reserve = REPORT_RESERVE + 1.25*estimate*reserve_fits
        accepted = admit(self.deadline, [estimate]*len(specs), reserve)
        self.events.append(dict(group=name, specs=specs, accepted=accepted, estimate_seconds=estimate,
            reserve_seconds=reserve, available_seconds=self.deadline.remaining(), margin=1.25))
        save_json(self.root/'admissions.json', self.events)
        if not accepted:
            return False
        for seed, variant, adversarial in specs:
            self.deadline.check(REPORT_RESERVE)
            # Our own CUDA context is allowed; never terminate another process.
            other = set(gpu_snapshot()['compute_pids'].splitlines()) - {str(os.getpid()), ''}
            while other:
                self.deadline.check(REPORT_RESERVE)
                time.sleep(30)
                other = set(gpu_snapshot()['compute_pids'].splitlines()) - {str(os.getpid()), ''}
            result = run_fit(self.root, self.config, self.bundle, self.plans, self.audit_cpu,
                             self.baseline, self.deadline, seed, variant, adversarial)
            self.results[f'{seed}_{variant}_{adversarial}'] = result
            save_json(self.root/'results.json', self.results)
            if result['status'] != 'completed':
                # Preserve the failure; do not reinterpret it as an architecture failure.
                raise RuntimeError(f'Fit {seed}/{variant}/{adversarial} ended as {result["status"]}.')
        return True


def report(study: Study, conclusion: str) -> None:
    """Write comparison/seed tables and training, variation, gradient/sensitivity figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for key, result in study.results.items():
        validation = result.get('validation', {})
        assessment = result['assessment']
        rows.append(dict(run=key, seed=result['seed'], variant=result['variant'],
            adversarial=result['adversarial'], status=result['status'],
            cd8_loss=validation.get('cd8_loss'), biology_loss=validation.get('biology_loss'),
            positive_cd8_logit_r2=validation.get('mean_r2_cd8'),
            cd8_improvement=assessment.get('cd8_improvement'),
            biology_improvement=assessment.get('biology_improvement'), passed=assessment['passed'],
            seconds=result['seconds'], completed_epochs=result.get('completed_epochs', 0)))
    frame = pd.DataFrame(rows)
    frame.to_csv(study.root/'comparison.csv', index=False)
    seed_lines = []
    if rows:
        for (variant, adversarial), group in frame.groupby(['variant','adversarial']):
            outcomes = ', '.join(
                f'{r.seed}: ' + ('pass' if r.passed else 'fail' if r.status == 'completed'
                                 else 'incomplete (' + r.status + ')') for r in group.itertuples())
            seed_lines.append(f'- {variant}, adversarial={adversarial}: {outcomes}.')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for result in study.results.values():
            directory = Path(result['directory'])
            if not (directory/'history.json').exists():
                continue
            history = json.loads((directory/'history.json').read_text())
            label = f'{result["seed"]}/{result["variant"]}/adv={result["adversarial"]}'
            for ax, key in zip(axes, ('cd8_loss','biology_loss','mean_r2_cd8')):
                ax.plot([h['epoch'] for h in history], [h['validation'][key] for h in history], label=label)
                ax.set(xlabel='Epoch', ylabel=key)
            audits = json.loads((directory/'audits.json').read_text())
            figure, axs = plt.subplots(2, 2, figsize=(14, 9))
            for name in audits[0]['layers']:
                if not (name.endswith('/output') or name == 'input/0/input'):
                    continue
                axs[0,0].plot([a['epoch'] for a in audits], [a['layers'][name]['centered_rms'] for a in audits], label=name)
            for prefix in ('encoder.embedding', 'encoder.peak_mlp', 'encoder.aggregation_mlp', 'biology_predictor'):
                axs[0,1].plot([a['epoch'] for a in audits],
                    [sum(v*v for k,v in a['gradients'].items() if k.startswith(prefix))**.5 for a in audits], label=prefix)
                axs[1,1].plot([h['epoch'] for h in history],
                    [sum(v*v for k,v in h['parameter_epoch_update_l2'].items() if k.startswith(prefix))**.5 for h in history], label=prefix)
            for key in ('pi','mu','sigma'):
                for location in ('center','neighbor'):
                    axs[1,0].plot([a['epoch'] for a in audits],
                        [a['sensitivity'][key][location+'_l2'] for a in audits], label=f'{key}/{location}')
            axs[0,1].set_yscale('symlog', linthresh=1e-8)
            axs[1,0].set_yscale('symlog', linthresh=1e-8)
            for ax, title in zip(axs.flat, ('Feature centered RMS (outputs)', 'Audit gradient L2', 'CD8 latent sensitivity', 'Epoch parameter displacement L2')):
                ax.set(title=title, xlabel='Epoch')
                ax.legend(fontsize=6, ncol=2 if ax is axs[0,0] else 1)
            figure.tight_layout(); figure.savefig(directory/'audits.png', dpi=140); plt.close(figure)
        for ax, key in zip(axes, ('cd8_loss','biology_loss','mean_r2_cd8')):
            ax.axhline(study.baseline['validation'][key], color='black', linestyle='--')
        axes[0].legend(fontsize=5)
        fig.tight_layout(); fig.savefig(study.root/'learning_curves.png', dpi=160); plt.close(fig)
    text = ('# CNN biology-head diagnosis\n\n' + conclusion + '\n\n' +
        '\n'.join(seed_lines) + '\n\nAcceptance uses independently rescored epoch 3 only. '
        'Three-epoch failure is a failure within this diagnostic horizon, not proof an architecture cannot learn. '
        'Pointwise success alone cannot establish neighborhood correctness or a spatial rescue. '
        'Single-factor interventions support only the matched comparison; layer collapse and sensitivity audits are descriptive evidence.\n\n'
        'See comparison.csv, results.json, admissions.json, and each fit’s settings, three checkpoints, history, audits, and resource accounting. '
        'No test-split analysis, tissue inference, heatmaps, or production retraining was run.\n')
    if rows:
        text += '\n| Seed | Biology head | Adversarial | CD8 loss | CD8 improvement | Biology improvement | CD8 logit R² | Decision |\n'
        text += '|---|---|---|---:|---:|---:|---:|---|\n'
        def number(value):
            return '—' if value is None else f'{value:.6f}'
        for row in rows:
            decision = ('pass' if row['passed'] else 'fail') if row['status'] == 'completed' else row['status']
            text += (f"| {row['seed']} | {row['variant']} | {row['adversarial']} | "
                     + ' | '.join(number(row[k]) for k in ('cd8_loss', 'cd8_improvement',
                                  'biology_improvement', 'positive_cd8_logit_r2')) + f' | {decision} |\n')
        text += '\nPositive improvement means lower loss than the train-only constant baseline.\n'
        def passed(variant, adversarial=False):
            return [r['passed'] for r in rows if r['variant'] == variant
                    and r['adversarial'] == adversarial and r['status'] == 'completed']
        if passed('M') == [True] and passed('C') and not any(passed('C')):
            text += ('\nThe matched MLP positive control learned, while the current CNN failed with adversarial '
                     'loss and gradient reversal disabled. Adversarial interference is therefore not necessary '
                     'for the observed three-epoch CNN failure.\n')
        if passed('P') == [True] and passed('C') and not any(passed('C')):
            text += ('\nThe pointwise CNN learned with the same hidden-stack design. This supports an interaction '
                     'between trainable spatial mixing and the current deep stack/optimization setting; it does '
                     'not show that LayerNorm or dropout alone prevents learning.\n')
        if passed('shallow') == [True]:
            text += ('\nReducing the hidden 3×3 stack to one convolution passed on the screening seed. '
                     'This intervention was not confirmed on additional seeds.\n')
        if passed('residual'):
            text += (f"\nResidual connections passed on {sum(passed('residual'))}/{len(passed('residual'))} completed "
                     'disabled-adversarial seeds. Selection used screening-seed epoch-3 CD8 loss, '
                     'so this is not a multi-seed ranking against the shallow variant.\n')
        if passed('pointwise_init') == [False]:
            text += ('\nThe pointwise-initialized 3×3 network failed despite starting from P’s function. '
                     'Changing the initial function alone was insufficient when surrounding coefficients '
                     'were subsequently allowed to train. This does not isolate the precise optimization mechanism.\n')
        restored = [r for r in rows if r['adversarial'] and r['status'] == 'completed']
        if restored:
            text += (f"\nFresh restored-adversarial fits passed on {sum(r['passed'] for r in restored)}/{len(restored)} seeds. "
                     'The GRL schedule retained its 300-epoch horizon and reaches only about 0.0125 during '
                     'these three epochs (configured maximum 0.25). These results do not establish later-horizon behavior.\n')
        text += ('\nAudits measure biology-only gradients on a fixed training-only set; parameter changes are '
                 'net displacement over an epoch. Center/neighbor sensitivities are CD8 derivatives with respect '
                 'to latent features. Feature and prediction variation are descriptive evidence, not proof '
                 'of a unique causal mechanism. LayerNorm removal, dropout removal, and lower biology learning '
                 'rate were not tested when P passed, as specified by the branching protocol.\n'
                 '\nRandom streams: run seeds control initial shared weights and sampling; CUDA dropout uses '
                 'the common derived seed (run seed + 1000), also used separately for shape-changing CPU '
                 'initialization. The executed source snapshot and rng_streams.json preserve this detail. '
                 'The constructor cleanup makes CUDA seeding explicit without changing the executed training stream.\n')
    (study.root/'report.md').write_text(text)
    save_json(study.root/'completion.json', dict(conclusion=conclusion,
        elapsed_seconds=240*60-study.deadline.remaining(), remaining_seconds=study.deadline.remaining(),
        source_yaml_unchanged=file_hash(Path('dann/config.yaml')) == study.source_hash,
        archive_unchanged=all(file_hash(Path(p)) == h for p,h in study.archive_hashes.items())))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    study = Study(args.output.resolve(), args.archive.resolve())
    conclusion = 'Preflight incomplete.'
    try:
        study.prepare()
        conclusion = route(study.group, study.get)
    except Exception:
        failure = traceback.format_exc()
        save_json(study.root/'failure.json', dict(error=failure))
        print(failure, flush=True)
        conclusion = 'Diagnosis stopped on a preserved execution/preflight failure; no uncompleted fit is a scientific negative.'
    finally:
        if study.bundle is not None:
            close_bundle(study.bundle)
        report(study, conclusion)
    print(conclusion, flush=True)


if __name__ == '__main__':
    main()
