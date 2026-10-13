"""Reproduce isolated analysis stage timings and compare full exported tables.

Run after a complete optimized analysis has populated its private stage cache.
Each screen includes a warmup and three measured repetitions. No production
artifacts are modified; --output must be a separate benchmark directory.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from dann.analysis_execution import (
    RenderQueue, StageCache, digest_file, extract_prepared, sampled_density_mean, write_artifacts,
)
from dann.config import analysis_execution_settings, load_config, resolve_device
from dann.spatial_execution import InferenceProgress, inference_performance


def read_stage(cache: Path, stage: str):
    """Read the latest validated generation in an explicitly selected run cache."""
    metrics = cache.parent / 'analysis_execution.json'
    entries = json.loads(metrics.read_text()).get('cache_entries', {}) if metrics.exists() else {}
    path = (cache / stage / entries[stage] / 'manifest.json' if stage in entries else
            max((cache / stage).glob('*/manifest.json'), key=lambda p: p.stat().st_mtime_ns))
    manifest = json.loads(path.read_text())
    arrays = {}
    for name, info in manifest['arrays'].items():
        source = path.parent / info['file']
        if digest_file(source) != info['sha256']:
            raise ValueError(f'Corrupt benchmark input: {source}')
        arrays[name] = np.load(source, mmap_mode='r', allow_pickle=False)
    return arrays, manifest.get('metadata', {})


def measure(function, *, cuda=False):
    """Include one warmup; retain three synchronized wall-clock measurements."""
    timings, result = [], None
    for repeat in range(4):
        result = None
        if cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = function()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if repeat:
            timings.append(elapsed)
    return dict(seconds=timings, median_seconds=statistics.median(timings)), result


def sampling_screen(config, cache, progress):
    from dann.ziln import sampled_density_mean as reference
    arrays, _ = read_stage(cache, 'extracted-' + config['analysis']['split'])
    values = [arrays[key] for key in ('pi', 'mu', 'sigma')]
    samples, seed = config['analysis']['density_mc_samples'], config['analysis']['density_mc_seed']
    rng = np.random.default_rng(seed)
    expected = reference(*values, num_samples=samples, rng=rng)
    state = rng.bit_generator.state
    results = []
    for workers in (0, 1, 2, 4):
        progress.set_phase(f'sampling screen: {workers} workers', len(expected))
        def run():
            candidate_rng = np.random.default_rng(seed)
            actual = sampled_density_mean(*values, num_samples=samples, rng=candidate_rng, workers=workers)
            np.testing.assert_array_equal(actual, expected)
            assert candidate_rng.bit_generator.state == state
            return None
        timing, _ = measure(run)
        results.append(dict(workers=workers, rows=len(expected), bitwise_equal=True, **timing))
        print(json.dumps(results[-1]), flush=True)
    return results


def gpu_screen(config, checkpoint_path, progress):
    from torch.utils.data import DataLoader
    from dann.analyze import load_model
    from dann.data_loader import create_data_bundle
    from dann.execution import PreparedLoader, build_validation_features, validation_execution
    from dann.targets import checkpoint_analysis_config
    from dann.tiles import tile_collate
    from dann.spatial_execution import release_preparation

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint_analysis_config(config, checkpoint)
    settings = analysis_execution_settings(config['analysis']['execution'])
    performance = inference_performance(config, {'performance': settings['performance']}, settings)
    config['performance'] = performance
    bundle = create_data_bundle(config, checkpoint.get('split_indices'), checkpoint.get('data_identity'),
                                requested_splits=['test', 'validation'], progress=progress)
    device = resolve_device(config['training']['device'])
    model, _ = load_model(checkpoint_path, config, bundle, device, checkpoint=checkpoint)
    def loader(tiles):
        return PreparedLoader(DataLoader(bundle.datasets['test'], batch_size=tiles, collate_fn=tile_collate),
                              performance, config['model'], validation=True)
    results = dict(encoding=[], prediction=[])
    try:
        with torch.inference_mode():
            features = None
            for rows, peaks in ((1024, 262144), (2048, 524288), (4096, 1048576)):
                performance.update(feature_cache_rows=rows, validation_peak_budget=peaks)
                progress.set_phase(f'encoding screen: {rows} rows, {peaks} peaks')
                with validation_execution(model, performance):
                    timing, features = measure(lambda: build_validation_features(model, loader(64), device,
                        context_datasets=list(bundle.datasets.values())), cuda=device.type == 'cuda')
                results['encoding'].append(dict(rows=rows, peaks=peaks, **timing))
                print(json.dumps(results['encoding'][-1]), flush=True)
            expected = None
            for tiles, gat in ((32, 32), (64, 32), (128, 64)):
                for workers in (2, 4, 8):
                    performance.update(validation_gat_tiles=gat, preparation_workers=workers,
                                       prepared_batches=2*workers)
                    prepared = loader(tiles)
                    prepared.inference = True
                    with validation_execution(model, performance):
                        timing, actual = measure(lambda: extract_prepared(model, prepared, device, features, progress),
                                                 cuda=device.type == 'cuda')
                    if expected is None:
                        expected = actual
                    for key in ('latent', 'pi', 'mu', 'sigma'):
                        np.testing.assert_allclose(actual[key], expected[key], rtol=5e-5, atol=5e-5)
                    np.testing.assert_array_equal(actual['row_ids'], expected['row_ids'])
                    results['prediction'].append(dict(tiles=tiles, gat=gat, preparation_workers=workers,
                        preparation_wait_seconds=prepared.wait_seconds, **timing))
                    print(json.dumps(results['prediction'][-1]), flush=True)
    finally:
        for current in bundle.loaders.values():
            release_preparation(current)
    return results


def rendering_screen(config, checkpoint_path, cache, output, progress):
    from dann.data_loader import load_anndata_metadata
    from dann.targets import checkpoint_analysis_config, target_contract
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint_analysis_config(config, checkpoint)
    settings = analysis_execution_settings(config['analysis']['execution'])
    performance = inference_performance(config, {'performance': settings['performance']}, settings)
    data = config['data']
    metadata = load_anndata_metadata(Path(data['path']), data['target_columns'], data['batch_column'],
                                    data['matrix_key'], data.get('cd8_normalization'))
    extracted = {}
    for split in dict.fromkeys([config['analysis']['split'], config['analysis']['ziln_scatter_split']]):
        extracted[split] = read_stage(cache, 'extracted-' + split)[0]
        extracted[split]['density_mean'] = read_stage(cache, 'density-' + split)[0]['means']
        extracted[split]['target_transform'] = target_contract(config)
    coordinates = read_stage(cache, 'umap')[0]['coordinates']
    peptides, activity = read_stage(cache, 'peptides')[0], read_stage(cache, 'activity')[0]
    staged = output / 'render-artifacts'
    staged.mkdir(exist_ok=True)
    queue = RenderQueue(output / 'render-arrays')
    progress.set_phase('rendering benchmark setup')
    write_artifacts(config, checkpoint_path, int(checkpoint['epoch'])+1, metadata,
        extracted[config['analysis']['split']], extracted[config['analysis']['ziln_scatter_split']],
        coordinates, peptides, activity, staged, queue.submit, table_writer=queue.table)
    results = []
    for workers in (4, 8, 16, 32):
        timing, _ = measure(lambda: queue.run(workers, performance, progress))
        results.append(dict(workers=workers, tasks=len(queue.tasks), **timing))
        print(json.dumps(results[-1]), flush=True)
    return results


def compare_exports(reference: Path, actual: Path) -> dict:
    """Compare identities/schemas exactly; expose UMAP and sampling differences."""
    import pandas as pd
    from PIL import Image
    expected = {path.relative_to(reference) for path in reference.rglob('*')
                if path.is_file() and not any(part.startswith('.') for part in path.relative_to(reference).parts)}
    observed = {path.relative_to(actual) for path in actual.rglob('*')
                if path.is_file() and not any(part.startswith('.') for part in path.relative_to(actual).parts)}
    assert expected == observed - {Path('analysis_execution.json')}, (expected-observed, observed-expected)
    report = dict(files=len(expected), tables={}, umap_coordinates_compared=False,
                  fp32_tolerance=dict(rtol=5e-5, atol=5e-5))
    for relative in sorted(expected):
        a, b = reference / relative, actual / relative
        if relative.suffix == '.png':
            with Image.open(a) as first, Image.open(b) as second:
                assert first.size == second.size, relative
        elif relative.suffix == '.npy':
            np.testing.assert_array_equal(np.load(a), np.load(b))
        elif relative.suffix == '.json':
            assert json.loads(a.read_text()) == json.loads(b.read_text())
        elif relative.suffix == '.csv':
            stats = dict(rows=0, max_absolute_difference=0., density_outlier_cells=0)
            from itertools import zip_longest
            for first, second in zip_longest(pd.read_csv(a, chunksize=16384, float_precision='round_trip'),
                                            pd.read_csv(b, chunksize=16384, float_precision='round_trip')):
                assert first is not None and second is not None, relative
                assert first.columns.tolist() == second.columns.tolist(), relative
                assert len(first) == len(second), relative
                stats['rows'] += len(first)
                for name in first:
                    if name in ('umap_1', 'umap_2'):
                        continue  # The agreed GPU tolerance can change a seeded UMAP layout.
                    x, y = first[name].to_numpy(), second[name].to_numpy()
                    if (x.dtype.kind not in 'f' or name.startswith(('raw_', 'valid_', 'label_'))
                            or name.startswith('Density_') or 'support' in relative.name
                            or relative.name == 'peptide_families.csv'):
                        pd.testing.assert_series_equal(first[name], second[name], check_exact=True)
                    elif 'density_mean' in name:
                        # Different FP32 pi values can flip rare discrete hurdles.
                        stats['density_outlier_cells'] += int((~np.isclose(x, y, rtol=5e-5, atol=5e-5, equal_nan=True)).sum())
                    else:
                        np.testing.assert_allclose(x, y, rtol=5e-5, atol=5e-5, equal_nan=True,
                                                   err_msg=f'{relative}: {name}')
                    if x.dtype.kind == 'f':
                        finite = np.isfinite(x) & np.isfinite(y)
                        if finite.any():
                            stats['max_absolute_difference'] = max(stats['max_absolute_difference'],
                                                                   float(np.max(np.abs(x[finite]-y[finite]))))
            report['tables'][str(relative)] = stats
    return report


def audit_density(reference, actual, config, progress):
    """Replay the unchanged sampler for all exported means, including hurdle flips."""
    import pandas as pd
    from dann.ziln import sampled_density_mean as legacy_sampler
    cache = actual / '.analysis-cache'
    names = json.loads((actual / 'target_transform.json').read_text())['target_columns']
    provenance = json.loads((actual / 'analysis_execution.json').read_text())
    sampling = provenance.get('density_sampling', {})
    samples = sampling.get('samples', config['analysis']['density_mc_samples'])
    seed = sampling.get('seed', config['analysis']['density_mc_seed'])
    main, scatter = config['analysis']['split'], config['analysis']['ziln_scatter_split']
    legacy_rng, optimized_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    report = {}
    for split in dict.fromkeys([main, scatter]):
        arrays = read_stage(cache, 'extracted-' + split)[0]
        means = read_stage(cache, 'density-' + split)[0]['means']
        values = [arrays[key] for key in ('pi', 'mu', 'sigma')]
        progress.set_phase(f'{split} frozen-parameter RNG replay', len(means))
        verified = legacy_sampler(*values, num_samples=samples, rng=optimized_rng)
        np.testing.assert_array_equal(verified, means)
        if split == main:
            columns = ['row_id', *[prefix+target for prefix in ('pi_', 'mu_', 'sigma_', 'density_mean_') for target in names]]
            frame = pd.read_csv(reference / 'umap/latent_umap.csv', usecols=columns, float_precision='round_trip')
            np.testing.assert_array_equal(frame.row_id, arrays['row_ids'])
            old_values = [frame[[prefix+target for target in names]].to_numpy(dtype='float32')
                          for prefix in ('pi_', 'mu_', 'sigma_')]
            old = legacy_sampler(*old_values, num_samples=samples, rng=legacy_rng)
            np.testing.assert_array_equal(old, frame[['density_mean_'+target for target in names]].to_numpy())
            old_export, new_export = old, means
        else:
            frame = pd.read_csv(reference / 'ziln_density_scatter.csv', float_precision='round_trip')
            lookup = np.full(int(arrays['row_ids'].max())+1, -1, dtype=np.int64)
            lookup[arrays['row_ids']] = np.arange(len(means))
            rows = lookup[frame.row_id.to_numpy()]
            targets = frame.target.map({name: i for i, name in enumerate(names)}).to_numpy()
            assert np.all(rows >= 0)
            # Non-exported zero observations still consume their normal RNG slots.
            # Their parameters cannot affect the other cells' means or RNG state.
            old_values = [value.copy() for value in values]
            for value, name in zip(old_values, ('pi', 'mu', 'sigma')):
                value[rows, targets] = frame['predicted_'+name].to_numpy(dtype='float32')
            old = legacy_sampler(*old_values, num_samples=samples, rng=legacy_rng)
            np.testing.assert_array_equal(old[rows, targets], frame.predicted_density_mean.to_numpy())
            old_export, new_export = old[rows, targets], means[rows, targets]
        assert legacy_rng.bit_generator.state == optimized_rng.bit_generator.state
        difference = np.abs(new_export-old_export)
        report[split] = dict(exported_means=int(old_export.size),
            outside_fp32_tolerance=int((~np.isclose(old_export, new_export, rtol=5e-5, atol=5e-5)).sum()),
            maximum_absolute_difference=float(difference.max()),
            reference_replay_bitwise_equal=True, final_rng_states_equal=True)
        progress.update(len(means))
    return report


def verify_umap(actual, config, progress):
    """Refit the exact full cached latent matrix with its recorded seed/settings."""
    from dann.analyze import fit_latent_umap
    cache = actual / '.analysis-cache'
    latent = read_stage(cache, 'extracted-' + config['analysis']['split'])[0]['latent']
    expected = read_stage(cache, 'umap')[0]['coordinates']
    analysis = {**config['analysis'],
                **json.loads((actual / 'analysis_execution.json').read_text()).get('umap_settings', {})}
    progress.set_phase('frozen latent UMAP verification', len(latent))
    coordinates = fit_latent_umap(np.array(latent, copy=True), analysis)
    np.testing.assert_array_equal(coordinates, expected)
    progress.update(len(latent))
    return dict(rows=len(latent), bitwise_equal=True, seed=analysis['umap_random_state'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--analysis-dir', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('gpu', 'sampling', 'rendering', 'compare', 'audit-density', 'verify-umap'), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'compare':
        result = compare_exports(args.reference, args.analysis_dir)
    else:
        config = load_config(args.config)
        checkpoint = Path(config['analysis'].get('checkpoint') or Path(config['training']['output_dir']) / 'best.pt')
        with InferenceProgress() as progress:
            if args.stage == 'verify-umap':
                result = verify_umap(args.analysis_dir, config, progress)
            elif args.stage == 'audit-density':
                result = audit_density(args.reference, args.analysis_dir, config, progress)
            elif args.stage == 'gpu':
                result = gpu_screen(config, checkpoint, progress)
            elif args.stage == 'sampling':
                result = sampling_screen(config, args.analysis_dir / '.analysis-cache', progress)
            else:
                result = rendering_screen(config, checkpoint, args.analysis_dir / '.analysis-cache', args.output, progress)
    (args.output / f'{args.stage}.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
