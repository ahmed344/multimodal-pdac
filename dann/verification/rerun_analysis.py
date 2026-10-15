"""Archive and repeat downstream analysis without modifying training artifacts.

Start with --roots COMBINATION ... --archive-root DIRECTORY. Resume using
--resume BATCH_DIRECTORY. No training entrypoint is invoked.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml


def save_json(path: Path, value: dict) -> None:
    """Publish batch evidence atomically."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inventory(root: Path, hashes: bool = False) -> dict:
    """Record file identity for immutable models or resumable stage artifacts."""
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink in managed output: {path}')
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(root))] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if hashes:
                result[str(path.relative_to(root))]['sha256'] = digest(path)
    return result


def artifacts_valid(root: Path, files: dict) -> bool:
    """Require every recorded artifact to remain present and unchanged."""
    return bool(files) and all((root / name).is_file() and
        (root / name).stat().st_size == value['size'] and
        (root / name).stat().st_mtime_ns == value['mtime_ns'] for name, value in files.items())


def archive_outputs(batch: Path, manifest: dict) -> None:
    """Move only inventoried analysis trees; reconcile interrupted renames."""
    for item in manifest['archives']:
        source, destination = Path(item['source']), Path(item['destination'])
        if source.exists() and destination.exists():
            raise FileExistsError(f'Archive collision: {destination}')
        if not source.exists() and not destination.exists():
            raise FileNotFoundError(f'Missing archive source and destination: {source}')
        if inventory(source if source.exists() else destination) != item['files']:
            raise ValueError(f'Archive inventory changed: {source}')
    for item in manifest['archives']:
        source, destination = Path(item['source']), Path(item['destination'])
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        item['state'] = 'archived'
        save_json(batch / 'manifest.json', manifest)
    manifest['archived'] = True
    save_json(batch / 'manifest.json', manifest)


def prepare(roots: list[Path], archive_root: Path, template: Path) -> Path:
    """Preflight all checkpoints and routing before archiving any results."""
    import torch
    from dann.config import load_config
    from dann.targets import checkpoint_analysis_config
    runtime = load_config(template)
    roots = sorted({root.resolve() for root in roots})
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError('Every requested combination root must exist.')
    runs, archives, inputs = [], [], {}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    batch = archive_root.resolve() / f'dann-analysis-rerun-{stamp}'
    if batch.exists():
        raise FileExistsError(batch)
    for root in roots:
        checkpoints = sorted(root.glob('*/model/best.pt'))
        if not checkpoints:
            raise FileNotFoundError(f'No checkpoints in {root}')
        for checkpoint_path in checkpoints:
            directory = checkpoint_path.parent.parent
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            candidate = copy.deepcopy(runtime)
            candidate['loss'] = copy.deepcopy(checkpoint['config']['loss'])
            candidate['training'] = copy.deepcopy(checkpoint['config']['training'])
            for key in ('path', 'context_path'):
                candidate['data'][key] = checkpoint['config']['data'][key]
            config = checkpoint_analysis_config(candidate, checkpoint)
            config['analysis']['execution']['cache_policy'] = 'reuse'  # Fresh roots; resume this batch only.
            if Path(config['analysis']['output_dir']).resolve() != directory / 'analysis':
                raise ValueError(f'Checkpoint output routing mismatch: {directory}')
            if Path(config['analysis']['checkpoint']).resolve() != checkpoint_path:
                raise ValueError(f'Checkpoint path mismatch: {checkpoint_path}')
            for key in ('path', 'context_path'):
                path = Path(config['data'][key]).resolve()
                stat = path.stat()
                inputs[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            relative = Path(root.name) / directory.name
            evidence = batch / 'batch_records' / relative
            run = dict(name=str(relative), directory=str(directory), config=str(evidence / 'config.yaml'),
                       model=inventory(directory / 'model', hashes=True), stages={}, effective_config=config)
            runs.append(run)
            for name in ('analysis', 'latent_variance'):
                source = directory / name
                if source.exists():
                    if source.is_symlink() or not source.is_dir():
                        raise ValueError(f'Unsafe archive source: {source}')
                    archives.append(dict(source=str(source), destination=str(batch / relative / name),
                                         files=inventory(source), state='pending'))
    if len({r['name'] for r in runs}) != len(runs):
        raise ValueError('Duplicate run names across roots.')
    archive_root.mkdir(parents=True, exist_ok=True)
    if any(Path(item['source']).stat().st_dev != archive_root.stat().st_dev for item in archives):
        raise ValueError('Archive must be on the same filesystem for atomic directory moves.')
    old_bytes = sum(v['size'] for item in archives for v in item['files'].values())
    required = max(old_bytes * 3, 50 * 1024**3)
    if shutil.disk_usage(archive_root).free < required:
        raise OSError(f'Need {required / 1024**3:.1f} GiB free for regenerated outputs and headroom.')
    batch.mkdir()
    for run in runs:
        path = Path(run['config'])
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(run.pop('effective_config'), sort_keys=False))
        load_config(path)
        run['config_sha256'] = digest(path)
    manifest = dict(created_utc=stamp, runs=runs, archives=archives, inputs=inputs,
                    archived=False, status='prepared', archive_bytes=old_bytes)
    save_json(batch / 'manifest.json', manifest)
    return batch


def commands(run: dict) -> list[tuple[str, list[str]]]:
    """Explicitly route every downstream artifact into its original run."""
    root = Path(run['directory']) / 'analysis'
    config = ['--config', run['config']]
    return [
        ('analyze', ['dann.analyze', *config, '--checkpoint', str(root.parent / 'model/best.pt'), '--output-dir', str(root)]),
        ('spatial', ['dann.spatial', *config, '--output', str(root / 'spatial/spatial_predictions.parquet'),
                     '--latent-output', str(root / 'spatial/spatial_latent.parquet'), '--overwrite']),
        ('spatial_heatmaps', ['dann.spatial_heatmaps', *config, '--output-dir', str(root / 'spatial/heatmaps')]),
        ('latent_variance', ['dann.latent_variance', *config, '--output-dir', str(root / 'latent_variance')]),
        ('peak_numerical', ['dann.verification.peak_analysis_audit', *config]),
        ('latent_numerical', ['dann.verification.latent_numerical_audit', *config]),
        ('peak_browser', ['dann.verification.peak_browser_audit', '--output-dir', str(root / 'peaks')]),
        ('latent_browser', ['dann.verification.latent_browser_audit', '--output-dir', str(root)]),
        ('artifacts', ['dann.verification.rerun_analysis', '--audit', run['config']]),
    ]


def stage_inventory(root: Path, stage: str) -> dict:
    """Capture stage-owned artifacts, excluding mutable downstream reports."""
    files = inventory(root)
    prefixes = {'spatial': 'spatial/', 'spatial_heatmaps': 'spatial/heatmaps/', 'latent_variance': 'latent_variance/'}
    if stage == 'analyze':
        return {k: v for k, v in files.items() if not k.startswith(('spatial/', 'latent_variance/'))
                and not k.endswith(('audit.json', 'artifact_audit.json'))}
    if stage in prefixes:
        return {k: v for k, v in files.items() if k.startswith(prefixes[stage])
                and (stage != 'spatial' or not k.startswith('spatial/heatmaps/'))}
    names = {'peak_numerical': 'peaks/reports/peak_audit.json', 'latent_numerical': 'latent/numerical_audit.json',
             'peak_browser': 'peaks/reports/peak_browser_audit.json', 'latent_browser': 'latent/browser_audit.json',
             'artifacts': 'artifact_audit.json'}
    return {k: v for k, v in files.items() if k == names[stage]}


def execute(batch: Path, manifest: dict, run: dict, stage: str, arguments: list[str]) -> bool:
    """Run one isolated stage, monitor memory, and terminate children on interruption."""
    import psutil
    evidence = Path(run['config']).parent
    previous = run['stages'].get(stage, {})
    attempt = previous.get('attempt', 0) + 1
    entry = dict(state='running', attempt=attempt, command=[sys.executable, '-m', *arguments],
                 started_utc=datetime.now(timezone.utc).isoformat())
    run['stages'][stage] = entry
    save_json(batch / 'manifest.json', manifest)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    started, peak_pss, peak_gpu = time.monotonic(), 0, 0
    command = [sys.executable, '-m', 'dann.verification.production_fits', '--worker',
               str(evidence / f'{stage}.{attempt}.resources.json'), *arguments]
    print(f"START {run['name']} {stage} attempt={attempt}", flush=True)
    with (evidence / f'{stage}.{attempt}.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        entry['pid'] = process.pid
        save_json(batch / 'manifest.json', manifest)
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    total = 0
                    for child in [parent, *parent.children(recursive=True)]:
                        try:
                            total += child.memory_full_info().pss
                        except psutil.Error:
                            pass
                    peak_pss = max(peak_pss, total)
                except psutil.Error:
                    pass
                try:
                    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10)
                    peak_gpu = max(peak_gpu, max(int(v) for v in gpu.split()))
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                if shutil.disk_usage(batch).free < 20 * 1024**3:
                    raise OSError('Stopping batch stage: less than 20 GiB disk headroom.')
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            entry.update(state='interrupted', error=traceback.format_exc())
            save_json(batch / 'manifest.json', manifest)
            raise
    entry.update(state='completed' if process.returncode == 0 else 'failed', returncode=process.returncode,
                 seconds=time.monotonic() - started, peak_host_pss_gib=peak_pss / 1024**3,
                 peak_device_used_mib=peak_gpu)
    if process.returncode == 0:
        entry['artifacts'] = stage_inventory(Path(run['directory']) / 'analysis', stage)
        if not entry['artifacts']:
            entry.update(state='failed', error='Stage produced no expected artifacts.')
    save_json(batch / 'manifest.json', manifest)
    print(f"{entry['state'].upper()} {run['name']} {stage} {entry['seconds']:.1f}s", flush=True)
    return entry['state'] == 'completed'


def orchestrate(batch: Path) -> dict:
    """Resume only this batch, rejecting changed checkpoints, configs, or inputs."""
    import fcntl
    with (batch / 'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((batch / 'manifest.json').read_text())
        for path, identity in manifest['inputs'].items():
            stat = Path(path).stat()
            if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != identity:
                raise ValueError(f'Input changed: {path}')
        for run in manifest['runs']:
            if inventory(Path(run['directory']) / 'model', hashes=True) != run['model']:
                raise ValueError(f"Training artifacts changed: {run['name']}")
            if digest(Path(run['config'])) != run['config_sha256']:
                raise ValueError(f"Configuration changed: {run['name']}")
        if not manifest['archived']:
            archive_outputs(batch, manifest)
        manifest['status'] = 'running'
        save_json(batch / 'manifest.json', manifest)
        for run in manifest['runs']:
            invalidate = False
            stages = commands(run)
            for index, (stage, arguments) in enumerate(stages):
                previous = run['stages'].get(stage, {})
                if not invalidate and previous.get('state') == 'completed' and artifacts_valid(
                        Path(run['directory']) / 'analysis', previous.get('artifacts', {})):
                    continue
                if not invalidate:
                    for dependent, _ in stages[index:]:
                        if dependent in run['stages']:
                            run['stages'][dependent]['state'] = 'stale'
                    save_json(batch / 'manifest.json', manifest)
                invalidate = True
                if not execute(batch, manifest, run, stage, arguments):
                    run['status'] = 'failed'
                    break
            else:
                run['status'] = 'completed'
            if inventory(Path(run['directory']) / 'model', hashes=True) != run['model']:
                raise ValueError(f"Training artifacts modified: {run['name']}")
            save_json(batch / 'manifest.json', manifest)
        manifest['status'] = 'completed' if all(r['status'] == 'completed' for r in manifest['runs']) else 'failed'
        save_json(batch / 'manifest.json', manifest)
        summary = dict(status=manifest['status'], archive=str(batch), runs=[dict(name=r['name'], status=r['status'],
            analysis=str(Path(r['directory']) / 'analysis'), stages={k: {f: v[f] for f in
            ('state', 'seconds', 'peak_host_pss_gib', 'peak_device_used_mib') if f in v}
            for k, v in r['stages'].items()}) for r in manifest['runs']])
        save_json(batch / 'summary.json', summary)
        return summary


def audit_outputs(config_path: Path) -> dict:
    """Verify full export coverage, artifact counts, and all trusted PNG files."""
    import h5py
    import numpy as np
    import pyarrow.parquet as pq
    from anndata.io import read_elem
    from dann.config import load_config
    from dann.heatmap_execution import validate_png
    config = load_config(config_path)
    root = Path(config['analysis']['output_dir'])
    with h5py.File(config['data']['context_path']) as handle:
        rows = len(read_elem(handle['obs'][handle['obs'].attrs['_index']]))
    for name in ('spatial_predictions.parquet', 'spatial_latent.parquet'):
        parquet = pq.ParquetFile(root / 'spatial' / name)
        assert parquet.metadata.num_rows == rows, name
        offset = 0
        for chunk in parquet.iter_batches(columns=['row_position'], batch_size=131072):
            values = chunk.column(0).to_numpy()
            np.testing.assert_array_equal(values, np.arange(offset, offset + len(values)))
            offset += len(values)
        if 'latent' in name:
            assert len([v for v in parquet.schema_arrow.names if v.startswith('latent_')]) == 256
        else:
            from dann.targets import target_contract
            assert json.loads(parquet.schema_arrow.metadata[b'dann_target_transform']) == target_contract(config)
            for target in config['data']['target_columns']:
                assert any(target in v for v in parquet.schema_arrow.names), target
    counts = dict(peak_png=len(list((root / 'peaks').rglob('*.png'))),
                  peak_html=len(list((root / 'peaks').rglob('*.html'))),
                  latent_html=len(list((root / 'latent').glob('*.html'))),
                  spatial_png=len(list((root / 'spatial/heatmaps').rglob('*.png'))))
    assert counts == dict(peak_png=78, peak_html=38, latent_html=5, spatial_png=16), counts
    assert not list(root.rglob('peptide_similarity_heatmap.png'))
    assert not (root.parent / 'latent_variance').exists()
    for name in ('latent_variance_summary.csv', 'pca_variance.csv', 'explained_variance.png'):
        assert (root / 'latent_variance' / name).is_file()
    pngs = {str(path.relative_to(root)): validate_png(path) for path in root.rglob('*.png')}
    report = dict(status='passed', tissue_rows=rows, counts=counts, png_dimensions=pngs)
    save_json(root / 'artifact_audit.json', report)
    return report


def main() -> int:
    """Start, resume, or audit a downstream-only batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path)
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit_outputs(args.audit), indent=2))
        return 0
    if args.resume:
        batch = args.resume.resolve()
    else:
        if not args.roots or not args.archive_root:
            parser.error('Use --roots and --archive-root, or --resume.')
        batch = prepare(args.roots, args.archive_root, args.config)
    print(f'BATCH {batch}', flush=True)
    result = orchestrate(batch)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
