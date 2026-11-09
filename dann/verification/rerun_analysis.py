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


def checkpoint_path(run: dict) -> Path:
    """Resolve the manifest checkpoint, preserving legacy best-only batches."""
    selected = run.get('checkpoint', 'best')
    if selected not in ('best', 'latest'):
        raise ValueError('Manifest checkpoint must be best or latest.')
    return Path(run['directory']) / 'model' / f'{selected}.pt'


def same_checkpoint(run: dict, stage: dict) -> bool:
    """Never reuse a completed stage from a different checkpoint selection."""
    return stage.get('checkpoint', 'best') == run.get('checkpoint', 'best')


def commands(run: dict) -> list[tuple[str, list[str]]]:
    """Explicitly route every downstream artifact into its original run."""
    root = Path(run['directory']) / 'analysis'
    config = ['--config', run['config']]
    checkpoint = ['--checkpoint', str(checkpoint_path(run))]
    scheduled = [
        ('analyze', ['dann.analyze', *config, *checkpoint, '--output-dir', str(root)]),
        ('spatial', ['dann.spatial', *config, *checkpoint, '--output', str(root / 'spatial/spatial_predictions.parquet'),
                     '--latent-output', str(root / 'spatial/spatial_latent.parquet'), '--overwrite']),
        ('spatial_heatmaps', ['dann.spatial_heatmaps', *config, '--output-dir', str(root / 'spatial/heatmaps')]),
        ('peak_numerical', ['dann.verification.peak_analysis_audit', *config]),
        ('latent_numerical', ['dann.verification.latent_numerical_audit', *config]),
        ('peak_browser', ['dann.verification.peak_browser_audit', '--output-dir', str(root / 'peaks')]),
        ('latent_browser', ['dann.verification.latent_browser_audit', '--output-dir', str(root)]),
        ('artifacts', ['dann.verification.rerun_analysis', '--audit', run['config']]),
    ]
    # Historical manifests predate feature groups; their omitted switches retain
    # the full workflow. Current templates honor the same downstream choices.
    effective = run.get('effective_config') or yaml.safe_load(Path(run['config']).read_text())
    if not isinstance(effective, dict) or 'analysis' not in effective:
        return scheduled
    from dann.config import analysis_feature_settings, peak_analysis_settings, peak_analysis_requested, spatial_heatmap_settings
    analysis = effective['analysis']
    f = analysis_feature_settings(analysis)
    peaks = peak_analysis_settings(analysis.get('peaks'))
    post = effective.get('post_training', {})
    heatmaps = spatial_heatmap_settings(analysis.get('spatial_heatmaps'))
    peak_enabled = analysis.get('enabled', True) and peak_analysis_requested(peaks)
    latent_tables = f['latent_exports']['tables'] and f['direct_umap']['tables'] and f['latent_interpretation']['tables']
    allow = dict(analyze=analysis.get('enabled', True),
        spatial=post.get('run_inference', True),
        spatial_heatmaps=analysis.get('enabled', True) and post.get('run_spatial_heatmaps', True) and heatmaps['enabled'] and (heatmaps['parameters'] or heatmaps['diagnostics']),
        peak_numerical=peak_enabled and peaks['tables'], peak_browser=peak_enabled and peaks['interactive'],
        latent_numerical=latent_tables, latent_browser=f['direct_umap']['interactive'] and f['prediction_scatter']['interactive'] and all(
            f['latent_interpretation'][k] for k in ('correlations', 'feature_umap', 'ihc_scatter')),
        artifacts=analysis.get('enabled', True))
    return [(name, args) for name, args in scheduled if allow[name]]


def stage_inventory(root: Path, stage: str) -> dict:
    """Capture stage-owned artifacts, excluding mutable downstream reports."""
    files = inventory(root)
    prefixes = {'spatial': 'spatial/', 'spatial_heatmaps': 'spatial/heatmaps/', 'latent_variance': 'latent_variance/'}
    if stage == 'analyze':
        return {k: v for k, v in files.items() if not k.startswith(('spatial/',))
                and not k.endswith(('audit.json', 'artifact_audit.json'))}
    if stage in prefixes:
        return {k: v for k, v in files.items() if k.startswith(prefixes[stage])
                and (stage != 'spatial' or not k.startswith('spatial/heatmaps/'))}
    names = {'peak_numerical': 'peaks/reports/peak_audit.json', 'latent_numerical': 'latent/numerical_audit.json',
             'peak_browser': 'peaks/reports/peak_browser_audit.json', 'latent_browser': 'latent/browser_audit.json',
             'artifacts': 'artifact_audit.json'}
    return {k: v for k, v in files.items() if k == names[stage]}


def recover_worker(run: dict, stage: str, previous: dict) -> None:
    """Stop a surviving invocation-owned worker before retrying an interrupted stage."""
    import psutil
    pid = previous.get('pid')
    if not pid:
        return
    try:
        process = psutil.Process(pid)
        if previous.get('process_created') != process.create_time():
            return  # PID was reused, or this historical receipt cannot identify it.
        evidence = Path(run['config']).parent
        expected = [sys.executable, '-m', 'dann.verification.production_fits', '--worker',
                    str(evidence/f'{stage}.{previous["attempt"]}.resources.json'), *previous['command'][2:]]
        if process.cmdline() != expected or os.getpgid(pid) != pid:
            raise RuntimeError(f'Cannot safely recover recorded worker {pid}.')
        os.killpg(pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except psutil.TimeoutExpired:
            os.killpg(pid, signal.SIGKILL)
            process.wait(timeout=20)
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass


def execute(batch: Path, manifest: dict, run: dict, stage: str, arguments: list[str]) -> bool:
    """Run one isolated stage, monitor memory, and terminate children on interruption."""
    import psutil
    evidence = Path(run['config']).parent
    previous = run['stages'].get(stage, {})
    recover_worker(run, stage, previous)
    attempt = previous.get('attempt', 0) + 1
    entry = dict(state='running', attempt=attempt, command=[sys.executable, '-m', *arguments],
                 checkpoint=run.get('checkpoint', 'best'),
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
        entry['process_created'] = psutil.Process(process.pid).create_time()
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
                if not invalidate and same_checkpoint(run, previous) and previous.get('state') == 'completed' and artifacts_valid(
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


def validate_html_links(path: Path) -> None:
    """Validate HTML resource attributes without mistaking embedded JavaScript for markup."""
    from html.parser import HTMLParser
    from urllib.parse import unquote, urlsplit

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key not in ('src', 'href') or not value:
                    continue
                parsed = urlsplit(value)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                if not (path.parent/unquote(parsed.path)).is_file():
                    raise ValueError(f'Broken report link in {path}: {value}')

    parser = Links()
    parser.feed(path.read_text())
    parser.close()


def audit_artifact_manifests(root: Path, config: dict, slides: list[str]) -> dict:
    """Check manifest-owned artifacts and configuration-derived heatmap coverage."""
    from dann.config import spatial_heatmap_settings, analysis_feature_settings, peak_analysis_settings, peak_analysis_requested
    f = analysis_feature_settings(config['analysis'])
    peak_settings = peak_analysis_settings(config['analysis'].get('peaks'))
    counts = {}
    sections = [(root, 'analysis_execution.json', 'analysis')]
    if config['analysis'].get('enabled', True) and peak_analysis_requested(peak_settings):
        sections.append((root/'peaks', 'peak_run.json', 'peak'))
    if (root/'latent/latent_run.json').is_file() and (f['direct_umap']['interactive'] or f['prediction_scatter']['interactive'] or f['latent_interpretation']['enabled']):
        sections.append((root/'latent', 'latent_run.json', 'latent'))
    for directory, name, label in sections:
        report = json.loads((directory/name).read_text())
        artifacts = report['artifacts']
        if not artifacts or len(set(artifacts)) != len(artifacts):
            raise ValueError(f'Empty or duplicate artifact manifest: {name}')
        for relative in artifacts:
            path = directory/relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.is_file():
                raise ValueError(f'Missing or unsafe manifest artifact: {path}')
            if path.suffix == '.html':
                validate_html_links(path)
        for suffix in ('.png', '.html'):
            listed = {relative for relative in artifacts if Path(relative).suffix == suffix}
            counts[label+suffix.replace('.', '_')] = len(listed)
        if label == 'peak' and peak_settings['spatial'] and peak_settings['spatial_cd8'] and (peak_settings['plots'] or peak_settings['interactive']):
            settings = json.loads((directory/'settings/cd8_family_comparison_settings.json').read_text())
            if set(settings['slides']) != set(slides):
                raise ValueError('Peak spatial viewers omit input slides.')
            families = settings['family_ids']
            expected = {f'spatial/families/family_{family:02d}_spatial.png' for family in families}
            if peak_settings['plots'] and peak_settings['spatial_families'] and not expected.issubset(artifacts):
                raise ValueError('Missing peak family spatial figures.')
    settings = spatial_heatmap_settings(config['analysis'].get('spatial_heatmaps'))
    if not settings['enabled'] or not config.get('post_training', {}).get('run_spatial_heatmaps', True) or not (settings['parameters'] or settings['diagnostics']):
        return counts
    heatmaps = root/'spatial/heatmaps'
    report = json.loads((heatmaps/'heatmap_run.json').read_text())
    settings = spatial_heatmap_settings(config['analysis'].get('spatial_heatmaps'))
    targets = config['data']['target_columns']
    expected = {str(heatmaps/region/f'{prefix}_{target}_{region}.png')
                for region in settings['regions'] for target in targets
                for prefix, flag in (('spatial_heatmap', 'parameters'), ('spatial_heatmap_diagnostics', 'diagnostics')) if settings[flag]}
    actual = {str(Path(item['path'])) for item in report['outputs']}
    if actual != expected:
        raise ValueError('Spatial heatmap coverage differs from targets/regions.')
    # Each target has separate named figures; the CLI's plotting order is not
    # the model's ordered prediction schema (validated separately above).
    if (set(report['densities']) != set(targets) or len(report['densities']) != len(targets)
            or report['settings'] != settings):
        raise ValueError('Spatial heatmap targets or settings changed.')
    # New manifests explicitly record slide coverage; old reports predate this field.
    if 'slides' in report and set(report['slides']) != set(slides):
        raise ValueError('Spatial heatmaps omit input slides.')
    from dann.heatmap_execution import validate_png
    dpi = int(config['analysis']['figure_dpi'])
    if report['dpi'] != dpi:
        raise ValueError('Spatial heatmap DPI changed.')
    for item in report['outputs']:
        width = 30 if Path(item['path']).name.startswith('spatial_heatmap_diagnostics_') else 36
        size = [width*dpi, 4*len(slides)*dpi]
        if item['size'] != size:
            raise ValueError('Spatial heatmap dimensions disagree with input slide count.')
        validate_png(item['path'], size)
    counts['spatial_png'] = len(expected)
    return counts


def audit_outputs(config_path: Path, *, write_report: bool = True) -> dict:
    """Verify complete identities, finite model exports, manifests, links and images."""
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    from anndata.io import read_elem
    from dann.config import load_config
    from dann.heatmap_execution import validate_png
    from dann.spatial import validate_support_output, _file_identity
    from dann.workflow import _validate_exports
    config = load_config(config_path)
    root = Path(config['analysis']['output_dir'])
    checkpoint_path = Path(config['analysis']['checkpoint'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    width = int(checkpoint['config']['model']['latent_dim'])
    if config['model']['latent_dim'] != width:
        raise ValueError('Runtime/checkpoint latent dimensions disagree.')
    from dann.config import analysis_feature_settings
    f = analysis_feature_settings(config['analysis'])
    split = config['analysis']['split']
    expected = np.asarray(checkpoint['split_indices'][split])
    with h5py.File(config['data']['path']) as handle:
        labels = np.asarray(read_elem(handle['obs'][config['data']['batch_column']])).astype(str)
        scope = config['analysis'].get('sample_scope', 'lesion' if f['variance']['enabled'] or f['pca_umap']['enabled'] else 'split')
        if scope == 'lesion':
            lesion = np.asarray(read_elem(handle['obs'][config['analysis'].get('lesion_column', 'Lesion')]), dtype=bool)
            expected = expected[lesion[expected]]
    slides = list(pd.unique(labels))
    rows = None
    if config.get('post_training', {}).get('run_inference', True):
        with h5py.File(config['data']['context_path']) as handle:
            rows = len(read_elem(handle['obs'][handle['obs'].attrs['_index']]))
            slides = list(map(str, pd.unique(read_elem(handle['obs'][config['data']['batch_column']]))))
        _validate_exports(config, checkpoint, checkpoint_path, Path(config['data']['context_path']), predictions=True, latents=True)
        prediction_path = Path(config['spatial']['output'])
        support = prediction_path.with_suffix('.support.parquet')
        provenance = json.loads(prediction_path.with_suffix('.provenance.json').read_text())
        if 'support_identity' in provenance and provenance['support_identity'] != _file_identity(support):
            raise ValueError('Support provenance mismatch.')
        if checkpoint.get('architecture', {}).get('spatial', {}).get('mode') == 'spatial' or support.exists():
            validate_support_output(support, rows)
    if config['analysis']['max_samples'] is not None:
        raise ValueError('Full batch audit requires uncapped validation analysis.')
    saved_rows = expected
    if f['latent_exports']['tables']:
        latent_path = root/'tables/latent_embeddings.csv'
        latent_columns = [f'latent_{i}' for i in range(width)]
        observed_columns = pd.read_csv(latent_path, nrows=0).columns.tolist()
        if [c for c in observed_columns if c.startswith('latent_')] != latent_columns:
            raise ValueError('Validation latent width/order differs from checkpoint.')
        identities = []
        for chunk in pd.read_csv(latent_path, chunksize=8192):
            identities.append(chunk.row_id.to_numpy())
            if not np.isfinite(chunk[latent_columns].to_numpy()).all():
                raise ValueError('Nonfinite validation latent values.')
        saved_rows = np.concatenate(identities)
        np.testing.assert_array_equal(saved_rows, expected)
    prediction_columns = [f'{prefix}_{target}' for target in config['data']['target_columns']
                          for prefix in ('pi', 'mu', 'sigma', 'density_mean')]
    numerical_columns = ['umap_1', 'umap_2', *prediction_columns]
    for name, directory in (('direct_umap', 'umap'), ('pca_umap', 'umap_pca')):
        if not f[name]['tables']:
            continue
        settings_path = root/directory/'settings.json'
        if settings_path.is_file() and json.loads(settings_path.read_text()).get('skipped'):
            continue
        umap = pd.read_csv(root/directory/'tables/latent_umap.csv', usecols=['row_id', *numerical_columns])
        np.testing.assert_array_equal(umap.row_id, saved_rows)
        if not np.isfinite(umap[numerical_columns].to_numpy()).all():
            raise ValueError('Nonfinite validation UMAP or prediction parameters.')
    counts = audit_artifact_manifests(root, config, slides)
    if f['variance']['enabled']:
        provenance_path = root/'latent_variance/provenance.json'
        skipped = provenance_path.is_file() and json.loads(provenance_path.read_text()).get('skipped')
        if not skipped and f['variance']['tables']:
            variance = pd.read_csv(root/'latent_variance/latent_variance_summary.csv').iloc[0]
            if (variance.latent_width, variance.n_observations, variance.n_slides) != (width, len(expected), len(np.unique(labels[expected]))):
                raise ValueError('Variance analysis coverage mismatch.')
            if len(pd.read_csv(root/'latent_variance/pca_variance.csv')) != width:
                raise ValueError('Incomplete variance spectrum.')
        if not skipped and f['variance']['explained_variance']:
            validate_png(root/'latent_variance/explained_variance.png')
    pngs = {str(path.relative_to(root)): validate_png(path) for path in root.rglob('*.png')}
    execution = json.loads((root/'analysis_execution.json').read_text())
    evaluated_epoch = int(checkpoint['epoch']) + 1
    if execution['checkpoint_epoch'] != evaluated_epoch or execution['source_identity']['checkpoint'] != digest(checkpoint_path):
        raise ValueError('Validation analysis checkpoint provenance mismatch.')
    report = dict(status='passed', tissue_rows=rows, validation_rows=len(saved_rows), latent_width=width,
                  checkpoint=str(checkpoint_path), evaluated_epoch=evaluated_epoch,
                  slides=slides, checkpoint_sha256=digest(checkpoint_path), counts=counts, png_dimensions=pngs)
    if write_report:
        save_json(root/'artifact_audit.json', report)
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
