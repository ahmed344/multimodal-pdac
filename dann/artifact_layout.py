"""Stable destinations and migration of recognized generated analysis artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path


def peak_relative_path(name: str) -> Path:
    """Map a peak artifact basename to its organized location."""
    path = Path(name)
    if path.name != name:
        raise ValueError('Expected an artifact basename.')
    if name.startswith('cd8_families_') and path.suffix in ('.png', '.html'):
        group = 'spatial/cd8_comparison'
    elif name.startswith('family_') and name.endswith('_spatial.png'):
        group = 'spatial/families'
    elif path.suffix in ('.csv', '.parquet'):
        group = 'tables'
    elif path.suffix == '.npy':
        group = 'hierarchy'
    elif name.endswith('_settings.json'):
        group = 'settings'
    elif name.endswith('_audit.json'):
        group = 'reports'
    elif name.startswith('peak_umap'):
        group = 'umap'
    elif name.startswith('prevalence_intensity'):
        group = 'prevalence_intensity'
    elif name == 'peak_family_ihc_correlations.png':
        group = 'correlations'
    elif path.suffix in ('.png', '.html'):
        group = 'similarity/families' if name.startswith('family_') and name != 'family_similarity.png' else 'similarity/global'
    else:
        return path
    return Path(group)/name


def peak_file(directory: Path, name: str) -> Path:
    """Read both legacy flat peak artifacts and the organized layout."""
    organized = directory/peak_relative_path(name)
    return directory/name if (directory/name).exists() and not organized.exists() else organized


def remove_owned(directory: Path, names: set[str]) -> None:
    """Remove only safe, manifest-owned relative files, never outside the root."""
    root = directory.resolve()
    parents = set()
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            continue
        candidate = directory/relative
        if candidate.resolve().is_relative_to(root):
            candidate.unlink(missing_ok=True)
            parents.update(p for p in candidate.parents if p != directory and p.is_relative_to(directory))
    for child in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        if child.is_dir() and not child.is_symlink():
            try:
                child.rmdir()
            except OSError:
                pass


def analysis_relative_path(relative: Path) -> Path:
    """Group known latent, similarity, and diagnostic outputs; leave others alone."""
    name = relative.name
    if relative.parent == Path('umap'):
        if name == 'latent_umap.csv':
            group = 'tables'
        elif name.startswith('latent_umap_ziln_diagnostics_'):
            group = 'diagnostics'
        elif name.startswith('latent_umap_ziln_'):
            group = 'parameters'
        elif name.startswith('latent_umap') and relative.suffix == '.png':
            group = 'overview'
        else:
            return relative
        return Path('umap')/group/name
    if relative.parent != Path('.'):
        return relative
    if name in ('latent_embeddings.csv', 'ziln_density_scatter.csv'):
        return Path('tables')/name
    if name in ('embedding_cosine_similarity.npy', 'peptide_families.csv'):
        return Path('similarity')/name
    if name == 'ziln_density_scatter.png':
        return Path('density')/name
    for split in ('train', 'validation', 'test'):
        if name in (f'{split}_{suffix}.csv' for suffix in ('density_errors', 'neighborhood_support', 'spatial_contrasts')):
            return Path('diagnostics')/split/name
    return relative


def analysis_file(directory: Path, relative: str) -> Path:
    """Resolve a known export in either the organized or legacy layout."""
    organized = directory/analysis_relative_path(Path(relative))
    legacy = directory/relative
    return legacy if legacy.exists() and not organized.exists() else organized


def organize_analysis(directory: Path) -> None:
    """Move recognized generated files and update the existing execution manifest."""
    migrate_latent_variance(directory)
    remove_retired_similarity_heatmap(directory)
    moved = {}
    for path in [*directory.glob('*'), *(directory/'umap').glob('*')]:
        if not path.is_file():
            continue
        old = path.relative_to(directory)
        new = analysis_relative_path(old)
        if new != old:
            destination = directory/new
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination)
            moved[str(old)] = str(new)
    report_path = directory/'analysis_execution.json'
    if moved and report_path.exists():
        report = json.loads(report_path.read_text())
        report['artifacts'] = [moved.get(name, name) for name in report.get('artifacts', [])]
        temporary = report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        os.replace(temporary, report_path)


def migrate_latent_variance(directory: Path) -> None:
    """Move recognized legacy variance exports, rejecting conflicts before mutation."""
    import hashlib
    source = directory.parent/'latent_variance'
    destination = directory/'latent_variance'
    names = ('latent_variance_summary.csv', 'pca_variance.csv', 'explained_variance.png')
    pending = []
    for name in names:
        old, new = source/name, destination/name
        if not old.is_file():
            continue
        if new.exists() and hashlib.sha256(old.read_bytes()).digest() != hashlib.sha256(new.read_bytes()).digest():
            raise ValueError(f'Conflicting latent variance output: {new}')
        pending.append((old, new))
    for old, new in pending:
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            old.unlink()
        else:
            os.replace(old, new)
    if source.is_dir() and not any(source.iterdir()):
        source.rmdir()


def remove_retired_similarity_heatmap(directory: Path) -> None:
    """Retire only the obsolete standalone PNG and its manifest entries."""
    retired = {'peptide_similarity_heatmap.png', 'similarity/peptide_similarity_heatmap.png'}
    remove_owned(directory, retired)
    report_path = directory/'analysis_execution.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        artifacts = report.get('artifacts', [])
        remaining = [name for name in artifacts if name not in retired]
        if remaining != artifacts:
            report['artifacts'] = remaining
            temporary = report_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, indent=2))
            os.replace(temporary, report_path)
