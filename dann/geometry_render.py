"""CD8 parameter exports and rendering from frozen geometry experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd

from dann.analysis_execution import digest_file, fingerprint, load_checkpoint_snapshot
from dann.config import latent_geometry_settings


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f'Missing geometry input: {path}')
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def plot_settings(analysis: Mapping[str, Any]) -> dict:
    """Use reference CD8 styling at 300 DPI, independent of layout settings."""
    defaults = dict(point_size=.1, density_cmap='jet', density_vmin_percentile=1.,
                    density_vmax_percentile=99.)
    return dict(figure_dpi=300, **{key: analysis.get(key, value) for key, value in defaults.items()})


def _row_ids(frame: pd.DataFrame, label: str) -> np.ndarray:
    if 'row_id' not in frame:
        raise ValueError(f'{label} requires row_id.')
    rows = frame.row_id.to_numpy()
    if rows.dtype.kind not in 'iu' or not len(rows) or np.any(rows < 0) or len(np.unique(rows)) != len(rows):
        raise ValueError(f'{label} requires unique nonnegative integer row IDs.')
    return rows


def validate_parameters(table: pd.DataFrame, observations: pd.DataFrame) -> None:
    """Reject invalid predictions, changed observations, or reordered export rows."""
    required = {'row_id', 'slide', 'Density_CD8', 'pi', 'presence_probability', 'mu', 'sigma', 'sampled_mean'}
    if not required.issubset(table):
        raise ValueError(f'Missing CD8 parameter columns: {sorted(required-set(table))}')
    if not np.array_equal(_row_ids(table, 'CD8 parameters'), _row_ids(observations, 'Observations')):
        raise ValueError('CD8 parameter row alignment differs from validation observations.')
    if table.slide.isna().any() or table.slide.astype(str).str.strip().eq('').any():
        raise ValueError('Missing CD8 slide identity.')
    if not np.array_equal(table.slide.astype(str), observations.slide.astype(str)):
        raise ValueError('CD8 slide identities differ from validation observations.')
    numeric = ['Density_CD8', 'pi', 'presence_probability', 'mu', 'sigma', 'sampled_mean']
    if not np.isfinite(table[numeric].to_numpy(float)).all():
        raise ValueError('Nonfinite CD8 parameters.')
    for key in ('Density_CD8', 'pi', 'presence_probability', 'sampled_mean'):
        if not table[key].between(0, 1).all():
            raise ValueError(f'CD8 {key} must be inside [0, 1].')
    if not (table.sigma > 0).all() or not np.allclose(table.presence_probability, 1-table.pi, rtol=0, atol=1e-12):
        raise ValueError('Invalid CD8 sigma or presence probability.')
    if not np.array_equal(table.Density_CD8.to_numpy(), observations.Density_CD8.to_numpy()):
        raise ValueError('CD8 observed densities differ from validation observations.')


def cd8_parameter_table(data: Mapping[str, np.ndarray], frame: pd.DataFrame,
                        targets: list[str], analysis: Mapping[str, Any]) -> pd.DataFrame:
    """Select CD8 by name and compute one seeded unconditional mean per row."""
    from dann.ziln import sampled_density_mean
    if targets.count('Density_CD8') != 1 or len(set(targets)) != len(targets):
        raise ValueError('Geometry requires unique targets including Density_CD8.')
    if 'row_ids' in data and not np.array_equal(data['row_ids'], _row_ids(frame, 'Observations')):
        raise ValueError('Inference parameter row alignment differs from observations.')
    index = targets.index('Density_CD8')
    table = frame[['row_id', 'slide', 'Density_CD8']].copy()
    for name in ('pi', 'mu', 'sigma'):
        values = np.asarray(data[name])
        if values.shape != (len(frame), len(targets)):
            raise ValueError(f'Inference {name} shape differs from rows/target order.')
        table[name] = values[:, index].astype(float)
    table['presence_probability'] = 1-table.pi
    table['sampled_mean'] = sampled_density_mean(table.pi, table.mu, table.sigma,
        num_samples=analysis.get('density_mc_samples', 1000),
        rng=np.random.default_rng(analysis.get('density_mc_seed', 20260719)))
    validate_parameters(table, frame)
    return table


def write_parameters(output: Path, table: pd.DataFrame, metadata: dict) -> None:
    """Publish the row-aligned parameter table with its provenance and checksum."""
    path = output/'cd8_parameters.csv'
    table.to_csv(path, index=False, float_format='%.17g')
    _write_json(output/'cd8_parameters.json', dict(metadata, version=1,
        table_sha256=digest_file(path), rows=len(table), row_ids_sha256=fingerprint(table.row_id.to_numpy())))


def plot_rows(count: int, cap: int | None, seed: int) -> np.ndarray:
    """Select the same display rows for every geometry and layout."""
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError('plot_max_samples must be null or a positive integer.')
    if cap is None or cap >= count:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, cap, replace=False))


def figure_path(output: Path, name: str, neighbors: int, seed: int, multiple: bool) -> Path:
    """Keep the reference filenames, nesting only when layouts would collide."""
    folder = output/'cd8_umap_parameters'
    if multiple:
        folder = folder/f'umap_k{neighbors}_seed{seed}'
    return folder/f'{name}.png'


def render_layout(coordinates: np.ndarray, parameters: pd.DataFrame, output: Path,
                  analysis: Mapping[str, Any], epsilon: float, cap: int | None, seed: int) -> None:
    """Render only the existing six-panel CD8 figure, sharing rows/color scales."""
    from dann.analyze import plot_latent_umap_ziln_per_target
    if coordinates.shape != (len(parameters), 2) or not np.isfinite(coordinates).all():
        raise ValueError('Nonfinite or misaligned UMAP coordinates.')
    selected = plot_rows(len(parameters), cap, seed)
    part = parameters.iloc[selected]
    # Factorize before capping so each slide keeps its color in every layout.
    slides, names = pd.factorize(parameters.slide, sort=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_latent_umap_ziln_per_target(coordinates[selected], part.Density_CD8.to_numpy(),
        part.pi.to_numpy(), part.mu.to_numpy(), part.sigma.to_numpy(), slides[selected],
        names.tolist(), 'Density_CD8', output, plot_settings(analysis), epsilon, part.sampled_mean.to_numpy())


def _legacy_parameters(source: Path, observations: pd.DataFrame, provenance: dict,
                       analysis_dir: Path | None) -> tuple[pd.DataFrame, dict]:
    """Read checksummed standard-analysis predictions from the identical checkpoint."""
    from dann.artifact_layout import analysis_file
    from dann.targets import checkpoint_contract
    checkpoint_path = Path(provenance['checkpoint'])
    checkpoint, checksum = load_checkpoint_snapshot(checkpoint_path)
    if checksum != provenance['checkpoint_sha256'] or int(checkpoint['epoch'])+1 != provenance['checkpoint_epoch']:
        raise ValueError('Geometry checkpoint identity has changed.')
    contract = checkpoint_contract(checkpoint)
    if fingerprint(checkpoint['data_identity']) != provenance['source_identity']:
        raise ValueError('Geometry source identity differs from checkpoint.')
    for split, checksum in provenance['saved_split_hashes'].items():
        if fingerprint(np.sort(checkpoint['split_indices'][split])) != checksum:
            raise ValueError(f'Geometry saved {split} split differs from checkpoint.')
    directory = analysis_dir or checkpoint_path.parent.parent/'analysis'
    reference = _read_json(directory/'umap/settings.json')
    if any(reference.get(key) != provenance[key] for key in ('checkpoint_sha256', 'checkpoint_epoch')):
        raise ValueError('Standard-analysis checkpoint differs from geometry checkpoint.')
    identity = reference.get('source_identity', {})
    if (identity.get('checkpoint') != provenance['checkpoint_sha256']
            or fingerprint(identity.get('identity')) != provenance['source_identity']
            or identity.get('source') != provenance['sources']['path']
            or reference.get('split') != 'validation'):
        raise ValueError('Standard-analysis source/split identity differs from geometry.')
    if _read_json(directory/'target_transform.json') != contract:
        raise ValueError('Standard-analysis target semantics differ from checkpoint.')
    path = analysis_file(directory, 'umap/latent_umap.csv')
    hashes = _read_json(directory/'latent/latent_run.json')['source_hashes']
    expected = hashes.get(str(path.resolve()))
    if not expected or not path.is_file() or digest_file(path) != expected:
        raise ValueError(f'Missing or inconsistent recorded prediction checksum: {path}')
    columns = ['row_id', 'split', 'batch', 'target_transform_json', 'Density_CD8', 'valid_Density_CD8',
               'pi_Density_CD8', 'mu_Density_CD8', 'sigma_Density_CD8', 'density_mean_Density_CD8']
    predictions = pd.read_csv(path, usecols=columns, float_precision='round_trip', dtype={'batch': str})
    rows = _row_ids(predictions, 'Standard-analysis predictions')
    if fingerprint(rows) != reference['row_ids_sha256'] or not predictions.split.eq('validation').all():
        raise ValueError('Standard-analysis prediction rows differ from recorded validation rows.')
    if predictions.target_transform_json.isna().any() or any(
            json.loads(value) != contract for value in predictions.target_transform_json.unique()):
        raise ValueError('Prediction table target semantics differ from checkpoint.')
    if not np.isin(observations.row_id, rows).all():
        raise ValueError('Standard-analysis predictions are missing eligible row IDs.')
    matched = predictions.set_index('row_id').loc[observations.row_id].reset_index()
    if (not matched.valid_Density_CD8.eq(True).all()
            or not np.allclose(matched.Density_CD8, observations.Density_CD8, rtol=1e-6, atol=1e-8)
            or not np.array_equal(matched.batch.astype(str), observations.batch.astype(str))):
        raise ValueError('Standard-analysis observed CD8/batch/eligibility differs from geometry.')
    table = observations[['row_id', 'slide', 'Density_CD8']].copy()
    for name in ('pi', 'mu', 'sigma'):
        table[name] = matched[f'{name}_Density_CD8'].to_numpy()
    table['presence_probability'] = 1-table.pi
    table['sampled_mean'] = matched.density_mean_Density_CD8.to_numpy()
    validate_parameters(table, observations)
    metadata = dict(checkpoint_sha256=provenance['checkpoint_sha256'], target_transform=contract,
        geometry_provenance_sha256=digest_file(source/'provenance.json'),
        logit_epsilon=checkpoint['config']['loss']['logit_epsilon'],
        parameter_source=str(path.resolve()), parameter_source_sha256=expected,
        sampling='Saved standard-analysis sampled means; reused without resampling.',
        validation_presence_max_abs_difference=float(np.max(np.abs(table.presence_probability-observations.cd8_presence))))
    return table, metadata


def render_experiment(source: Path, output: Path, analysis: Mapping[str, Any], *,
                      analysis_dir: Path | None = None) -> dict:
    """Copy numerical artifacts unchanged and render saved layouts without fitting."""
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Rendering destination must be separate from the source experiment.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Use a new output directory: {output}')
    display = latent_geometry_settings(analysis.get('latent_geometry'))
    settings = latent_geometry_settings(_read_json(source/'settings.json'))
    provenance = _read_json(source/'provenance.json')
    summary = _read_json(source/'summary.json')
    if not settings['layouts'] or not summary.get('completed_arms'):
        raise ValueError('Experiment has no completed saved layouts to render.')
    observations = pd.read_csv(source/'validation_observations.csv', float_precision='round_trip',
                               dtype={'slide': str, 'batch': str})
    rows = _row_ids(observations, 'Validation observations')
    if (len(rows) != provenance['eligible_counts']['validation']
            or fingerprint(rows) != provenance['eligible_row_hashes']['validation']):
        raise ValueError('Validation rows differ from geometry provenance.')
    for filename in ('preprocessing.npz', 'thresholds.json', 'train_observations.csv', 'eligibility.csv'):
        if not (source/filename).is_file():
            raise ValueError(f'Missing geometry input: {source/filename}')
    native = (source/'cd8_parameters.csv').is_file()
    if native:
        metadata = _read_json(source/'cd8_parameters.json')
        from dann.targets import target_contract
        contract = metadata.get('target_transform')
        if (not isinstance(contract, dict) or 'Density_CD8' not in contract.get('target_columns', [])
                or target_contract({'data': contract}) != contract):
            raise ValueError('Missing or invalid CD8 target semantics.')
        if (metadata.get('version') != 1 or metadata.get('table_sha256') != digest_file(source/'cd8_parameters.csv')
                or metadata.get('checkpoint_sha256') != provenance['checkpoint_sha256']
                or metadata.get('geometry_provenance_sha256') != digest_file(source/'provenance.json')
                or ('target_transform' in provenance and contract != provenance['target_transform'])
                or metadata.get('logit_epsilon') != provenance.get('logit_epsilon', metadata.get('logit_epsilon'))
                or metadata.get('rows') != len(rows) or metadata.get('row_ids_sha256') != fingerprint(rows)):
            raise ValueError('CD8 parameter checksum, identity, or target semantics differ from provenance.')
        table = pd.read_csv(source/'cd8_parameters.csv', float_precision='round_trip', dtype={'slide': str})
        validate_parameters(table, observations)
    else:
        if (source/'cd8_parameters.json').exists():
            raise ValueError('Missing exported cd8_parameters.csv.')
        table, metadata = _legacy_parameters(source, observations, provenance, analysis_dir)
    epsilon = metadata.get('logit_epsilon')
    if not isinstance(epsilon, (int, float)) or not 0 < epsilon < .5:
        raise ValueError('Missing or invalid CD8 logit epsilon.')
    layouts = []
    for arm in summary['completed_arms']:
        if not isinstance(arm, str) or Path(arm).name != arm or arm in ('.', '..'):
            raise ValueError('Invalid geometry arm name.')
        with np.load(source/arm/'representation.npz', allow_pickle=False) as representation:
            if not np.array_equal(representation['row_ids'], rows) or len(representation['values']) != len(rows):
                raise ValueError(f'Representation row alignment differs: {arm}')
        for k in settings['n_neighbors']:
            for seed in settings['seeds']:
                folder = source/arm/f'umap_k{k}_seed{seed}'
                saved = _read_json(folder/'settings.json')
                if saved.get('requested_neighbors') != k or saved.get('random_state') != seed:
                    raise ValueError(f'Layout settings differ from experiment: {folder}')
                path = folder/'coordinates.csv'
                if not path.is_file():
                    raise ValueError(f'Missing saved coordinates: {path}')
                coordinates = pd.read_csv(path, float_precision='round_trip')
                if (not np.array_equal(_row_ids(coordinates, 'Coordinates'), rows)
                        or not np.isfinite(coordinates[['umap_1', 'umap_2']].to_numpy()).all()):
                    raise ValueError(f'Coordinate row alignment or values differ: {path}')
                layouts.append((arm, k, seed, path))
    # Validate every dependency before creating any destination artifacts.
    numerical = [p for p in source.rglob('*') if p.is_file() and p.suffix in ('.csv', '.npz', '.json')]
    checksums = {str(p.relative_to(source)): digest_file(p) for p in numerical}
    output.mkdir(parents=True, exist_ok=True)
    for path in numerical:
        destination = output/path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if digest_file(destination) != checksums[str(path.relative_to(source))]:
            raise ValueError(f'Numerical artifact changed while copying: {path}')
    if not native:
        write_parameters(output, table, metadata)
    figures = []
    multiple = len(settings['n_neighbors'])*len(settings['seeds']) > 1
    for arm, k, seed, path in layouts:
        print(f'Rendering CD8: {arm}, neighbors={k}, seed={seed}', flush=True)
        coordinates = pd.read_csv(output/path.relative_to(source), float_precision='round_trip')
        destination = figure_path(output, arm, k, seed, multiple)
        render_layout(coordinates[['umap_1', 'umap_2']].to_numpy(), table, destination,
                      analysis, epsilon, display['plot_max_samples'], settings['seeds'][0])
        figures.append(str(destination.relative_to(output)))
    report = dict(source_experiment=str(source), checkpoint_sha256=provenance['checkpoint_sha256'],
        parameter_provenance=metadata, copied_artifact_sha256=checksums, figures=figures,
        eligible_rows=len(rows), displayed_rows=len(plot_rows(len(rows), display['plot_max_samples'], settings['seeds'][0])),
        plot_max_samples=display['plot_max_samples'], plot_settings=plot_settings(analysis),
        inference_rerun=False, umap_rerun=False)
    # A subsequent render also preserves the previous rendering provenance.
    report_path = output/'render_provenance.json'
    generation = 1
    while report_path.exists():
        generation += 1
        report_path = output/f'render_provenance_{generation}.json'
    _write_json(report_path, report)
    return report
