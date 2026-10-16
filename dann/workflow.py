"""Sequential, CLI-triggered DANN post-training work on the best checkpoint."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
import torch

from dann.analyze import run_analysis
from dann.checkpoints import checkpoint_config
from dann.config import (
    density_sampling_settings, inference_input, latent_variance_settings,
    post_training_settings, spatial_heatmap_settings,
)
from dann.latent_variance import run_latent_variance
from dann.spatial import (
    _file_identity, run_inference, validate_input_schema, validate_latent_output,
    validate_output,
)
from dann.spatial_heatmaps import run_spatial_heatmaps
from dann.targets import checkpoint_contract


class PostTrainingError(RuntimeError):
    """A named downstream stage failed; completed artifacts remain available."""

    def __init__(self, stage: str, error: Exception):
        self.stage = stage
        super().__init__(f"Post-training stage '{stage}' failed: {error}. "
                         "Completed training and downstream artifacts are preserved.")


def _workflow_config(config: Mapping[str, Any], checkpoint: Mapping[str, Any],
                     checkpoint_path: Path) -> dict[str, Any]:
    """Keep saved training contracts and current downstream controls and paths."""
    effective = checkpoint_config(checkpoint)
    contract = checkpoint_contract(checkpoint)
    effective['data'].update({key: contract[key] for key in ('target_columns', 'cd8_normalization')})
    effective['data']['context_path'] = config['data'].get('context_path')
    for section in ('analysis', 'performance', 'post_training'):
        if section in config:
            effective[section] = copy.deepcopy(config[section])
    effective['training']['device'] = config['training']['device']

    # Rebase resolved destinations when resume returns a different run's best.pt.
    # Without results, historical configurations put analysis beneath output_dir.
    requested_root = Path(config.get('results', {}).get('run_dir')
                          or config['training']['output_dir']).resolve()
    actual_root = (checkpoint_path.parent.parent if 'results' in effective
                   else checkpoint_path.parent).resolve()

    def relocate(value: str | Path) -> str:
        path = Path(value).resolve()
        try:
            return str(actual_root / path.relative_to(requested_root))
        except ValueError:
            # Explicit destinations outside the run tree remain explicit.
            return str(path)

    effective.pop('results', None)  # Paths are resolved here, not from saved run names.
    effective['training']['output_dir'] = str(checkpoint_path.parent)
    analysis = effective['analysis']
    analysis['checkpoint'] = str(checkpoint_path)
    analysis['output_dir'] = relocate(config['analysis']['output_dir'])
    spatial_defaults = {
        'output': str(Path(config['analysis']['output_dir']) / 'spatial/spatial_predictions.parquet'),
        'latents': str(Path(config['analysis']['output_dir']) / 'spatial/spatial_latent.parquet'),
    }
    effective['spatial'] = {key: relocate(config.get('spatial', {}).get(key, default))
                            for key, default in spatial_defaults.items()}
    variance = latent_variance_settings(config.get('latent_variance'))
    variance.update(input=config['data'].get('context_path'),
                    latents=effective['spatial']['latents'],
                    output_dir=relocate(config.get('latent_variance', {}).get(
                        'output_dir', Path(config['analysis']['output_dir']) / 'latent_variance')))
    effective['latent_variance'] = variance
    return effective


def _validate_exports(config: Mapping[str, Any], checkpoint: Mapping[str, Any],
                      checkpoint_path: Path, input_path: Path, *,
                      predictions: bool, latents: bool) -> None:
    """Validate complete exports and their association with this checkpoint/input."""
    prediction_path = Path(config['spatial']['output'])
    latent_path = Path(config['spatial']['latents'])
    provenance_path = prediction_path.with_suffix('.provenance.json')
    try:
        paths = [provenance_path]
        if predictions:
            paths.append(prediction_path)
        if latents:
            paths.append(latent_path)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Required inference artifact is missing: {path}")
        provenance = json.loads(provenance_path.read_text())
        identities = [('checkpoint_identity', checkpoint_path), ('source_identity', input_path)]
        if predictions:
            identities.append(('prediction_identity', prediction_path))
        if latents:
            identities.append(('latent_identity', latent_path))
        for key, path in identities:
            if provenance.get(key) != _file_identity(path):
                raise ValueError(f"Inference provenance {key} is missing or incompatible with {path}")
        data = config['data']
        rows = validate_input_schema(
            input_path, Path(data['path']), data['matrix_key'], data['target_columns'],
            config['model']['num_peaks'], (checkpoint.get('data_identity') or {}).get('feature_order'),
        )
        if provenance.get('selected_rows') != rows or provenance.get('context_rows') != rows:
            raise ValueError('Inference provenance does not describe the complete tissue input')
        if predictions:
            count, seed = density_sampling_settings(config['analysis'])
            if (provenance.get('density_mc_samples'), provenance.get('density_mc_seed')) != (count, seed):
                raise ValueError('Inference provenance disagrees with current density sampling settings')
            metadata = pq.read_schema(prediction_path).metadata or {}
            if json.loads(metadata.get(b'dann_target_transform', b'null')) != checkpoint_contract(checkpoint):
                raise ValueError('Prediction target transformation differs from the checkpoint')
            validate_output(prediction_path, input_path, data['target_columns'], rows)
        if latents:
            validate_latent_output(latent_path, input_path, config['model']['latent_dim'], rows)
    except Exception as error:
        raise ValueError(
            f"Cannot reuse inference exports: {error}. Regenerate them for this checkpoint "
            "and tissue input with post_training.run_inference: true and "
            "post_training.overwrite_inference: true, or supply compatible exports."
        ) from error


def run_post_training_workflow(config: Mapping[str, Any], checkpoint_path: Path) -> dict[str, Path]:
    """Run enabled stages once in order, returning successful artifact locations.

    Current runtime settings control the stages; the supplied best checkpoint
    controls training data, preprocessing, model, targets, and saved splits.
    No stage is implicitly enabled to satisfy another stage's dependencies.
    """
    outputs: dict[str, Path] = {}
    stage = 'configuration'
    try:
        switches = post_training_settings(config.get('post_training'))
        if not switches['enabled'] or not any(switches[key] for key in (
                'run_analysis', 'run_inference', 'run_spatial_heatmaps', 'run_latent_variance')):
            return outputs
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        effective = _workflow_config(config, checkpoint, checkpoint_path)
        # Validators need metadata only; release potentially large training state.
        checkpoint.pop('model_state', None)
        checkpoint.pop('optimizer_state', None)
        analysis = effective['analysis']

        def record(name: str, path: Path) -> None:
            outputs[name] = Path(path)
            print(f"Post-training {name}: {path}", flush=True)

        if switches['run_analysis']:
            stage = 'analysis'
            record(stage, run_analysis(effective, checkpoint_path,
                                      output_override=Path(analysis['output_dir'])))
        if switches['run_inference']:
            stage = 'inference'
            tissue = inference_input(effective)
            count, seed = density_sampling_settings(analysis)
            record(stage, run_inference(
                input_path=tissue, checkpoint_path=checkpoint_path,
                output_path=Path(effective['spatial']['output']),
                latent_output_path=Path(effective['spatial']['latents']),
                device_override=effective['training']['device'],
                batch_size_override=analysis['batch_size'], num_workers_override=analysis['num_workers'],
                density_mc_samples=count, density_mc_seed=seed,
                overwrite=switches['overwrite_inference'],
                execution_overrides={'spatial_execution': analysis.get('spatial_execution', {}),
                                     'performance': effective.get('performance', {})},
            ))
            record('latents', Path(effective['spatial']['latents']))
        for stage in ('spatial_heatmaps', 'latent_variance'):
            if not switches[f'run_{stage}']:
                continue
            tissue = inference_input(effective)
            if not switches['run_inference']:
                _validate_exports(effective, checkpoint, checkpoint_path, tissue,
                                  predictions=stage == 'spatial_heatmaps', latents=stage == 'latent_variance')
            if stage == 'spatial_heatmaps':
                record(stage, run_spatial_heatmaps(
                    input_path=tissue, predictions_path=Path(effective['spatial']['output']),
                    output_dir=Path(analysis['output_dir']) / 'spatial/heatmaps',
                    densities=effective['data']['target_columns'],
                    logit_epsilon=effective['loss']['logit_epsilon'], dpi=analysis['figure_dpi'],
                    batch_column=effective['data']['batch_column'],
                    **spatial_heatmap_settings(analysis.get('spatial_heatmaps')),
                ))
            else:
                record(stage, run_latent_variance(effective))
        return outputs
    except Exception as error:
        raise PostTrainingError(stage, error) from error
