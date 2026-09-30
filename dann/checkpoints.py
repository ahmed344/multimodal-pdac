"""Versioned architecture contracts and deterministic legacy compatibility."""
from typing import Any, Mapping

from dann.config import component_settings, execution_settings


def architecture_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe selected architecture and numerical settings, independent of run paths."""
    settings = component_settings(config['model'])
    selected = {'spectral_encoder': settings['spectral_encoder']}
    for name, group in [('aggregation', settings['aggregation']), *settings['heads'].items()]:
        selected[name] = {'type': group['type'], group['type']: group[group['type']]}
    return {'version': 2, 'components': selected,
            'numerics': {key: config['model'][key] for key in
                         ('num_peaks', 'latent_dim', 'activation', 'dropout', 'use_layer_norm', 'sigma_min')},
            'spatial': execution_settings(config)}


def legacy_parameter_conversion(state: Mapping[str, Any]) -> dict[str, Any]:
    """Identity conversion: legacy parameter keys and registration order are retained."""
    return dict(state)


def validate_checkpoint(checkpoint: Mapping[str, Any], config: Mapping[str, Any],
                        model=None, optimizer: bool = False) -> None:
    """Reject architecture or optimizer registration changes before loading state."""
    saved = checkpoint.get('architecture', architecture_contract(checkpoint['config']))
    if saved != architecture_contract(config):
        raise ValueError('Checkpoint components or numerical settings are incompatible.')
    if optimizer and model is not None:
        names = list(dict(model.named_parameters()))
        saved_names = checkpoint.get('optimizer_parameter_names', list(checkpoint['model_state']))
        if saved_names != names:
            raise ValueError('Cannot resume: optimizer parameter ordering is incompatible.')
        ids = [i for group in checkpoint['optimizer_state']['param_groups'] for i in group['params']]
        if len(ids) != len(names):
            raise ValueError('Cannot resume: optimizer parameter count is incompatible.')
        for parameter, index in zip(model.parameters(), ids):
            state = checkpoint['optimizer_state']['state'].get(index, {})
            if any(state[k].shape != parameter.shape for k in ('exp_avg', 'exp_avg_sq') if k in state):
                raise ValueError('Cannot resume: optimizer parameter shapes are incompatible.')


def training_data_contract(config: Mapping[str, Any], identity=None) -> dict[str, Any]:
    """Record the labeled-only source and allowed held-out spectral context."""
    from pathlib import Path
    from dann.tiles import row_identity
    return {
        'version': 1,
        'spectra_source': 'data.path',
        'context_source': 'data.path',
        'supervision': 'selected_split_core_labels_only',
        'cross_split_spectral_context': 'all_occupied_labeled_file_rows',
        'labeled_path': str(Path(config['data']['path']).resolve()),
        'labeled_identity': identity if identity is not None else row_identity(Path(config['data']['path']), config['data']),
    }


def validate_training_data_contract(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Reject unknown or incompatible supervised provenance before loading weights."""
    saved = checkpoint.get('training_data_contract')
    if saved is None:
        raise ValueError('Missing training_data_contract: legacy provenance is unknown; inference only.')
    expected = training_data_contract(config)
    # A relocated copy with identical ordered rows/features remains compatible.
    if {k: v for k, v in saved.items() if k != 'labeled_path'} != {k: v for k, v in expected.items() if k != 'labeled_path'}:
        raise ValueError('Incompatible training_data_contract for supervised training/analysis.')
    if checkpoint.get('data_identity', saved['labeled_identity']) != saved['labeled_identity']:
        raise ValueError('Checkpoint data identity and training_data_contract disagree.')
