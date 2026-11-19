"""CD8-only geometry output, frozen layouts, and verified legacy prediction reuse."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from dann import geometry_render as r
from dann.analysis_execution import digest_file, fingerprint
from dann.config import latent_geometry_settings
from dann.targets import target_contract


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def parameter_fixture():
    frame = pd.DataFrame(dict(row_id=np.arange(8), slide=['001', '002']*4,
        Density_CD8=np.array([0., .1, .2, .3, .4, 0., .8, 1.])))
    values = dict(row_ids=frame.row_id.to_numpy(), pi=np.tile([.9, .25], (8, 1)),
                  mu=np.tile([12., -2.], (8, 1)), sigma=np.tile([3., .5], (8, 1)))
    values['pi'][0, 1] = 1.
    targets = ['Density_Tumor', 'Density_CD8']
    return frame, values, targets


def test_cd8_target_order_sampling_and_row_alignment():
    frame, values, targets = parameter_fixture()
    config = dict(density_mc_samples=64, density_mc_seed=53)
    a = r.cd8_parameter_table(values, frame, targets, config)
    b = r.cd8_parameter_table(values, frame, targets, config)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(a.mu, -2.)
    np.testing.assert_array_equal(a.presence_probability, 1-values['pi'][:, 1])
    assert a.sampled_mean[0] == 0
    assert not np.array_equal(a.sampled_mean, r.cd8_parameter_table(values, frame, targets,
        dict(config, density_mc_seed=54)).sampled_mean)
    with pytest.raises(ValueError, match='row alignment'):
        r.cd8_parameter_table(values, frame.iloc[::-1], targets, config)
    with pytest.raises(ValueError, match='unique targets'):
        r.cd8_parameter_table(values, frame, ['Density_CD8', 'Density_CD8'], config)


def test_six_panel_mapping_zero_styling_and_shared_limits(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.colors import to_rgba
    frame, values, targets = parameter_fixture()
    table = r.cd8_parameter_table(values, frame, targets, dict(density_mc_samples=16))
    figures = []
    def capture(figure, path, **kwargs):
        assert kwargs['dpi'] == 300
        figures.append(figure)
    monkeypatch.setattr(Figure, 'savefig', capture)
    config = dict(point_size=.2, density_cmap='viridis')
    coords = np.arange(16).reshape(8, 2)
    r.render_layout(coords, table, tmp_path/'first.png', config, 1e-5, None, 5)
    r.render_layout(coords*3, table, tmp_path/'second.png', config, 1e-5, None, 5)
    axes = figures[0].axes[:6]
    assert [axis.get_title() for axis in axes] == [
        'Latent UMAP by Logit_Density_CD8', 'logit(sampled mean density): Density_CD8',
        'Latent UMAP by batch', 'Latent UMAP by 1-pi_Density_CD8',
        'Latent UMAP by mu_Density_CD8', 'Latent UMAP by sigma_Density_CD8']
    for axis, other in zip(axes, figures[1].axes[:6]):
        assert sum(len(c.get_offsets()) for c in axis.collections) == len(table)
        for collection, same in zip(axis.collections, other.collections):
            assert collection.get_clim() == same.get_clim()
            np.testing.assert_array_equal(collection.get_array(), same.get_array())
            assert collection.get_sizes()[0] == pytest.approx(.2*.35)
    np.testing.assert_allclose(axes[0].collections[0].get_facecolors()[0], to_rgba('#d0d0d0'))
    np.testing.assert_array_equal(axes[0].collections[0].get_offsets(), coords[table.Density_CD8 == 0])
    np.testing.assert_array_equal(axes[1].collections[0].get_offsets(), coords[table.sampled_mean == 0])
    for axis, column in zip(axes[3:], ['presence_probability', 'mu', 'sigma']):
        np.testing.assert_array_equal(axis.collections[0].get_array(), table[column])
        assert axis.collections[0].cmap.name == 'viridis'
    np.testing.assert_array_equal(axes[2].collections[0].get_offsets(), coords[table.slide == '001'])
    for figure in figures:
        plt.close(figure)


def saved_experiment(tmp_path, *, native=True, multiple=False):
    source = tmp_path/'experiment'
    source.mkdir()
    frame, values, targets = parameter_fixture()
    table = r.cd8_parameter_table(values, frame, targets, dict(density_mc_samples=16))
    frame['batch'] = frame.slide
    frame['cd8_presence'] = table.presence_probability
    settings = latent_geometry_settings(dict(geometries=['raw_euclidean'], guided_geometry='raw_euclidean',
        cd8_weights=[0.], n_neighbors=[3, 5] if multiple else [3], seeds=[1, 2] if multiple else [1],
        plot_max_samples=2))  # Historical caps must not silently limit the new default.
    contract = target_contract({'data': dict(target_columns=targets)})
    identity = dict(rows=12, row_digest='source rows')
    checkpoint = dict(epoch=4, target_columns=targets, target_transform=contract,
        config=dict(data=dict(target_columns=targets), loss=dict(logit_epsilon=1e-5)),
        data_identity=identity, split_indices=dict(validation=np.arange(8), train=np.arange(8, 12)))
    checkpoint_path = tmp_path/'best.pt'
    torch.save(checkpoint, checkpoint_path)
    provenance = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=digest_file(checkpoint_path),
        checkpoint_epoch=5, source_identity=fingerprint(identity), sources=dict(path=['source.h5ad', 10, 20, 30]),
        saved_split_hashes={k: fingerprint(v) for k, v in checkpoint['split_indices'].items()},
        eligible_counts=dict(validation=len(frame)), eligible_row_hashes=dict(validation=fingerprint(frame.row_id.to_numpy())),
        target_transform=contract, logit_epsilon=1e-5)
    write_json(source/'settings.json', settings)
    write_json(source/'provenance.json', provenance)
    write_json(source/'summary.json', dict(completed_arms=['raw_euclidean']))
    write_json(source/'thresholds.json', dict(thresholds=[.1]))
    np.savez(source/'preprocessing.npz', mean=np.zeros(2))
    for name in ('validation_observations.csv', 'train_observations.csv', 'eligibility.csv'):
        frame.to_csv(source/name, index=False, float_format='%.17g')
    arm = source/'raw_euclidean'
    arm.mkdir()
    np.savez(arm/'representation.npz', row_ids=frame.row_id.to_numpy(), values=np.zeros((8, 2)))
    for k in settings['n_neighbors']:
        for seed in settings['seeds']:
            folder = arm/f'umap_k{k}_seed{seed}'
            write_json(folder/'settings.json', dict(requested_neighbors=k, random_state=seed,
                init='random', initialization_exception=dict(reason='saved recovery')))
            pd.DataFrame(dict(row_id=frame.row_id, umap_1=np.arange(8), umap_2=np.arange(8)*2)).to_csv(folder/'coordinates.csv', index=False)
            (folder/'layout.html').write_text('historical viewer')
            (folder/'layout.png').write_bytes(b'historical figure')
    metadata = dict(checkpoint_sha256=provenance['checkpoint_sha256'], target_transform=contract,
        logit_epsilon=1e-5, geometry_provenance_sha256=digest_file(source/'provenance.json'))
    if native:
        r.write_parameters(source, table, metadata)
    analysis = tmp_path/'analysis'
    reference = dict(checkpoint_sha256=provenance['checkpoint_sha256'], checkpoint_epoch=5,
        split='validation', row_ids_sha256=fingerprint(frame.row_id.to_numpy()[::-1]),
        source_identity=dict(checkpoint=provenance['checkpoint_sha256'], identity=identity,
                             source=provenance['sources']['path']))
    write_json(analysis/'umap/settings.json', reference)
    write_json(analysis/'target_transform.json', contract)
    predictions = frame[['row_id', 'batch', 'Density_CD8']].copy()
    predictions['split'] = 'validation'
    predictions['valid_Density_CD8'] = True
    predictions['target_transform_json'] = json.dumps(contract)
    for name in ('pi', 'mu', 'sigma'):
        predictions[f'{name}_Density_CD8'] = table[name]
    predictions['density_mean_Density_CD8'] = table.sampled_mean
    path = analysis/'umap/latent_umap.csv'
    predictions.iloc[::-1].to_csv(path, index=False, float_format='%.17g')
    write_json(analysis/'latent/latent_run.json', dict(source_hashes={str(path): digest_file(path)}))
    return source, analysis, table


@pytest.mark.parametrize('native,multiple,cap', [(True, False, None), (True, True, 3), (False, False, None)])
def test_render_only_never_infers_fits_or_samples_and_copies_bytes(tmp_path, monkeypatch, native, multiple, cap):
    from PIL import Image
    import umap
    from dann import latent_geometry, analyze, ziln
    source, analysis_dir, expected = saved_experiment(tmp_path, native=native, multiple=multiple)
    def forbidden(*args, **kwargs):
        pytest.fail('Rendering-only must never infer, sample, or fit geometry/UMAP.')
    monkeypatch.setattr(latent_geometry, 'extract_checkpoint', forbidden)
    monkeypatch.setattr(latent_geometry, 'load_checkpoint_snapshot', forbidden)
    monkeypatch.setattr(analyze, 'extract_latent_predictions', forbidden)
    monkeypatch.setattr(analyze, 'load_model', forbidden)
    monkeypatch.setattr(umap, 'UMAP', forbidden)
    monkeypatch.setattr(ziln, 'sampled_density_mean', forbidden)
    calls = []
    def render(coordinates, table, path, analysis, epsilon, limit, seed):
        pd.testing.assert_frame_equal(table, expected, check_dtype=False)
        assert len(coordinates) == 8 and limit == cap
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (2, 2)).save(path)
        calls.append(path)
    monkeypatch.setattr(r, 'render_layout', render)
    monkeypatch.setattr('dann.config.load_config', lambda path: dict(analysis=dict(latent_geometry=dict(plot_max_samples=cap))))
    output = tmp_path/'rendered'
    before = {str(p.relative_to(source)): digest_file(p) for p in source.rglob('*') if p.is_file()}
    assert latent_geometry.main(['--render-only', str(source), '--output', str(output),
                                '--analysis-dir', str(analysis_dir)]) == 0
    for name, checksum in before.items():
        assert digest_file(source/name) == checksum
        if (source/name).suffix in ('.json', '.csv', '.npz'):
            assert digest_file(output/name) == checksum
    expected_names = ([f'cd8_umap_parameters/umap_k{k}_seed{s}/raw_euclidean.png' for k in (3, 5) for s in (1, 2)]
                      if multiple else ['cd8_umap_parameters/raw_euclidean.png'])
    assert sorted(str(p.relative_to(output)) for p in calls) == sorted(expected_names)
    assert sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.suffix not in ('.json', '.npz', '.csv') and p.is_file()) == sorted(expected_names)
    report = json.loads((output/'render_provenance.json').read_text())
    assert report['displayed_rows'] == (cap or 8)
    assert report['inference_rerun'] is report['umap_rerun'] is False
    # Exported parameters are self-contained, including after migration from a legacy run.
    monkeypatch.setattr(r, 'load_checkpoint_snapshot', forbidden)
    r.render_experiment(output, tmp_path/'again', dict(latent_geometry=dict(plot_max_samples=cap)))


@pytest.mark.parametrize('problem,match', [
    ('checksum', 'checksum'), ('checkpoint', 'checkpoint'), ('semantics', 'semantics'),
    ('missing_rows', 'missing eligible'), ('duplicate_rows', 'unique'), ('density', 'observed CD8'),
    ('source', 'source/split'), ('coordinates', 'row alignment'), ('missing_layout', 'Missing saved coordinates')])
def test_legacy_render_rejects_inconsistent_inputs_before_writing(tmp_path, problem, match):
    source, directory, _ = saved_experiment(tmp_path, native=False)
    prediction_path = directory/'umap/latent_umap.csv'
    predictions = pd.read_csv(prediction_path, float_precision='round_trip', dtype={'batch': str})
    reference_path = directory/'umap/settings.json'
    reference = json.loads(reference_path.read_text())
    if problem == 'checksum':
        prediction_path.write_text(prediction_path.read_text()+'\n')
    elif problem == 'checkpoint':
        reference['checkpoint_sha256'] = 'wrong'
    elif problem == 'semantics':
        write_json(directory/'target_transform.json', {})
    elif problem == 'source':
        reference['source_identity']['source'] = ['changed']
    elif problem in ('missing_rows', 'duplicate_rows', 'density'):
        if problem == 'missing_rows':
            predictions = predictions.iloc[1:]
        elif problem == 'duplicate_rows':
            predictions.loc[0, 'row_id'] = predictions.row_id.iloc[1]
        else:
            predictions.loc[0, 'Density_CD8'] = .44
        predictions.to_csv(prediction_path, index=False, float_format='%.17g')
        reference['row_ids_sha256'] = fingerprint(predictions.row_id.to_numpy())
        write_json(directory/'latent/latent_run.json', dict(source_hashes={str(prediction_path): digest_file(prediction_path)}))
    else:
        path = next(source.glob('*/umap_*/coordinates.csv'))
        if problem == 'missing_layout':
            path.unlink()
        else:
            coordinates = pd.read_csv(path)
            coordinates.iloc[::-1].to_csv(path, index=False)
    write_json(reference_path, reference)
    output = tmp_path/'invalid'
    with pytest.raises(ValueError, match=match):
        r.render_experiment(source, output, {}, analysis_dir=directory)
    assert not output.exists()


@pytest.mark.parametrize('problem,match', [
    ('checksum', 'checksum'), ('checkpoint', 'identity'), ('semantics', 'semantics'),
    ('rows', 'row alignment'), ('sigma', 'sigma'), ('presence', 'presence'), ('missing', 'Missing exported')])
def test_native_parameters_reject_inconsistent_inputs(tmp_path, problem, match):
    source, _, _ = saved_experiment(tmp_path)
    path = source/'cd8_parameters.csv'
    metadata_path = source/'cd8_parameters.json'
    metadata = json.loads(metadata_path.read_text())
    table = pd.read_csv(path, float_precision='round_trip', dtype={'slide': str})
    if problem == 'checksum':
        path.write_text(path.read_text()+'\n')
    elif problem == 'checkpoint':
        metadata['checkpoint_sha256'] = 'other'
    elif problem == 'semantics':
        metadata['target_transform']['cd8_normalization']['enabled'] = True
    elif problem == 'missing':
        path.unlink()
    else:
        if problem == 'rows':
            table = table.iloc[::-1]
        elif problem == 'sigma':
            table.loc[0, 'sigma'] = 0
        else:
            table.loc[0, 'presence_probability'] = .5
        table.to_csv(path, index=False, float_format='%.17g')
        metadata['table_sha256'] = digest_file(path)
    write_json(metadata_path, metadata)
    output = tmp_path/'invalid'
    with pytest.raises(ValueError, match=match):
        r.render_experiment(source, output, {})
    assert not output.exists()


@pytest.mark.parametrize('cap', [0, -1, True, 1.2])
def test_plot_caps_reject_invalid_values(cap):
    with pytest.raises(ValueError, match='plot_max_samples'):
        latent_geometry_settings(dict(plot_max_samples=cap))


def test_plot_caps_are_deterministic_and_optional():
    assert latent_geometry_settings()['plot_max_samples'] is None
    np.testing.assert_array_equal(r.plot_rows(8, None, 1), np.arange(8))
    np.testing.assert_array_equal(r.plot_rows(8, 3, 1), r.plot_rows(8, 3, 1))
    assert len(r.plot_rows(8, 3, 1)) == 3


def test_render_refuses_existing_destination_and_missing_output(tmp_path):
    from dann.latent_geometry import main
    source, _, _ = saved_experiment(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        r.render_experiment(source, source/'nested', {})
    destination = tmp_path/'occupied'
    destination.mkdir()
    (destination/'keep.txt').write_text('keep')
    with pytest.raises(ValueError, match='new output'):
        r.render_experiment(source, destination, {})
    with pytest.raises(SystemExit):
        main(['--render-only', str(source)])
