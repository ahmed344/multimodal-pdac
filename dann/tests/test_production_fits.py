"""Production orchestration gates and failure isolation without real training."""
import json
from pathlib import Path

import pytest

from dann.verification.production_fits import assess, orchestrate, DOWNSTREAM


@pytest.mark.parametrize('metrics,passed', [
    ((.9989, 1.9, .01), True),
    ((.9991, 1.9, .01), False),
    ((.998, 2., .01), False),
    ((.998, 1.9, 0.), False),
    ((.998, 1.9, float('nan')), False),
])
def test_validation_gate(metrics, passed):
    result = assess(dict(zip(('cd8_loss', 'biology_loss', 'mean_r2_cd8'), metrics)),
                    dict(cd8_loss=1., biology_loss=2.))
    assert result['passed'] is passed


@pytest.mark.parametrize('failure', ['train_A', 'validation_A', 'criteria_A', 'analyze_A', None])
def test_sequential_routing_and_preserved_failures(tmp_path, failure):
    runs = []
    calls = []
    for name in ('A', 'B'):
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'assessment.json').write_text(json.dumps({
            'assessment': {'passed': failure != 'criteria_' + name}}))
        runs.append(dict(name=name, config=str(directory / 'config.yaml'), directory=str(directory)))

    def runner(run, stage, command):
        calls.append((run['name'], stage))
        assert command[-1] == run['config'] or stage == 'validation'
        return failure != stage + '_' + run['name']

    results = orchestrate(dict(runs=runs), tmp_path, runner, lambda _: {'verified': True})
    assert calls[:2] == [('A', 'train'), ('B', 'train')]
    assert results['B']['downstream'] == 'completed'
    assert results['B']['verification']['verified']
    if failure in ('train_A', 'validation_A', 'criteria_A'):
        assert not any(name == 'A' and stage in [m.split('.')[-1] for m in DOWNSTREAM]
                       for name, stage in calls)
    if failure == 'analyze_A':
        assert results['A']['downstream'] == 'failed_dann.analyze'
        assert ('A', 'spatial') not in calls
    assert json.loads((tmp_path / 'results.json').read_text()) == results


def test_baseline_fits_only_valid_training_labels():
    from types import SimpleNamespace
    import numpy as np
    from dann.config import load_config
    from dann.verification.production_fits import fit_baseline

    config = load_config(Path('dann/config.yaml'))
    targets = np.tile(np.array([0., .1, .2, .4, 0., .3, .5, .7, np.nan], dtype=np.float32)[:, None], (1, 4))
    valid = np.ones_like(targets, dtype=bool)
    valid[3, 0] = False
    bundle = SimpleNamespace(metadata=SimpleNamespace(targets=targets, target_valid_mask=valid),
                             split_indices=dict(train=np.arange(4), validation=np.arange(4, 8), test=np.array([8])))
    first = fit_baseline(bundle, config)
    np.testing.assert_allclose(first['parameters']['pi_logits'][0], np.log(.5), atol=1e-6)
    bundle.metadata.targets[4:8] *= .5
    second = fit_baseline(bundle, config)
    assert first['parameters'] == second['parameters']
    assert first['validation'] != second['validation']
    assert first['fit_split'] == 'train' and first['score_split'] == 'validation'
