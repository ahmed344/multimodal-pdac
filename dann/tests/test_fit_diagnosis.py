"""Numerical and identity contracts of the training-only fit diagnosis."""
import numpy as np
import pandas as pd
import pytest
import torch

from dann.losses import ZILNLoss
from dann.verification.fit_diagnosis import (
    altered_batches, constant_parameters, metrics, select_rows,
)


def test_constant_mle_ignores_masked_labels_and_matches_known_logit_distribution():
    targets = torch.tensor([[0.], [0.], [torch.sigmoid(torch.tensor(-3.))],
                            [torch.sigmoid(torch.tensor(-1.))], [float('nan')]])
    mask = torch.tensor([[True], [True], [True], [True], [False]])
    parameters = constant_parameters(targets, mask, 1e-5, 1e-4)
    torch.testing.assert_close(parameters['pi_logits'], torch.zeros_like(targets))
    torch.testing.assert_close(parameters['mu'], torch.full_like(targets, -2.))
    torch.testing.assert_close(parameters['sigma'], torch.ones_like(targets))
    result = ZILNLoss([1.])(*parameters.values(), targets, mask)
    assert float(result.total) == pytest.approx(np.log(2)+.25)


def test_gate_requires_both_loss_improvements_and_positive_correlation():
    targets = torch.tensor([[0.], [.05], [.1], [.3]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    batch = {'targets': targets, 'target_valid_mask': mask}
    out = {'pi_logits': torch.tensor([[5.], [-5.], [-5.], [-5.]]),
           'mu': torch.logit(targets.clamp_min(1e-5)), 'sigma': torch.ones_like(targets)}
    score = metrics(out, batch, ZILNLoss([1.]))
    baseline = {'biology': score['biology']+.06, 'cd8': score['cd8']+.06}
    assert metrics(out, batch, ZILNLoss([1.]), baseline)['passed']
    baseline['cd8'] = score['cd8']+.04
    assert not metrics(out, batch, ZILNLoss([1.]), baseline)['passed']
    baseline['cd8'] += 1
    baseline['biology'] = score['biology']+.04
    assert not metrics(out, batch, ZILNLoss([1.]), baseline)['passed']
    baseline['biology'] += 100
    out['mu'] = -out['mu']
    assert not metrics(out, batch, ZILNLoss([1.]), baseline)['passed']


def test_seeded_selection_uses_only_training_rows_and_balances_each_tile():
    keys = pd.MultiIndex.from_tuples([(slide, x, y) for slide in ('a', 'b', 'c')
        for x in range(16) for y in range(8)], names=['batch', 'x', 'y'])
    targets = np.ones((len(keys), 4))*.1
    targets[::2, 0] = 0
    mask = np.ones_like(targets, dtype=bool)
    # Each tile retains at least eight entries of each class after exclusions.
    train = np.arange(len(keys))[np.arange(len(keys)) % 5 != 0]
    mask[::7, 0] = False
    rows = select_rows(keys, train, targets, mask, 123)
    np.testing.assert_array_equal(rows, select_rows(keys, train, targets, mask, 123))
    assert len(rows) == len(np.unique(rows)) == 64
    assert np.isin(rows, train).all()
    assert mask[rows, 0].all()
    frame = keys.to_frame(index=False).iloc[rows].copy()
    frame['tile'] = frame.x//8
    frame['zero'] = targets[rows, 0] == 0
    groups = frame.groupby(['batch', 'tile']).zero.agg(['sum', 'count'])
    assert frame.batch.nunique() == 2
    assert len(groups) == 4
    assert (groups['sum'] == 8).all()
    assert (groups['count'] == 16).all()


def test_shuffle_moves_whole_spectra_without_moving_targets_or_positions():
    batch = dict(peak_indices=torch.tensor([1, 2, 3, 4, 5, 6]),
                 intensities=torch.tensor([.1, .2, .3, .4, .5, .6]),
                 sample_indices=torch.tensor([0, 0, 2, 3, 3, 3]),
                 peak_counts=torch.tensor([2, 0, 1, 3]),
                 targets=torch.arange(4)[:, None], spatial_positions=torch.tensor([0, 2, 3, 4]))
    changed = altered_batches(batch, 3)
    shuffled = changed['shuffled_spectra']
    def spectra(value):
        return sorted(tuple(zip(value['peak_indices'][value['sample_indices'] == i].tolist(),
                                value['intensities'][value['sample_indices'] == i].tolist()))
                      for i in range(len(value['peak_counts'])))
    assert spectra(shuffled) == spectra(batch)
    assert torch.equal(shuffled['targets'], batch['targets'])
    assert torch.equal(shuffled['spatial_positions'], batch['spatial_positions'])
    assert not changed['zero_amplitudes']['intensities'].any()
    assert not changed['empty_spectra']['peak_counts'].any()
    assert len(changed['empty_spectra']['peak_indices']) == 0
