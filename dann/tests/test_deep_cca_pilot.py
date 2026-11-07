"""Preregistered CCA selection and preservation decisions."""
import copy

import pytest

from dann.verification.deep_cca_pilot import PROBES, primary_score, preservation, rank


def record(name='control', error=1., cd8=1., weight=.05, dimension=16):
    probes = {n: {metric: error} for n, metric in PROBES}
    cd8_metrics = dict(loss=cd8, presence=dict(brier=.1),
        exceedance={str(t): dict(brier=.1) for t in (.1, .2, .3)},
        positive_pit_ks=.02, intervals={'0.5': dict(coverage=.5), '0.9': dict(coverage=.9)})
    return dict(id=name, dimension=dimension, weight=weight, spectrum=dict(noncollapsed=True),
        probes=copy.deepcopy(probes), stromal_probes=probes,
        distributions=dict(cd8=cd8_metrics), stromal_neighbors=dict(excess=.1))


def test_ratios_floor_and_preservation():
    control = record(dimension=None)
    candidate = record('candidate', .98)
    assert primary_score(candidate, control) == pytest.approx(.98)
    assert preservation(candidate, control)['eligible']
    control['stromal_probes']['tumor']['mse'] = 0.
    candidate['stromal_probes']['tumor']['mse'] = 2e-8
    assert primary_score(candidate, control) == pytest.approx((2+6*.98)/7)
    candidate['stromal_neighbors']['excess'] = .111
    assert not preservation(candidate, control)['eligible']


def test_ties_use_likelihood_then_smaller_weight_dimension():
    control = record(dimension=None)
    a, b = record('a', .98, cd8=.99), record('b', .985, cd8=.98)
    assert rank([a, b], control)[0]['id'] == 'b'
    b['distributions']['cd8']['loss'] = .99
    b['weight'] = .015
    assert rank([a, b], control)[0]['id'] == 'b'
    a['weight'] = .015
    b['dimension'] = 32
    assert rank([a, b], control)[0]['id'] == 'a'
    a['spectrum']['noncollapsed'] = False
    assert [r['id'] for r in rank([a, b], control)] == ['b']
