"""Scientific comparison metrics, independent of a completed full-data fit."""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from dann.losses import ZILNLoss
from dann.verification.pooling_diagnostics import (
    checkpoint_summary, covariance_metrics, distribution_metrics, relative_change,
    variance_component_counts, weighted_auc,
)


def test_covariance_separates_shared_offset_from_centered_rank():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(100,4))
    before=covariance_metrics(x,{'slide':np.arange(100)%2})
    after=covariance_metrics(x+100,{'slide':np.arange(100)%2})
    np.testing.assert_allclose(before['eigenvalues'],after['eigenvalues'])
    assert before['participation_ratio']==pytest.approx(after['participation_ratio'])
    assert after['shared_offset_energy_fraction']>.999
    assert after['slide']['between_fraction']==pytest.approx(before['slide']['between_fraction'])
    change=relative_change(x,x+100)
    assert change['relative_centered_rms']<1e-12
    assert abs(change['mean_centered_cosine_change'])<1e-12
    assert relative_change(np.zeros_like(x),np.zeros_like(x))['mean_cosine_change']==0


def test_weighted_auc_ties_and_patient_multiplicity():
    y=np.array([0,1,1,0,1,0,0,1]);p=np.array([.1,.1,.2,.4,.7,.7,.8,.9]);w=np.array([1,2,0,4,3,1,0,2])
    assert weighted_auc(y,p,w)==pytest.approx(roc_auc_score(y,p,sample_weight=w))
    assert np.isnan(weighted_auc(y,p,np.where(y,1,0)))


def test_distribution_likelihood_matches_production_mask_and_positive_branch():
    rng=np.random.default_rng(7);n=80
    y=rng.uniform(.001,.999,(n,4));y[rng.random(y.shape)<.4]=0
    mask=rng.random(y.shape)>.1
    values={'pi_logits':rng.normal(size=y.shape),'mu':rng.normal(size=y.shape),'sigma':rng.uniform(.4,2,y.shape)}
    scores=distribution_metrics(values,y,mask,[.1,.2,.4],1e-5)
    loss=ZILNLoss([1.],1e-5,'mean',False)
    expected=loss(*(torch.tensor(values[k][:,:1]) for k in ('pi_logits','mu','sigma')),torch.tensor(y[:,:1]),torch.tensor(mask[:,:1]))
    assert scores['cd8']['loss']==pytest.approx(float(expected.total))
    assert scores['cd8']['positive_pit_histogram'].sum()==np.sum(mask[:,0]&(y[:,0]>0))
    assert scores['cd8']['n']==mask[:,0].sum()


def test_variance_component_counts_exact_threshold_and_constant_representation():
    assert variance_component_counts([1.,0.,9.])=={'pcs_90':1,'pcs_99':2}
    assert variance_component_counts(np.zeros(4))=={'pcs_90':0,'pcs_99':0}
    with pytest.raises(ValueError,match='finite nonnegative'):
        variance_component_counts([1.,np.nan])


def test_summary_distinguishes_selected_checkpoint_from_lowest_cd8_epoch():
    history=pd.DataFrame(dict(epoch=[3,1,2],validation_cd8_loss=[.8,.8,.9],
        validation_batch_loss=[3.,1.,2.],validation_biology_loss=[.4,.5,.6],
        validation_biology_weighted_loss=[4.,5.,6.],validation_he_mse=[.01,.02,.03]))
    summary=checkpoint_summary(history,2,[9.,1.,0.])
    assert summary['selected_epoch']==2 and summary['stopped_epoch']==3
    assert summary['cd8_loss']==.9
    assert summary['lowest_cd8_loss']==.8 and summary['lowest_cd8_loss_epoch']==1
    assert summary['ihc_loss']==.6 and summary['he_mse']==.03
    assert summary['discriminator_loss']==2.
    assert summary['latent_pcs_90']==1 and summary['latent_pcs_99']==2
