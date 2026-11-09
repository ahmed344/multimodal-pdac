"""Score/loss pairing and effective-coefficient activation in plotting copies."""
from pathlib import Path
import numpy as np
import pandas as pd
from dann.config import load_config
from dann.objective_curves import active_weighted_history, weighted_history, objective_layout


def test_cca_train_validation_activation_and_real_zero():
    c = load_config(Path('dann/config.yaml'))
    frame = pd.DataFrame(dict(epoch=[1,2,3,4],train_cca_weighted_loss=[0,0,-.2,0],
        validation_cca_weighted_loss=[0,-.1,-.2,0],train_cca_correlation=[.1,.2,.3,.4],
        train_cca_effective_weight=[0,.05,.1,0],validation_cca_effective_weight=[.1,.1,.1,.1],
        train_total_loss=[1,2,3,4]))
    original=frame.copy(deep=True)
    result = active_weighted_history(frame,c)
    np.testing.assert_allclose(result.train_cca_weighted_loss,[np.nan,0,-.2,np.nan],equal_nan=True)
    np.testing.assert_array_equal(result.validation_cca_weighted_loss,frame.validation_cca_weighted_loss)
    np.testing.assert_array_equal(result.train_cca_correlation,frame.train_cca_correlation)
    np.testing.assert_array_equal(result.train_total_loss,frame.train_total_loss)
    pd.testing.assert_frame_equal(frame,original)


def test_historical_schedule_reconstructed_and_disabled_components_hidden():
    c = load_config(Path('dann/config.yaml'))
    frame = pd.DataFrame(dict(epoch=[1,2,3],train_cca_loss=[0,0,-2],validation_cca_loss=[0,0,-2],
        train_cca_correlation=[.1,.2,.3],validation_cca_correlation=[.1,.2,.3]))
    result=active_weighted_history(weighted_history(frame,c),c)
    np.testing.assert_allclose(result.train_cca_weighted_loss,[np.nan,0,-.2],equal_nan=True)
    np.testing.assert_allclose(result.validation_cca_weighted_loss,[0,0,-.2])
    c['model']['heads']['he']['enabled']=False
    rows=objective_layout(result,c)[0]
    assert not any('_cca_' in s[0] for row in rows for p in row for s in p['series'])


def test_disabled_panels_cannot_break_vertical_pairs():
    c = load_config(Path('dann/config.yaml'))
    frame=pd.DataFrame([dict(epoch=1,train_hurdle_acc=.5,train_hurdle_weighted_loss=0,
        train_mean_r2=.1,train_positive_weighted_loss=-1,train_cca_correlation=.4,train_cca_weighted_loss=0,
        train_batch_accuracy=.5,train_batch_weighted_loss=1,
        train_hurdle_acc_cd8=.6,train_hurdle_weighted_loss_cd8=0,
        train_mean_r2_tumor=.4,train_positive_weighted_loss_tumor=-.3)])
    for he in (False,True):
        c['model']['heads']['he']['enabled']=he
        for disc in (False,True):
            c['model']['heads']['discriminator']['enabled']=disc
            overall,ihc,*_=objective_layout(frame,c)
            for rows,pairs in [(overall,[('hurdle_acc','hurdle_weighted_loss'),('mean_r2','positive_weighted_loss'),
                                        ('cca_correlation','cca_weighted_loss'),('batch_accuracy','batch_weighted_loss')]),
                               (ihc,[('hurdle_acc_cd8','hurdle_weighted_loss_cd8'),('mean_r2_tumor','positive_weighted_loss_tumor')])]:
                locations={}
                for r,row in enumerate(rows):
                    col=0
                    for p in row:
                        for key,*_ in p['series']: locations[key]=(r,col,p['span'])
                        col+=p['span']
                for score,loss in pairs:
                    score,loss='train_'+score,'train_'+loss
                    if score in locations and loss in locations:
                        assert locations[loss] == (locations[score][0]+1,*locations[score][1:])
