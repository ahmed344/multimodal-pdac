"""Independent references for the saved latent feature correlations."""
from pathlib import Path
import argparse
import json
import time
import h5py
from anndata.io import read_elem
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def audit(config):
    start=time.perf_counter();root=Path(config['analysis']['output_dir']);out=root/'latent'
    latent=pd.read_csv(root/'tables/latent_embeddings.csv',float_precision='round_trip')
    frame=pd.read_csv(root/'umap/tables/latent_umap.csv',float_precision='round_trip').set_index('row_id').loc[latent.row_id].reset_index()
    with h5py.File(config['data']['path']) as handle:lesion=np.asarray(read_elem(handle['obs/Lesion']),dtype=bool)[latent.row_id]
    ids=np.unique([0,1,127,len(latent.filter(regex='^latent_').columns)-1]);ids=ids[ids<len(latent.filter(regex='^latent_').columns)]
    pairs=np.load(out/'tables/feature_correlations.npy');counts=np.load(out/'tables/contributing_slides.npy')
    exported=pd.read_csv(out/'tables/ihc_associations.csv');per_slide=pd.read_csv(out/'tables/slide_ihc_associations.csv')
    checked=0
    for i in ids:
        for j in ids:
            references=[]
            for slide in latent.batch.unique():
                mask=lesion&(latent.batch.to_numpy()==slide)
                x,y=latent[f'latent_{i}'].to_numpy()[mask],latent[f'latent_{j}'].to_numpy()[mask]
                valid=np.isfinite(x)&np.isfinite(y);x,y=x[valid],y[valid]
                if len(x)>=20 and np.ptp(x)>0 and np.ptp(y)>0:references.append(spearmanr(x,y).statistic)
            np.testing.assert_allclose(pairs[i,j],np.mean(references) if references else np.nan,atol=1e-12,equal_nan=True)
            assert counts[i,j]==len(references);checked+=1
        for target in config['data']['target_columns']:
            references=[]
            for slide in latent.batch.unique():
                mask=lesion&(latent.batch.to_numpy()==slide)&frame[f'valid_{target}'].to_numpy(dtype=bool)
                x,y=latent[f'latent_{i}'].to_numpy()[mask],frame[target].to_numpy()[mask]
                valid=np.isfinite(x)&np.isfinite(y);x,y=x[valid],y[valid]
                expected=np.nan
                if len(x)>=20 and np.ptp(x)>0 and np.ptp(y)>0:
                    expected=spearmanr(x,y).statistic;references.append(expected)
                row=per_slide[(per_slide.feature==i)&(per_slide.slide==slide)&(per_slide.target==target)].iloc[0]
                np.testing.assert_allclose(row.spearman,expected,atol=1e-12,equal_nan=True)
                assert row.valid_observations==len(x);checked+=1
            row=exported[(exported.feature==i)&(exported.target==target)].iloc[0]
            np.testing.assert_allclose(row.spearman,np.mean(references) if references else np.nan,atol=1e-12,equal_nan=True)
            assert row.contributing_slides==len(references)
    report=dict(status='passed',reference_checks=checked,seconds=time.perf_counter()-start)
    (out/'numerical_audit.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    from dann.config import load_config
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('dann/config.yaml'))
    print(json.dumps(audit(load_config(parser.parse_args().config)),indent=2))
