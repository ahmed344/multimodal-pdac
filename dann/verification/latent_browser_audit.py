"""Headless offline checks of the five latent viewers and bounded CD8 overviews."""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from dann.html_display import format_number


def audit(root, include_comparisons=True):
    from playwright.sync_api import sync_playwright
    started=time.perf_counter();output=root/'latent'
    source=pd.read_csv(root/'tables/latent_embeddings.csv',float_precision='round_trip')
    correlation=np.load(output/'tables/feature_correlations.npy');order=np.load(output/'tables/feature_order.npy')
    results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,env={k:v for k,v in os.environ.items() if k != 'DISPLAY'},args=['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader','--enable-webgl'])
        for path in sorted(output.glob('*.html')):
            context=browser.new_context(viewport={'width':1600,'height':1100},offline=True)
            page=context.new_page();errors=[];requests=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
            page.goto(path.resolve().as_uri(),timeout=120000)
            if path.name in ('sample_umap.html','latent_ihc_scatter.html'):
                page.wait_for_function('window.latentViewer && latentViewer.ready',timeout=120000)
                assert page.evaluate('latentViewer.metadata.features.length')==len(source.filter(regex='^latent_').columns)
                for feature in ('latent_0','latent_127','latent_255'):
                    if feature not in source:continue
                    values=page.evaluate('(name)=>latentViewer.loadColumn(name).then(v=>[v[0],v[Math.floor(v.length/2)],v[v.length-1]])',feature)
                    np.testing.assert_array_equal(values,source[feature].iloc[[0,len(source)//2,len(source)-1]].to_numpy())
                page.locator('#color').select_option('latent_255' if 'latent_255' in source else 'latent_0')
                page.wait_for_function('latentViewer.ready',timeout=120000)
                page.locator('#slide').select_option(index=1)
                page.wait_for_function('latentViewer.ready',timeout=120000)
                assert page.evaluate('latentViewer.shown')<len(source)
                if path.name=='latent_ihc_scatter.html':
                    page.locator('#scale').select_option('logit');page.wait_for_function('latentViewer.ready',timeout=120000)
                else:
                    page.locator('#color').select_option('Density_CD8');page.wait_for_function('latentViewer.ready',timeout=120000)
                palette=page.evaluate("()=>{const t=document.getElementById('plot')._fullData.at(-1);return t.marker.colorscale;}")
                name='turbo' if path.name=='sample_umap.html' else 'RdBu_r'
                expected_palette=page.evaluate('(name)=>latentViewer.metadata.palettes[name]',name)
                assert palette==expected_palette
                assert page.evaluate("document.getElementById('plot')._fullLayout.xaxis.title.text") not in ('','Click to enter X axis title')
                raw=page.evaluate("()=>{const p=document.getElementById('plot'),t=p.data.at(-1);return {data:t.customdata[0],text:t.text[0]};}")
                assert 'Row '+str(int(raw['data'][0])) in raw['text']
                assert format_number(raw['data'][2]) in raw['text']
                page.evaluate("Plotly.relayout(document.getElementById('plot'),{'xaxis.range':[-1,1]})")
                assert page.evaluate("document.getElementById('plot').layout.xaxis.range")==[-1,1]
                for value in [1.23456,.123456,.00123456,.0000123456,0,-0.0000123456]:
                    assert page.evaluate('(v)=>formatNumber(v)',value)==format_number(value)
            else:
                page.wait_for_function("document.querySelector('.js-plotly-plot')._fullLayout",timeout=120000)
                if path.name=='feature_correlations.html':
                    r,c=len(order)//2,len(order)//3
                    value=page.evaluate('([r,c])=>{const p=document.querySelector(".js-plotly-plot"),t=p._fullData.at(-1);return t.z[r][c];}',[r,c])
                    expected=correlation[order[r],order[c]]
                    np.testing.assert_allclose(value,expected)
                    page.evaluate('''([r,c])=>{const p=document.querySelector('.js-plotly-plot'),t=p.data.at(-1);Plotly.Fx.hover(p,[{curveNumber:p.data.length-1,xval:c,yval:r}],t.xaxis+t.yaxis);}''',[r,c])
                    assert format_number(expected) in page.locator('.hoverlayer').text_content()
                    ranges=page.evaluate('''async()=>{const p=document.querySelector('.js-plotly-plot'),t=p.data.at(-1);await Plotly.relayout(p,{['yaxis'+t.yaxis.slice(1)+'.range']:[10,2]});return Object.keys(p._fullLayout).filter(k=>/^yaxis[0-9]*$/.test(k)).map(k=>p._fullLayout[k].range);}''')
                    assert all(r==ranges[0] for r in ranges)
                else:
                    page.evaluate("()=>{const p=document.querySelector('.js-plotly-plot');return Plotly.restyle(p,{visible:p.data.map((_,i)=>i===1)});}")
            assert 'WebGL is not supported' not in page.locator('body').inner_text()
            assert not errors,(path.name,errors)
            assert not requests,(path.name,requests)
            if path.name in ('sample_umap.html','latent_ihc_scatter.html'):
                page.evaluate("Plotly.relayout(document.getElementById('plot'),{'xaxis.autorange':true,'yaxis.autorange':true})")
            previews=Path('/tmp/latent-previews');previews.mkdir(exist_ok=True)
            page.screenshot(path=str(previews/(path.stem+'.png')))
            results.append(path.name);print('Browser passed:',path.name,flush=True);context.close()
        for path in sorted((root/'peaks/spatial/cd8_comparison').glob('*.png')) if include_comparisons else []:
            with Image.open(path) as image:
                image.load();assert max(image.size)<=8000 and image.width*image.height<=64_000_000
            page=browser.new_page();page.goto(path.resolve().as_uri(),timeout=120000)
            page.wait_for_function('document.images.length && document.images[0].complete && document.images[0].naturalWidth>0',timeout=120000)
            page.close();print('Browser opened:',path.name,flush=True)
        browser.close()
    report=dict(status='passed',html_files=results,seconds=time.perf_counter()-started)
    (output/'browser_audit.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(audit(parser.parse_args().output_dir),indent=2))
