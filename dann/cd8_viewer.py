"""Offline spatial comparison pages backed by individually decoded native grids."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import colormaps
from dann.html_display import format_number


def write_cd8_viewer(slides, cd8, signal, families, limits, cd8_limit, cd8_label, region, path):
    """Write lossless per-panel images and an offline viewport with bounded decoding."""
    from dann.peak_analysis import spatial_grid
    asset_dir = path.with_name(path.stem + '_assets')
    asset_dir.mkdir(exist_ok=True)
    titles = [cd8_label, *[f'Family {int(f)}' for f in families]]
    body = ['<div class="head corner">Slide</div>']
    for title, limit in zip(titles, np.r_[cd8_limit, limits]):
        body.append(f'<div class="head">{html.escape(title)}<br><small>0 – {format_number(limit)}</small><div class="bar"></div></div>')
    for row, slide in enumerate(slides):
        body.append(f'<div class="row-label" id="slide-{row}">{html.escape(slide["name"])}</div>')
        for col, limit in enumerate(np.r_[cd8_limit, limits]):
            grid = spatial_grid(cd8 if col == 0 else signal[:, col-1], slide, region)
            pixels = colormaps['turbo'](np.clip(np.nan_to_num(grid, nan=0.)/max(limit, 1e-30), 0, 1), bytes=True)
            pixels[~np.isfinite(grid)] = [221, 221, 221, 255]
            name = f'{row:03d}_{col:02d}.webp'
            Image.fromarray(pixels).save(asset_dir/name, lossless=True)
            note = 'No lesion' if region == 'lesion' and slide['crop'] is None else 'No observed CD8' if col == 0 and not np.isfinite(grid).any() else ''
            title = html.escape(f'{slide["name"]} · {titles[col]} · 0 – {format_number(limit)}', quote=True)
            body.append(f'<div class="panel" title="{title}"><img data-src="{asset_dir.name}/{name}" alt="{title}"><span>{note}</span></div>')
    options = ''.join(f'<option value="{i}">{html.escape(s["name"])}</option>' for i, s in enumerate(slides))
    palette = ','.join('#%02x%02x%02x' % tuple(rgb[:3]) for rgb in colormaps['turbo'](np.linspace(0, 1, 16), bytes=True))
    document = '''<!doctype html><html><head><meta charset="utf-8"><title>CD8 and family signals</title>
<style>body{font:14px system-ui;color:#222;margin:20px}.controls{display:flex;gap:20px;margin:12px 0}
#viewport{height:78vh;overflow:auto;border:1px solid #bbb;position:relative;cursor:grab}
#grid{display:grid;grid-template-columns:140px repeat(COLUMNS,var(--cell));--cell:220px;width:max-content;gap:4px}
.head{position:sticky;top:0;background:white;z-index:3;height:65px;padding:8px;box-sizing:border-box;text-align:center}
.row-label{position:sticky;left:0;background:white;z-index:2;display:flex;align-items:center;padding:8px}
.corner{left:0;z-index:4}.panel{width:var(--cell);height:var(--cell);background:#ddd;position:relative;display:grid;place-items:center}
.panel img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}.panel span{position:absolute;background:#fffd}
.bar{height:8px;margin:5px;background:linear-gradient(to right,PALETTE)}button,select,input{font:inherit}</style></head>
<body><!-- cd8-comparison-viewer --><h1>Observed CD8 and family signals — REGION</h1>
<p>All slides and families, using identical column scales across tissue and lesion. Gray areas have no data. Detailed panels retain the native spatial grid.</p>
<div class="controls"><label>Slide <select id="slide">OPTIONS</select></label><label>Panel size <input id="zoom" type="range" min="100" max="1000" value="220"></label><button id="reset">Reset view</button></div>
<div id="viewport"><div id="grid">BODY</div></div><script>
const viewport=document.getElementById('viewport'), grid=document.getElementById('grid');
const observer=new IntersectionObserver(entries=>entries.forEach(e=>{const img=e.target;
 if(e.isIntersecting) img.src=img.dataset.src; else img.removeAttribute('src');}),{root:viewport,rootMargin:'250px'});
document.querySelectorAll('img').forEach(img=>observer.observe(img));
document.getElementById('zoom').oninput=e=>grid.style.setProperty('--cell',e.target.value+'px');
document.getElementById('slide').onchange=e=>{viewport.scrollTop=document.getElementById('slide-'+e.target.value).offsetTop-80;};
document.getElementById('reset').onclick=()=>{grid.style.setProperty('--cell','220px');document.getElementById('zoom').value=220;viewport.scrollTo(0,0);};
let drag=null;viewport.onpointerdown=e=>{drag=[e.clientX,e.clientY,viewport.scrollLeft,viewport.scrollTop];viewport.setPointerCapture(e.pointerId);e.preventDefault();};
viewport.onpointermove=e=>{if(drag){viewport.scrollLeft=drag[2]+drag[0]-e.clientX;viewport.scrollTop=drag[3]+drag[1]-e.clientY;}};
viewport.onpointerup=viewport.onpointercancel=()=>drag=null;
window.cd8Viewer={ready:true,rows:ROWS,columns:COLUMNS};
</script></body></html>'''
    for key, value in [('COLUMNS',str(len(titles))),('ROWS',str(len(slides))),('REGION',region),('PALETTE',palette),('OPTIONS',options),('BODY',''.join(body))]:
        document = document.replace(key,value)
    path.write_text(document)
