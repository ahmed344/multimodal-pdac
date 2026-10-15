"""Offline bounded-resolution full similarity viewer with exact packed FP32 hover."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.cluster import hierarchy


def pack_upper(matrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Pack the ordered symmetric upper triangle, including the original diagonal."""
    n = len(order)
    packed = np.empty(n*(n+1)//2, dtype='<f4')
    start = 0
    for i, row in enumerate(order):
        size = n-i
        packed[start:start+size] = matrix[row, order[i:]]
        start += size
    return packed


def plot_full_similarity(matrix: np.ndarray, order: np.ndarray, mz: np.ndarray,
                         annotations: list[dict], linkage: np.ndarray, path: Path) -> None:
    """Write one packed matrix and an offline canvas viewer with a shared row viewport."""
    from dann.peak_plots import _color_options, restricted_linkage
    order = np.asarray(order, dtype=np.int64)
    n = len(order)
    count = max(int(np.max(a['values'])) for a in annotations if a['kind'] == 'family')
    strips = []
    for annotation in annotations:
        values = np.asarray(annotation['values'])[order]
        display = values.copy()
        if annotation['kind'] == 'intensity':
            display = np.log10(np.where(values > 0, values, np.nan))
        options = _color_options(annotation, count)
        cmap = options['cmap']
        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]
        norm = options.get('norm') or matplotlib.colors.Normalize(options.get('vmin'), options.get('vmax'))
        rgba = cmap(norm(np.ma.masked_invalid(display)), bytes=True)
        rgba[~np.isfinite(display)] = [204, 204, 204, 255]
        strips.append(dict(label=annotation['label'], kind=annotation['kind'],
            values=[float(v) if np.isfinite(v) else None for v in values], colors=rgba[:, :3].tolist(),
            limit=annotation.get('limit'), gamma=annotation.get('gamma'), clipped=annotation.get('clipped_count', 0)))
    dendrogram = hierarchy.dendrogram(restricted_linkage(linkage, order), no_plot=True) if n > 1 else {'icoord': [], 'dcoord': []}
    tree = [[[(float(y)-5)/10, float(x)] for y, x in zip(ys, xs)]
            for ys, xs in zip(dendrogram['icoord'], dendrogram['dcoord'])]
    metadata = dict(n=n, mz=mz[order].tolist(), peaks=order.tolist(), strips=strips, tree=tree,
        palette=matplotlib.colormaps['RdBu_r'](np.linspace(0, 1, 1025), bytes=True)[:, :3].tolist())
    from dann.html_display import formatter_script
    script = formatter_script() + Path(__file__).with_name('peak_viewer.js').read_text()
    with path.open('w') as output:
        output.write('''<!doctype html><html><head><meta charset="utf-8"><title>Full peak similarity</title>
<style>body{font:14px system-ui;margin:20px;color:#222}button,input{font:inherit;margin-right:8px}
#headings{display:flex;height:300px;align-items:flex-end;margin-top:12px}
.heading{width:28px;flex:none;height:290px;writing-mode:vertical-rl;transform:rotate(180deg);box-sizing:border-box;padding:8px 3px}
#view{position:relative;width:max-content}canvas{display:block;touch-action:none;cursor:crosshair}
#tip{display:none;position:fixed;background:#fffffff5;border:1px solid #888;padding:10px;pointer-events:none;z-index:2;white-space:pre}
#legends{display:flex;gap:22px;flex-wrap:wrap;max-width:1250px;margin-top:16px}.legend{max-width:260px}
</style></head><body><h1>All peak-pair cosine similarities</h1>
<p>Wheel to zoom; drag to pan; double-click to reset. The overview samples cell centers at up to 800 × 800 pixels.
Zoom exposes individual cells. Hover always reads the original FP32 cell, including clipped annotation values.</p>
<button id="reset">Reset</button><label>Row <input id="row" type="number" min="0" value="0" style="width:80px"></label>
<label>Column <input id="col" type="number" min="0" value="0" style="width:80px"></label><button id="inspect">Inspect cells</button>
<span id="status"></span><div id="view"><div id="headings"></div><canvas id="matrix"></canvas></div>
<div id="legends"></div><div id="tip"></div>
<script id="metadata" type="application/json">''')
        output.write(json.dumps(metadata, allow_nan=False).replace('<', '\\u003c'))
        output.write('</script><script id="packed" type="application/octet-stream">')
        # Encode bounded chunks, each divisible by three bytes except the last.
        packed = pack_upper(matrix, order)
        raw = memoryview(packed).cast('B')
        for start in range(0, len(raw), 3*1024*1024):
            output.write(base64.b64encode(raw[start:start+3*1024*1024]).decode('ascii'))
        output.write('</script><script>\n'+script+'\n</script></body></html>')
