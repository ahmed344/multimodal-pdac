"""Headless offline peak-viewer audit (requires Playwright and Chromium).

Run after peak generation: ``python -m dann.verification.peak_browser_audit
--output-dir path/to/analysis/peaks``. Browser dependencies are verification-only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from dann.artifact_layout import peak_file
from dann.html_display import format_number


def audit(output: Path, screenshots: Path | None = None) -> dict:
    """Check all HTML files offline, exact matrix hover, labels, and shared zoom."""
    from playwright.sync_api import sync_playwright
    started = time.perf_counter()
    frame = pd.read_csv(peak_file(output, 'peak_annotations.csv'), float_precision='round_trip')
    order = np.load(peak_file(output, 'peak_order.npy'))
    run = json.loads((peak_file(output, 'peak_run.json')).read_text())
    cache = output.parent/'.analysis-cache'/'peak-hierarchy'/run['cache_entries']['peak-hierarchy']
    manifest = json.loads((cache/'manifest.json').read_text())
    similarity = np.load(cache/manifest['arrays']['similarity']['file'], mmap_mode='r')
    displays = json.loads((peak_file(output, 'ihc_display_settings.json')).read_text())
    if screenshots:
        screenshots.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, env={k:v for k,v in os.environ.items() if k != 'DISPLAY'}, args=['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader','--enable-webgl'])
        for path in sorted(output.rglob('*.html')):
            context = browser.new_context(viewport={'width':1600, 'height':1500}, offline=True)
            page = context.new_page()
            errors, requests = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(request.url) if request.url.startswith(('http:', 'https:')) else None)
            page.goto(path.resolve().as_uri(), timeout=120000)
            checked_cells = 0
            if path.name.startswith('cd8_families_'):
                page.wait_for_function('window.cd8Viewer && cd8Viewer.ready')
                assert page.locator('.panel').count() == page.evaluate('cd8Viewer.rows*cd8Viewer.columns')
                page.wait_for_function("Array.from(document.querySelectorAll('img[src]')).length > 0 && Array.from(document.querySelectorAll('img[src]')).every(i=>i.complete && i.naturalWidth>0)")
                page.locator('#slide').select_option(index=10 if page.evaluate('cd8Viewer.rows')>10 else 0)
                page.locator('#zoom').fill('500')
                page.locator('#zoom').dispatch_event('input')
                page.locator('#reset').click()
                assert page.locator('#viewport').evaluate('(e)=>e.scrollTop') == 0
            elif path.name == 'full_similarity.html':
                page.wait_for_function('window.peakViewer && peakViewer.ready', timeout=120000)
                n = len(order)
                pairs = [[0, n-1], [n-1, 0], [n//2, n//3], [n-1, n-1], [1, 2]]
                actual = page.evaluate('(pairs) => pairs.map(([r,c]) => peakViewer.at(r,c))', pairs)
                for (r, c), value in zip(pairs, actual):
                    assert value == float(similarity[order[r], order[c]])
                checked_cells = len(pairs)
                assert page.locator('.heading').count() == 8
                assert page.evaluate("Array.from(document.querySelectorAll('.heading')).every(e => getComputedStyle(e).writingMode === 'vertical-rl' && e.scrollHeight <= e.clientHeight)")
                # Both triangle halves, diagonal, and matrix/strip row alignment after zoom.
                r, c = n//2, n//3
                page.evaluate('([r,c]) => peakViewer.setView(c-2, r-2, 5)', [r, c])
                page.wait_for_function('peakViewer.view.span === 5')
                box = page.locator('#matrix').bounding_box()
                geometry = page.evaluate('({left:peakViewer.left,size:peakViewer.size})')
                py = box['y'] + geometry['size']*.5
                page.mouse.move(box['x']+geometry['left']+geometry['size']*.5, py)
                text = page.locator('#tip').inner_text()
                assert text.split('Cosine ')[1] == format_number(similarity[order[r], order[c]])
                assert f'peak {order[r]}' in text and f'peak {order[c]}' in text
                # The first association strip uses the exact same row as the matrix.
                page.mouse.move(box['x']+150+4*28+14, py)
                strip_text = page.locator('#tip').inner_text()
                target = list(displays)[0]
                expected = frame[f'spearman_{target}'].iloc[order[r]]
                value_text = strip_text.split('\n')[1].split(': ')[-1]
                if np.isfinite(expected):
                    assert value_text == format_number(expected)
                assert f'peak {order[r]}' in strip_text
                # Exercise wheel and drag event paths as well as direct viewport control.
                page.mouse.move(box['x']+geometry['left']+geometry['size']*.5, py)
                page.mouse.wheel(0, -200)
                page.wait_for_function('peakViewer.view.span < 5')
                old = page.evaluate('peakViewer.view')
                page.mouse.down(); page.mouse.move(box['x']+geometry['left']+geometry['size']*.5+30, py+20); page.mouse.up()
                assert page.evaluate('peakViewer.view') != old
                page.locator('#reset').click()
                page.wait_for_function('peakViewer.view.span === peakViewer.data.n')
            else:
                page.wait_for_function("document.querySelector('.js-plotly-plot')._fullLayout", timeout=120000)
                if 'similarity' in path.name:
                    if path.name.startswith('top_'):
                        members = pd.read_csv(peak_file(output, 'top_similarity_membership.csv')).peak_index.to_numpy()
                    else:
                        family = int(path.stem.split('_')[1])
                        members = order[frame.family.to_numpy()[order] == family]
                    n = len(members)
                    r, c = n//2, n//3
                    details = page.evaluate('''([r,c]) => {
                        const p=document.querySelector('.js-plotly-plot'), trace=p._fullData.at(-1);
                        const values={z:trace.z[r][c], mz:Array.from(trace.customdata[r][c]),
                            labels:p._fullLayout.annotations.map(a=>a.textangle),
                            boxes:Array.from(document.querySelectorAll('.annotation')).map(e=>{const b=e.getBoundingClientRect();return [b.y,b.bottom]})};
                        Plotly.Fx.hover(p,[{curveNumber:p.data.length-1,xval:c,yval:r}], trace.xaxis+trace.yaxis);
                        return values;
                    }''', [r, c])
                    assert details['z'] == float(similarity[members[r], members[c]])
                    np.testing.assert_array_equal(details['mz'], frame.mz.to_numpy()[members[[r,c]]])
                    assert details['labels'] == [-90]*8
                    assert all(a >= 0 and b <= 800 for a, b in details['boxes'])
                    hover = page.locator('.hoverlayer').text_content()
                    assert 'Cosine' in hover and 'm/z' in hover
                    checked_cells = 1
                    # Plotly matched axes must propagate matrix zoom to every strip/tree.
                    ranges = page.evaluate('''async () => {
                        const p=document.querySelector('.js-plotly-plot'), trace=p._fullData.at(-1);
                        const axis='yaxis'+trace.yaxis.slice(1), n=trace.z.length;
                        const limits=n>3 ? [2.5,.5] : [n-.5,-.5];
                        await Plotly.relayout(p,{[axis+'.range']:limits});
                        return Object.keys(p._fullLayout).filter(k=>/^yaxis[0-9]*$/.test(k)).map(k=>p._fullLayout[k].range);
                    }''')
                    assert all(values == ranges[0] for values in ranges)
                else:
                    # Exercise an IHC selector, checking original values remain hover data.
                    target = list(displays)[0]
                    values = page.evaluate('''async () => {
                        const p=document.querySelector('.js-plotly-plot'), i=5;
                        await Plotly.restyle(p,{visible:p.data.map((_,j)=>j===i)});
                        const t=p._fullData[i];
                        return {raw:Array.from(t.customdata[0]), color:t.marker.color[0], ticks:t.marker.colorbar.ticktext};
                    }''')
                    raw = values['raw'][3]
                    limit, gamma = displays[target]['limit'], displays[target]['gamma']
                    if raw is not None:
                        np.testing.assert_allclose(values['color'], np.sign(raw)*min(abs(raw/limit),1)**gamma)
                    assert len(values['ticks']) == 5
            assert 'WebGL is not supported' not in page.locator('body').inner_text()
            assert not errors, (path.name, errors)
            assert not requests, (path.name, requests)
            if screenshots and path.name in ('full_similarity.html', 'top_512_similarity.html', 'family_01_similarity.html'):
                page.screenshot(path=str(screenshots/(path.stem+'.png')), full_page=True)
            results.append(dict(file=path.name, exact_cells_checked=checked_cells, offline=True))
            print(f'Browser passed: {path.name}', flush=True)
            context.close()
        browser.close()
    report = dict(status='passed', html_count=len(results), seconds=time.perf_counter()-started, files=results)
    (output/'reports').mkdir(exist_ok=True)
    (output/'reports'/'peak_browser_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--screenshots', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir, args.screenshots), indent=2))


if __name__ == '__main__':
    main()
