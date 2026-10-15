/* peak-similarity-viewer: no network dependencies; one shared row viewport. */
'use strict';
(() => {
  const data = JSON.parse(document.getElementById('metadata').textContent);
  const packedNode = document.getElementById('packed');
  let encoded = packedNode.textContent.trim();
  const buffer = new ArrayBuffer(data.n * (data.n + 1) / 2 * 4);
  const bytes = new Uint8Array(buffer);
  let offset = 0;
  for (let i = 0; i < encoded.length; i += 4 * 1024 * 1024) {
    const chunk = atob(encoded.slice(i, i + 4 * 1024 * 1024));
    for (let j = 0; j < chunk.length; j++) bytes[offset++] = chunk.charCodeAt(j);
  }
  encoded = null;
  packedNode.remove();
  const packed = new DataView(buffer);
  const n = data.n, size = Math.min(800, Math.max(400, n)), treeWidth = 150, stripWidth = 28;
  const left = treeWidth + data.strips.length * stripWidth;
  const canvas = document.getElementById('matrix');
  canvas.width = left + size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const tip = document.getElementById('tip');
  const headings = document.getElementById('headings');
  const treeHeading = document.createElement('div');
  treeHeading.style.width = treeWidth + 'px'; treeHeading.textContent = 'Original hierarchy'; headings.append(treeHeading);
  const headerHeight = Math.max(300, ...data.strips.map(s => (s.label.length + 16) * 8 + 40));
  headings.style.height = headerHeight + 'px';
  for (const strip of data.strips) {
    const heading = document.createElement('div'); heading.className = 'heading'; heading.style.height = (headerHeight - 10) + 'px';
    heading.textContent = strip.label + (strip.kind === 'association' ? ' (tails clipped)' : strip.kind === 'intensity' ? ' (log10)' : '');
    headings.append(heading);
    if (strip.kind === 'association') {
      const legend = document.createElement('div'); legend.className = 'legend';
      legend.textContent = strip.label + ': limits ±' + formatNumber(strip.limit) +
        '; signed power ' + formatNumber(strip.gamma) + '; ' + strip.clipped + ' clipped tails. Ticks: ' +
        [-1, -.5, 0, .5, 1].map(v => formatNumber(v * strip.limit)).join(', ');
      document.getElementById('legends').append(legend);
    }
  }
  const similarityLegend = document.createElement('div');
  similarityLegend.textContent = 'Cosine similarity: blue −1 · white 0 · red +1';
  document.getElementById('legends').append(similarityLegend);
  let view = {x: 0, y: 0, span: n}, scheduled = false;
  function at(row, col) {
    if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row >= n || col >= n) return NaN;
    if (row > col) [row, col] = [col, row];
    return packed.getFloat32((row * n - row * (row - 1) / 2 + col - row) * 4, true);
  }
  const rowAt = pixel => Math.min(n - 1, Math.max(0, Math.floor(view.y + pixel * view.span / size)));
  const colAt = pixel => Math.min(n - 1, Math.max(0, Math.floor(view.x + pixel * view.span / size)));
  function draw() {
    scheduled = false;
    const image = ctx.createImageData(size, size);
    for (let y = 0; y < size; y++) {
      const row = rowAt(y + .5);
      for (let x = 0; x < size; x++) {
        const value = at(row, colAt(x + .5));
        const rgb = data.palette[Math.max(0, Math.min(1024, Math.round((value + 1) * 512)))];
        const index = (y * size + x) * 4;
        image.data[index] = rgb[0]; image.data[index + 1] = rgb[1]; image.data[index + 2] = rgb[2]; image.data[index + 3] = 255;
      }
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(image, left, 0);
    for (let y = 0; y < size; y++) {
      const row = rowAt(y + .5);
      data.strips.forEach((strip, i) => {
        ctx.fillStyle = `rgb(${strip.colors[row].join(',')})`;
        ctx.fillRect(treeWidth + i * stripWidth, y, stripWidth - 2, 1);
      });
    }
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, treeWidth - 3, size); ctx.clip();
    const maxHeight = Math.max(1e-12, ...data.tree.map(branch => branch[1][1]));
    ctx.strokeStyle = '#444'; ctx.lineWidth = .7; ctx.beginPath();
    for (const branch of data.tree) {
      if (branch[3][0] < view.y - 1 || branch[0][0] > view.y + view.span) continue;
      branch.forEach(([row, height], i) => {
        const x = treeWidth - 6 - height / maxHeight * (treeWidth - 12), y = (row + .5 - view.y) / view.span * size;
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
    }
    ctx.stroke(); ctx.restore();
    if (view.span < 80) {
      ctx.strokeStyle = '#0002'; ctx.lineWidth = .5; ctx.beginPath();
      for (let row = Math.ceil(view.y); row < view.y + view.span; row++) {
        const y = (row - view.y) / view.span * size;
        ctx.moveTo(treeWidth, y); ctx.lineTo(left + size, y);
      }
      for (let col = Math.ceil(view.x); col < view.x + view.span; col++) {
        const x = left + (col - view.x) / view.span * size;
        ctx.moveTo(x, 0); ctx.lineTo(x, size);
      }
      ctx.stroke();
    }
    document.getElementById('status').textContent = `Rows ${rowAt(0)}–${rowAt(size - .01)}, columns ${colAt(0)}–${colAt(size - .01)} of ${n}`;
  }
  function setView(x, y, span) {
    span = Math.min(n, Math.max(1, span));
    view = {x: Math.max(0, Math.min(n - span, x)), y: Math.max(0, Math.min(n - span, y)), span};
    if (!scheduled) { scheduled = true; requestAnimationFrame(draw); }
  }
  const reset = () => setView(0, 0, n);
  document.getElementById('reset').onclick = reset; canvas.ondblclick = reset;
  document.getElementById('inspect').onclick = () => {
    const row = Number(document.getElementById('row').value), col = Number(document.getElementById('col').value);
    if (Number.isFinite(row) && Number.isFinite(col)) setView(col - 5, row - 5, 10);
  };
  function point(event) {
    const box = canvas.getBoundingClientRect();
    return [(event.clientX - box.left) * canvas.width / box.width, (event.clientY - box.top) * canvas.height / box.height];
  }
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const [px, py] = point(event), fx = Math.max(0, Math.min(1, (px - left) / size)), fy = py / size;
    const span = Math.min(n, Math.max(1, view.span * Math.exp(event.deltaY * .002)));
    setView(view.x + fx * (view.span - span), view.y + fy * (view.span - span), span);
  }, {passive: false});
  let dragging = null;
  canvas.onpointerdown = event => { dragging = {point: point(event), view: {...view}}; canvas.setPointerCapture(event.pointerId); };
  canvas.onpointerup = canvas.onpointercancel = () => { dragging = null; };
  canvas.onpointerleave = () => { tip.style.display = 'none'; };
  canvas.onpointermove = event => {
    const [px, py] = point(event);
    if (dragging) {
      const old = dragging.view;
      setView(old.x - (px - dragging.point[0]) / size * old.span, old.y - (py - dragging.point[1]) / size * old.span, old.span);
      tip.style.display = 'none'; return;
    }
    const row = rowAt(py), col = colAt(px - left);
    let message = `Row ${row} · peak ${data.peaks[row]} · m/z ${formatNumber(data.mz[row], 'mz')}`;
    if (px >= left) {
      message += `\nColumn ${col} · peak ${data.peaks[col]} · m/z ${formatNumber(data.mz[col], 'mz')}\nCosine ${formatNumber(at(row, col))}`;
    } else if (px >= treeWidth) {
      const strip = data.strips[Math.floor((px - treeWidth) / stripWidth)];
      const value = strip.values[row];
      message += `\n${strip.label}: ${formatNumber(value, strip.kind === 'family' ? 'integer' : 'float')}`;
      if (strip.kind === 'association' && Math.abs(value) > strip.limit) message += '\nColor clipped; value unchanged';
    }
    tip.textContent = message; tip.style.display = 'block';
    tip.style.left = Math.min(event.clientX + 16, window.innerWidth - 440) + 'px';
    tip.style.top = Math.min(event.clientY + 16, window.innerHeight - 130) + 'px';
  };
  window.peakViewer = {at, setView, get view() {return {...view};}, rowAt, colAt, data, left, size, ready: true};
  draw();
})();
