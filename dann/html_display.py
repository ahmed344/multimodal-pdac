"""Display-only number formatting shared by offline scientific viewers."""
from pathlib import Path
import math


def format_number(value, kind: str = 'float') -> str:
    """Format a value without rounding the numerical data behind a view."""
    if value is None or not math.isfinite(float(value)):
        return 'Unavailable'
    value = float(value)
    if kind == 'integer':
        return str(int(value))
    if kind == 'mz':
        return f'{value:.4f}'
    if value == 0:
        return '0'
    magnitude = abs(value)
    if magnitude < .0001:
        return f'{value:.2e}'
    digits = 2 if magnitude >= 1 else 3 if magnitude >= .01 else 4
    return f'{value:.{digits}f}'


def formatter_script() -> str:
    return Path(__file__).with_name('html_display.js').read_text()


def write_plotly(figure, path: Path) -> None:
    """Embed Plotly and format visible numeric ticks after drawing and zooming."""
    for trace in figure.data:
        if hasattr(trace, 'marker') and hasattr(trace.marker, 'colorbar'):
            trace.marker.colorbar.tickformat = '.8g'
        if hasattr(trace, 'colorbar'):
            trace.colorbar.tickformat = '.8g'
    figure.update_xaxes(tickformat='.8g')
    figure.update_yaxes(tickformat='.8g')
    script = formatter_script() + '''
const plot = document.getElementById('{plot_id}');
function formatTicks() {
  plot.querySelectorAll('.xtick text,.ytick text,.cbaxis text').forEach(node => {
    const raw = node.textContent.replaceAll('−','-');
    if (/^[+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:e[+-]?\\d+)?$/i.test(raw)) {
      const integer = node.closest('.cbaxis') && plot.data.some(t => t.visible !== false && t.marker && t.marker.colorbar && t.marker.colorbar.title && /family|batch/i.test(t.marker.colorbar.title.text || ''));
      const mz = node.closest('.cbaxis') && plot.data.some(t => t.visible !== false && t.marker && t.marker.colorbar && t.marker.colorbar.title && t.marker.colorbar.title.text === 'm/z');
      node.textContent = formatNumber(Number(raw), integer ? 'integer' : mz ? 'mz' : 'float');
    }
  });
}
plot.on('plotly_afterplot',formatTicks); formatTicks();
'''
    figure.write_html(path, include_plotlyjs=True, full_html=True, post_script=script)
