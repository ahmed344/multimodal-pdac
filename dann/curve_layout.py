"""Canvas dimensions shared by curve rendering and artifact memory budgeting."""

FIGURE_WIDTH = 24.0
ROW_HEIGHT = 3.5
HEADER_HEIGHT = 2.8
FOOTER_HEIGHT = 0.5
FIGURE_ROWS = {"loss": 6, "ihc": 10}


def figure_size(kind: str) -> tuple[float, float]:
    """Return the fixed four-column figure size in inches."""
    return FIGURE_WIDTH, FIGURE_ROWS[kind] * ROW_HEIGHT + HEADER_HEIGHT + FOOTER_HEIGHT


def render_reservation_bytes(dpi: int) -> int:
    """Reserve four RGBA buffers for the larger of two sequentially drawn figures."""
    return int(max(width * height for width, height in map(figure_size, FIGURE_ROWS)) * dpi**2 * 4 * 4)
