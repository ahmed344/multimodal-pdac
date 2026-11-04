"""Canvas dimensions shared by curve rendering and artifact memory budgeting."""

import math

FIGURE_WIDTH = 24.0
ROW_HEIGHT = 3.5
HEADER_HEIGHT = 2.8
FOOTER_HEIGHT = 0.5
FIGURE_ROWS = {"loss": 6, "ihc": 10}


def figure_size(kind: str, rows: int | None = None) -> tuple[float, float]:
    """Return the four-column canvas size for the actual panel layout."""
    return FIGURE_WIDTH, (FIGURE_ROWS[kind] if rows is None else rows) * ROW_HEIGHT + HEADER_HEIGHT + FOOTER_HEIGHT


def render_reservation_bytes(dpi: int, rows: dict[str, int] | None = None) -> int:
    """Reserve four RGBA buffers for the larger of two sequentially drawn figures."""
    return math.ceil(max(width * height for width, height in (figure_size(k, v) for k, v in (rows or FIGURE_ROWS).items())) * dpi**2 * 4 * 4)
