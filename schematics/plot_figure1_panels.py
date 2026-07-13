"""Generate the compact pipeline overview schematic.

Usage:
    uv run --extra heatmap python schematics/plot_figure1_panels.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FIGURE_WIDTH_PT = 520.0
MARGIN = 6.0
OVERVIEW_HEIGHT = 72.0

DARK = "#222222"
BLUE = "#0057b8"
RED = "#d62728"
LIGHT_BLUE = "#dce9f8"
LIGHT_GREEN = "#ddf1e9"
LIGHT_ORANGE = "#f7ecd3"
LIGHT_RED = "#f8dedb"
LIGHT_GRAY = "#ededed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("schematics/plots"))
    parser.add_argument("--formats", nargs="+", default=["svg"], choices=["svg", "pdf"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_svgs = {"pipeline-overview": build_overview_panel()}
    for stem, svg in panel_svgs.items():
        svg_path = args.output_dir / f"{stem}.svg"
        if "svg" in args.formats or "pdf" in args.formats:
            svg_path.write_text(svg)
            print(f"Wrote {svg_path}")
        if "pdf" in args.formats:
            write_pdf(svg_path, args.output_dir / f"{stem}.pdf")


def build_overview_panel() -> str:
    width = FIGURE_WIDTH_PT
    height = OVERVIEW_HEIGHT + 2 * MARGIN
    return "\n".join(
        [
            svg_header(width, height),
            style_block(),
            overview_strip(MARGIN, MARGIN, width - 2 * MARGIN, OVERVIEW_HEIGHT),
            "</svg>\n",
        ]
    )


def svg_header(width: float, height: float) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
        f'<svg width="{width:g}pt" height="{height:g}pt" viewBox="0 0 {width:g} {height:g}" '
        'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
    )


def style_block() -> str:
    return """
<defs>
  <style type="text/css">
    text { font-family: DejaVu Sans, Arial, sans-serif; fill: #222222; }
    .panel-label { font-size: 13px; font-weight: 700; }
    .overview-title { font-size: 10px; font-weight: 700; }
    .overview-subtitle { font-size: 8.8px; }
    .overview-box { stroke: #222222; stroke-width: 1; rx: 5; ry: 5; }
    .mini-line { stroke: #222222; stroke-width: 1.1; fill: none; stroke-linecap: round; }
    .mini-thin { stroke: #222222; stroke-width: 0.7; fill: none; }
    .mini-red { stroke: #d62728; stroke-width: 0.9; fill: none; }
    .overview-arrow { stroke: #222222; stroke-width: 1.2; fill: none; marker-end: url(#arrowhead); }
  </style>
  <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 8 3 L 0 6 z" fill="#222222"/>
  </marker>
</defs>
"""


def overview_strip(x: float, y: float, width: float, height: float) -> str:
    steps = [
        ("Step 1", "partition", LIGHT_BLUE, mini_partition),
        ("Step 2", "covariance", LIGHT_GREEN, mini_covariance),
        ("Step 3", "vector", LIGHT_ORANGE, mini_vector),
        ("Step 4", "breakpoints", LIGHT_RED, mini_breakpoints),
        ("Step 5", "BED", LIGHT_GRAY, mini_bed),
    ]
    box_y = y + 3
    box_h = height - 6
    gap = 12.0
    box_w = (width - gap * (len(steps) - 1)) / len(steps)
    parts = []
    for i, (title, subtitle, fill, draw_mini) in enumerate(steps):
        bx = x + i * (box_w + gap)
        parts.append(f'<rect class="overview-box" x="{bx:g}" y="{box_y:g}" width="{box_w:g}" height="{box_h:g}" fill="{fill}"/>')
        parts.append(f'<text class="overview-title" x="{bx + box_w / 2:g}" y="{box_y + 11:g}" text-anchor="middle">{escape(title)}</text>')
        parts.append(f'<text class="overview-subtitle" x="{bx + box_w / 2:g}" y="{box_y + box_h - 5:g}" text-anchor="middle">{escape(subtitle)}</text>')
        parts.append(draw_mini(bx + 7, box_y + 16, box_w - 14, box_h - 30))
        if i < len(steps) - 1:
            y0 = box_y + box_h / 2
            parts.append(f'<path class="overview-arrow" d="M {bx + box_w + 2:g} {y0:g} L {bx + box_w + gap - 2.5:g} {y0:g}"/>')
    return "\n".join(parts)


def mini_partition(x: float, y: float, width: float, height: float) -> str:
    y_line = y + height * 0.38
    bar_y = y + height * 0.58
    parts = [f'<path class="mini-line" d="M {x:g} {y_line:g} L {x + width:g} {y_line:g}"/>']
    for fraction in (0.08, 0.22, 0.41, 0.58, 0.77, 0.92):
        tick_x = x + width * fraction
        parts.append(f'<path class="mini-thin" d="M {tick_x:g} {y_line - 4:g} L {tick_x:g} {y_line + 4:g}"/>')
    for start, end, fill in ((0.02, 0.36, LIGHT_GREEN), (0.30, 0.66, LIGHT_ORANGE), (0.58, 0.98, LIGHT_GREEN)):
        parts.append(f'<rect x="{x + width * start:g}" y="{bar_y:g}" width="{width * (end - start):g}" height="5" fill="{fill}" stroke="#222222" stroke-width="0.5"/>')
    for fraction in (0.36, 0.66):
        bp_x = x + width * fraction
        parts.append(f'<path class="mini-red" d="M {bp_x:g} {y_line - 9:g} L {bp_x:g} {bar_y + 7:g}"/>')
    return "\n".join(parts)


def mini_covariance(x: float, y: float, width: float, height: float) -> str:
    table_w = width * 0.30
    matrix_region_x = x + width * 0.55
    matrix_region_w = width * 0.34
    parts = draw_genotype_table(x, y + 1, table_w, height - 2)
    parts.append(f'<path class="overview-arrow" d="M {x + table_w + 5:g} {y + height / 2:g} L {matrix_region_x - 5:g} {y + height / 2:g}"/>')
    parts.extend(draw_covariance_matrix(matrix_region_x, y, matrix_region_w, height, scale=0.18))
    return "\n".join(parts)


def mini_vector(x: float, y: float, width: float, height: float) -> str:
    matrix_region_w = width * 0.34
    plot_x = x + width * 0.58
    plot_y = y + 4
    plot_w = width * 0.39
    plot_h = height - 8
    pts = [(0.0, 0.70), (0.18, 0.25), (0.34, 0.42), (0.52, 0.12), (0.70, 0.36), (0.9, 0.18)]
    d = " ".join(("M" if i == 0 else "L") + f" {plot_x + plot_w * px:g} {plot_y + plot_h * py:g}" for i, (px, py) in enumerate(pts))
    parts = draw_covariance_matrix(x, y, matrix_region_w, height, scale=0.16)
    parts.append(f'<path class="overview-arrow" d="M {x + matrix_region_w + 5:g} {y + height / 2:g} L {plot_x - 6:g} {y + height / 2:g}"/>')
    parts.append(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="1.4"/>')
    parts.append(f'<path class="mini-thin" d="M {plot_x:g} {plot_y + plot_h:g} L {plot_x + plot_w:g} {plot_y + plot_h:g}"/>')
    return "\n".join(parts)


def draw_covariance_matrix(x: float, y: float, width: float, height: float, *, scale: float) -> list[str]:
    cell = min(width * scale, height * scale)
    matrix_w = 4 * cell
    matrix_h = 4 * cell
    matrix_x = x + (width - matrix_w) / 2
    matrix_y = y + (height - matrix_h) / 2
    parts = []
    for i in range(4):
        for j in range(4):
            fill = BLUE if abs(i - j) <= 1 else "#b7cee6"
            parts.append(f'<rect x="{matrix_x + j * cell:g}" y="{matrix_y + i * cell:g}" width="{cell:g}" height="{cell:g}" fill="{fill}" stroke="white" stroke-width="0.25"/>')
    return parts


def draw_genotype_table(x: float, y: float, width: float, height: float) -> list[str]:
    row_h = height / 4
    col_w = width / 3
    values = ((0, 1, 2), (1, 0, 1), (2, 1, 0))
    fills = ("#ffffff", "#b7cee6", BLUE)
    parts = [
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" fill="{LIGHT_BLUE}" stroke="#222222" stroke-width="0.6"/>',
    ]
    for row in range(1, 4):
        parts.append(f'<path class="mini-thin" d="M {x:g} {y + row * row_h:g} L {x + width:g} {y + row * row_h:g}"/>')
    for col in range(1, 3):
        parts.append(f'<path class="mini-thin" d="M {x + col * col_w:g} {y:g} L {x + col * col_w:g} {y + height:g}"/>')
    for row, row_values in enumerate(values):
        for col, value in enumerate(row_values):
            marker_w = col_w * 0.48
            marker_h = row_h * 0.38
            marker_x = x + (col + 0.5) * col_w - marker_w / 2
            marker_y = y + (row + 1.5) * row_h - marker_h / 2
            parts.append(f'<rect x="{marker_x:g}" y="{marker_y:g}" width="{marker_w:g}" height="{marker_h:g}" fill="{fills[value]}" stroke="#222222" stroke-width="0.25"/>')
    return parts


def mini_breakpoints(x: float, y: float, width: float, height: float) -> str:
    plot_x = x
    plot_y = y + 4
    plot_w = width
    plot_h = height - 8
    pts = [(0, 0.30), (0.16, 0.66), (0.31, 0.28), (0.50, 0.70), (0.69, 0.34), (0.86, 0.64), (1.0, 0.32)]
    d = " ".join(("M" if i == 0 else "L") + f" {plot_x + plot_w * px:g} {plot_y + plot_h * py:g}" for i, (px, py) in enumerate(pts))
    parts = [f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="1.2"/>']
    for px, py in (pts[1], pts[3], pts[5]):
        parts.append(f'<circle cx="{plot_x + plot_w * px:g}" cy="{plot_y + plot_h * py:g}" r="2.2" fill="{LIGHT_RED}" stroke="{RED}" stroke-width="1"/>')
    return "\n".join(parts)


def mini_bed(x: float, y: float, width: float, height: float) -> str:
    y_line = y + height * 0.35
    y_blocks = y + height * 0.58
    parts = [f'<path class="mini-line" d="M {x:g} {y_line:g} L {x + width:g} {y_line:g}"/>']
    breakpoints = (0.30, 0.58, 0.78)
    edges = (0.0, *breakpoints, 1.0)
    fills = (LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GREEN, LIGHT_RED)
    for fraction in breakpoints:
        bp_x = x + width * fraction
        parts.append(f'<path class="mini-red" d="M {bp_x:g} {y_line - 8:g} L {bp_x:g} {y_blocks + 7:g}"/>')
    for start, end, fill in zip(edges[:-1], edges[1:], fills, strict=True):
        parts.append(f'<rect x="{x + width * start:g}" y="{y_blocks:g}" width="{width * (end - start):g}" height="6" fill="{fill}" stroke="#222222" stroke-width="0.5"/>')
    return "\n".join(parts)


def write_pdf(svg_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
    except FileNotFoundError:
        print("Skipped PDF: rsvg-convert is not installed")
        return
    print(f"Wrote {pdf_path}")


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    main()
