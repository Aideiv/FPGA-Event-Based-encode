#!/usr/bin/env python3
"""
Nature-Journal-Style Interactive Figure Generator
==================================================
Produces self-contained HTML files with:
  - SVG vector graphics (resolution-independent)
  - Hover tooltips with data annotations
  - Zoom/pan via mouse wheel + drag
  - Responsive layout (mobile + desktop)
  - Nature-inspired typography (Harding, sans-serif)
  - Embedded CSS + JS (zero dependencies)

Usage:
    python figures/generate_figures.py

Output:
    figures/fig1_architecture.html    — System architecture
    figures/fig2_normal_flow.html     — VecKM normal flow
    figures/fig3_evasion_levels.html  — Graded evasion response
    figures/fig4_fpga_pipeline.html   — FPGA pipeline timing
    figures/fig5_ttc_prediction.html  — TTC estimation
"""

import json
import math
import os
import textwrap
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
# SHARED CSS + JS FRAMEWORK (Nature journal styling)
# ═══════════════════════════════════════════════════════════════════════════

NATURE_CSS = textwrap.dedent("""\
    @import url('https://fonts.googleapis.com/css2?family=Hind:wght@300;400;500;600;700&display=swap');

    :root {
        --nature-blue: #1a56db;
        --nature-dark: #222222;
        --nature-gray: #6c757d;
        --nature-light: #e9ecef;
        --nature-white: #ffffff;
        --nature-red: #d94841;
        --nature-orange: #e67e22;
        --nature-green: #27ae60;
        --nature-purple: #8854d0;
        --nature-teal: #0d9488;
        --figure-bg: #fafbfc;
        --border-color: #dee2e6;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
        --radius: 6px;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
        font-family: 'Hind', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #f5f6f8;
        color: var(--nature-dark);
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
    }

    .figure-container {
        max-width: 1100px;
        margin: 2rem auto;
        background: var(--nature-white);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        box-shadow: var(--shadow-md);
        overflow: hidden;
    }

    .figure-header {
        padding: 1.5rem 2rem 0.75rem;
        border-bottom: 1px solid var(--nature-light);
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    .figure-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--nature-blue);
        background: #eef2ff;
        padding: 0.2em 0.6em;
        border-radius: 3px;
    }

    .figure-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--nature-dark);
        letter-spacing: -0.01em;
    }

    .figure-caption {
        padding: 0.75rem 2rem;
        font-size: 0.9rem;
        color: var(--nature-gray);
        line-height: 1.55;
    }

    /* Toolbar */
    .toolbar {
        display: flex;
        gap: 0.25rem;
        padding: 0.5rem 2rem;
        background: var(--figure-bg);
        border-bottom: 1px solid var(--nature-light);
        align-items: center;
        flex-wrap: wrap;
    }

    .toolbar button {
        font-family: inherit;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.35em 0.75em;
        border: 1px solid var(--border-color);
        border-radius: 4px;
        background: var(--nature-white);
        color: var(--nature-dark);
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        gap: 0.3em;
    }

    .toolbar button:hover {
        background: #eef2ff;
        border-color: var(--nature-blue);
        color: var(--nature-blue);
    }

    .toolbar button.active {
        background: var(--nature-blue);
        color: white;
        border-color: var(--nature-blue);
    }

    .toolbar .spacer { flex: 1; }

    .toolbar .zoom-display {
        font-size: 0.8rem;
        color: var(--nature-gray);
        font-variant-numeric: tabular-nums;
        min-width: 3.5em;
        text-align: center;
    }

    /* SVG Viewport */
    .svg-viewport {
        position: relative;
        overflow: hidden;
        cursor: grab;
        background: var(--figure-bg);
        min-height: 400px;
        user-select: none;
        -webkit-user-select: none;
    }

    .svg-viewport:active { cursor: grabbing; }
    .svg-viewport.panning { cursor: grabbing; }

    .svg-viewport svg {
        display: block;
        width: 100%;
        height: auto;
    }

    .svg-viewport svg .clickable { cursor: pointer; }
    .svg-viewport svg .clickable:hover { opacity: 0.85; }

    /* Tooltip */
    .figure-tooltip {
        position: absolute;
        pointer-events: none;
        background: rgba(34, 34, 34, 0.94);
        color: #fff;
        padding: 0.5em 0.75em;
        border-radius: 5px;
        font-size: 0.8rem;
        line-height: 1.45;
        max-width: 300px;
        box-shadow: var(--shadow-lg);
        opacity: 0;
        transform: translateY(4px);
        transition: opacity 0.15s ease, transform 0.15s ease;
        z-index: 100;
        backdrop-filter: blur(6px);
    }

    .figure-tooltip.visible {
        opacity: 1;
        transform: translateY(0);
    }

    .figure-tooltip .tooltip-title {
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.2em;
    }

    .figure-tooltip .tooltip-detail {
        color: #adb5bd;
        font-size: 0.75rem;
    }

    /* Legend */
    .legend-bar {
        display: flex;
        gap: 1.25rem;
        padding: 0.75rem 2rem;
        flex-wrap: wrap;
        border-top: 1px solid var(--nature-light);
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.4em;
        font-size: 0.8rem;
        color: var(--nature-dark);
    }

    .legend-swatch {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        flex-shrink: 0;
    }

    .legend-line {
        width: 18px;
        height: 2px;
        flex-shrink: 0;
    }

    /* Footer */
    .figure-footer {
        padding: 0.5rem 2rem 1.25rem;
        font-size: 0.75rem;
        color: var(--nature-gray);
    }

    /* Responsive */
    @media (max-width: 768px) {
        .figure-container { margin: 1rem; border-radius: 4px; }
        .figure-header, .figure-caption, .toolbar, .legend-bar, .figure-footer {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .figure-title { font-size: 1rem; }
    }

    @media (max-width: 480px) {
        .figure-container { margin: 0.5rem; }
        .toolbar button { font-size: 0.7rem; padding: 0.25em 0.5em; }
    }
""")

NATURE_JS = textwrap.dedent("""\
    (function() {
        const viewport = document.querySelector('.svg-viewport');
        const svg = viewport.querySelector('svg');
        const tooltip = document.querySelector('.figure-tooltip');
        const zoomDisplay = document.querySelector('.zoom-display');

        let scale = 1.0;
        let panX = 0, panY = 0;
        let isPanning = false;
        let startX, startY;
        let startPanX, startPanY;
        const MIN_SCALE = 0.3;
        const MAX_SCALE = 5.0;

        function updateTransform() {
            svg.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
            svg.style.transformOrigin = '0 0';
            zoomDisplay.textContent = Math.round(scale * 100) + '%';
        }

        // Zoom with mouse wheel
        viewport.addEventListener('wheel', function(e) {
            e.preventDefault();
            const rect = viewport.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const factor = e.deltaY < 0 ? 1.12 : 0.89;
            const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));

            panX = mx - (mx - panX) * (newScale / scale);
            panY = my - (my - panY) * (newScale / scale);
            scale = newScale;
            updateTransform();
        }, { passive: false });

        // Pan with mouse drag
        viewport.addEventListener('mousedown', function(e) {
            if (e.target.closest('.clickable') && !e.shiftKey) return;
            isPanning = true;
            viewport.classList.add('panning');
            startX = e.clientX;
            startY = e.clientY;
            startPanX = panX;
            startPanY = panY;
        });

        window.addEventListener('mousemove', function(e) {
            if (!isPanning) return;
            panX = startPanX + (e.clientX - startX);
            panY = startPanY + (e.clientY - startY);
            updateTransform();
        });

        window.addEventListener('mouseup', function() {
            isPanning = false;
            viewport.classList.remove('panning');
        });

        // Toolbar buttons
        document.getElementById('btn-zoom-in').addEventListener('click', function() {
            scale = Math.min(MAX_SCALE, scale * 1.25);
            updateTransform();
        });

        document.getElementById('btn-zoom-out').addEventListener('click', function() {
            scale = Math.max(MIN_SCALE, scale * 0.8);
            updateTransform();
        });

        document.getElementById('btn-reset').addEventListener('click', function() {
            scale = 1.0;
            panX = 0;
            panY = 0;
            updateTransform();
        });

        // Tooltip handling
        const tooltipElements = svg.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(function(el) {
            el.addEventListener('mouseenter', function(e) {
                const data = JSON.parse(el.getAttribute('data-tooltip'));
                tooltip.innerHTML = '<div class="tooltip-title">' + data.title + '</div>' +
                    (data.detail ? '<div class="tooltip-detail">' + data.detail + '</div>' : '');
                tooltip.classList.add('visible');
            });
            el.addEventListener('mousemove', function(e) {
                const rect = viewport.getBoundingClientRect();
                let tx = e.clientX - rect.left + 16;
                let ty = e.clientY - rect.top - 10;
                if (tx + 310 > rect.width) tx = e.clientX - rect.left - 310;
                if (ty < 10) ty = 20;
                tooltip.style.left = tx + 'px';
                tooltip.style.top = ty + 'px';
            });
            el.addEventListener('mouseleave', function() {
                tooltip.classList.remove('visible');
            });
        });

        // Touch support
        let lastTouchDist = 0;
        viewport.addEventListener('touchstart', function(e) {
            if (e.touches.length === 1) {
                isPanning = true;
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                startPanX = panX;
                startPanY = panY;
            } else if (e.touches.length === 2) {
                isPanning = false;
                lastTouchDist = Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY
                );
            }
        }, { passive: false });

        viewport.addEventListener('touchmove', function(e) {
            e.preventDefault();
            if (e.touches.length === 1 && isPanning) {
                panX = startPanX + (e.touches[0].clientX - startX);
                panY = startPanY + (e.touches[0].clientY - startY);
                updateTransform();
            } else if (e.touches.length === 2) {
                const dist = Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY
                );
                const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * (dist / lastTouchDist)));
                const rect = viewport.getBoundingClientRect();
                const mx = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
                const my = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top;
                panX = mx - (mx - panX) * (newScale / scale);
                panY = my - (my - panY) * (newScale / scale);
                scale = newScale;
                lastTouchDist = dist;
                updateTransform();
            }
        }, { passive: false });

        viewport.addEventListener('touchend', function() { isPanning = false; });
    })();
""")


def wrap_html(title, label, caption, legend_items, svg_content, extra_head=""):
    """Wrap SVG content in a full Nature-style interactive HTML page."""
    legend_html = ""
    if legend_items:
        items = []
        for item in legend_items:
            if item.get("type") == "swatch":
                items.append(
                    f'<div class="legend-item">'
                    f'<span class="legend-swatch" style="background:{item["color"]}"></span>'
                    f'{item["label"]}</div>'
                )
            elif item.get("type") == "line":
                items.append(
                    f'<div class="legend-item">'
                    f'<span class="legend-line" style="background:{item["color"]}"></span>'
                    f'{item["label"]}</div>'
                )
        legend_html = '<div class="legend-bar">' + "".join(items) + "</div>"

    return textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{NATURE_CSS}</style>
    {extra_head}
    </head>
    <body>
    <div class="figure-container">
        <div class="figure-header">
            <span class="figure-label">{label}</span>
            <span class="figure-title">{title}</span>
        </div>
        <div class="figure-caption">{caption}</div>
        <div class="toolbar">
            <button id="btn-zoom-in" title="Zoom In">🔍⁺ Zoom In</button>
            <button id="btn-zoom-out" title="Zoom Out">🔍⁻ Zoom Out</button>
            <button id="btn-reset" title="Reset View">↺ Reset</button>
            <span class="spacer"></span>
            <span class="zoom-display">100%</span>
        </div>
        <div class="svg-viewport">
            {svg_content}
            <div class="figure-tooltip"></div>
        </div>
        {legend_html}
        <div class="figure-footer">
            Interactive figure — Scroll to zoom, drag to pan, hover for details.
        </div>
    </div>
    <script>{NATURE_JS}</script>
    </body>
    </html>
    """)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: System Architecture
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig1_architecture():
    """Dual-pipeline system architecture with FPGA + ARM."""
    W, H = 1100, 700

    def ttip(title, detail=""):
        return f'data-tooltip=\'{json.dumps({"title": title, "detail": detail})}\''

    # Color palette
    C_EVENT = "#7c3aed"
    C_FPGA = "#2563eb"
    C_ARM = "#059669"
    C_MOTOR = "#d94841"
    C_FLOW = "#e67e22"
    C_CNN = "#0d9488"
    C_AUX = "#6c757d"
    C_BG_BOX = "#f8f9ff"
    C_BORDER = "#c4c9d4"

    # Helper: rounded rectangle with tooltip
    def box(x, y, w, h, label, color, tooltip_title, tooltip_detail="", rx=8, cls=""):
        extra = " class=\"clickable {}\"".format(cls) if cls else " class=\"clickable\""
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="1.5"{extra} '
            f'{ttip(tooltip_title, tooltip_detail)}/>\n'
            f'<text x="{x + w/2}" y="{y + h/2 + 5}" text-anchor="middle" '
            f'font-family="Hind, sans-serif" font-size="12" font-weight="600" '
            f'fill="{color}"{extra} '
            f'{ttip(tooltip_title, tooltip_detail)}>{label}</text>\n'
        )

    def arrow(x1, y1, x2, y2, color="#6c757d", label=""):
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        # Shaft
        result = (
            f'<line x1="{x1 + ux*14}" y1="{y1 + uy*14}" x2="{x2 - ux*14}" y2="{y2 - uy*14}" '
            f'stroke="{color}" stroke-width="2" marker-end="url(#arrowhead-{color.replace("#","")})"/>'
        )
        if label:
            angle = math.degrees(math.atan2(dy, dx))
            result += (
                f'\n<text x="{mid_x}" y="{mid_y - 8}" text-anchor="middle" '
                f'font-family="Hind, sans-serif" font-size="10" fill="{color}" '
                f'transform="rotate({angle}, {mid_x}, {mid_y})">{label}</text>'
            )
        return result

    def section_box(x, y, w, h, label, color, opacity=0.04):
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
            f'fill="{color}" fill-opacity="{opacity}" stroke="{color}" stroke-width="1" '
            f'stroke-dasharray="6,3" stroke-opacity="0.4"/>\n'
            f'<text x="{x + 12}" y="{y + 18}" font-family="Hind, sans-serif" '
            f'font-size="11" font-weight="700" fill="{color}" letter-spacing="0.05em">{label}</text>'
        )

    arrowheads = ""
    for c in [C_EVENT, C_FPGA, C_ARM, C_MOTOR, C_FLOW, C_CNN, C_AUX]:
        ch = c.replace("#", "")
        arrowheads += (
            f'<marker id="arrowhead-{ch}" markerWidth="10" markerHeight="7" '
            f'refX="9" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 10 3.5, 0 7" fill="{c}"/>'
            f'</marker>\n'
        )

    svg_parts = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family: Hind, sans-serif;">',
        '<defs>', arrowheads, '</defs>',
        # Background
        f'<rect width="{W}" height="{H}" fill="#fafbfc" rx="6"/>',

        # Section boxes
        section_box(20, 20, 1060, 660, "DRONE MAIN LOOP", "#6c757d"),

        # Column 1: Event Camera
        box(35, 70, 140, 120, "Event Camera\n(AER)", C_EVENT,
            "Event Camera (Address-Event Representation)",
            "Prophesee Gen4 / Inivation DVXplorer\nAsynchronous per-pixel brightness changes\nMicrosecond resolution, 10M events/sec"),

        # Arrow: Camera → FPGA
        arrow(175, 130, 240, 130, C_EVENT, "AER Bus"),

        # Column 2: FPGA Fabric
        section_box(240, 50, 280, 600, "FPGA FABRIC", C_FPGA, 0.05),
        box(260, 85, 235, 36, "AER Interface", C_FPGA,
            "AER Interface", "Parallel bus receiver, event decoding, timestamp tagging\n10M events/sec sustained throughput"),
        box(260, 133, 235, 36, "Ring Buffer", C_FPGA,
            "Ring Buffer (4096 events)", "Sliding window of recent events\nConfigurable depth, zero-copy reads"),
        box(260, 181, 235, 36, "Spatial Hash k-NN", C_FPGA,
            "Spatial Hash k-NN", "Grid-based O(k) neighbor search\nk=32 neighbors, grid radius=0.15\nReplaces O(n²) brute force"),
        box(260, 229, 235, 36, "Normalization", C_FPGA,
            "Coordinate Normalization", "Pixel → normalized coords\nSpatial: × pxl_radius (0.0225)\nTemporal: × t_radius (0.01)"),
        box(260, 277, 235, 36, "Systolic Encoder", C_FPGA,
            "Systolic Encoder Array", "VecKM encoder, d=128\n3-layer MLP, INT8 quantization\nPipelined systolic dataflow"),
        box(260, 325, 235, 36, "Ensemble Logic", C_FPGA,
            "Ensemble Aggregation", "Mean + variance across ensemble members\nUncertainty estimation"),
        box(260, 373, 235, 36, "PWM Output", C_FPGA,
            "PWM Motor Output", "50Hz ESC protocol\n4-channel PWM generation"),

        # Arrow: FPGA → ARM
        arrow(520, 200, 590, 200, C_ARM, "Flow\nData"),

        # Column 3: ARM CPU
        section_box(590, 50, 240, 600, "ARM CPU", C_ARM, 0.05),
        box(610, 85, 200, 36, "Collision Predictor", C_ARM,
            "Collision Predictor (TTC)", "Looming-based TTC via Lee's τ\nKalman filter object tracking\nSafe bearing computation"),
        box(610, 133, 200, 36, "Evasion Controller", C_ARM,
            "Evasion Controller", "5-level graded response w/ hysteresis\nPotential-field vector computation\nVelocity command generation"),

        # Pipeline 1 (Flow-Based)
        section_box(610, 200, 200, 160, "PIPELINE 1: VecKM Flow", C_FLOW, 0.04),
        box(625, 225, 170, 30, "Object Detector", C_FLOW,
            "VecKM Object Detector", "Normal flow + DBSCAN clustering\nFlow divergence → looming detection\n~100Hz inference rate"),
        box(625, 265, 170, 30, "TTC Estimator", C_FLOW,
            "TTC Estimator", "Lee's τ from flow divergence\nDense per-event TTC option"),
        box(625, 305, 170, 30, "Evasion Planner", C_FLOW,
            "Evasion Planner", "Graded velocity output\nLateral + vertical + yaw"),

        # Pipeline 2 (CNN)
        section_box(610, 380, 200, 160, "PIPELINE 2: Direct CNN", C_CNN, 0.04),
        box(625, 405, 170, 30, "Event Frame Agg.", C_CNN,
            "Event Frame Aggregator", "80×80 @ 1kHz frame rate\n1ms integration windows\n5-frame temporal stack"),
        box(625, 445, 170, 30, "DPU CNN", C_CNN,
            "DPU-Optimized CNN", "5-class evasion (STAY/L/R/U/D)\n4 conv layers [32,64,128,128]\nDirect action prediction"),
        box(625, 485, 170, 30, "Action Smoother", C_CNN,
            "Action Probability Smoother", "EMA smoothing α=0.3\nConfidence threshold 0.4"),

        # Auxiliary modules
        section_box(610, 555, 200, 90, "AUXILIARY", C_AUX, 0.04),
        box(625, 573, 80, 28, "Contrast Max", C_AUX,
            "Contrast Maximizer", "IWE-based flow refinement\nGallego et al. CVPR 2018"),
        box(712, 573, 80, 28, "Depth Est.", C_AUX,
            "Depth Estimator", "E2Depth monocular U-Net\nFrom event frames, 3DV 2020"),

        # Arrow: ARM → Motors
        arrow(830, 200, 910, 200, C_MOTOR, "Velocity\nCommands"),

        # Column 4: Motors
        box(910, 70, 140, 120, "Drone\nMotors", C_MOTOR,
            "Quadrotor Motors", "4 × brushless DC motors\n50Hz PWM ESC protocol\nGraded velocity flight"),

        # Safety watchdog callout
        box(410, 610, 200, 32, "Safety Watchdog", "#d94841",
            "Safety Watchdog", "RC failsafe (500ms) • Alt ceiling (120m)\nNaN guard • Motor timeout (30s)\nVelocity smoothing (EMA α=0.3)"),

        # Pipeline label arrows
        arrow(845, 280, 910, 280, C_FLOW, "~100Hz"),
        arrow(845, 460, 910, 460, C_CNN, "~1kHz"),

        # Clock annotation
        f'<text x="980" y="660" text-anchor="end" font-size="10" fill="#6c757d">'
        f'100MHz system clock | Inference period: 100k cycles (1ms)</text>',
        '</svg>'
    ]

    legend = [
        {"type": "swatch", "color": C_EVENT, "label": "Event Camera"},
        {"type": "swatch", "color": C_FPGA, "label": "FPGA Fabric"},
        {"type": "swatch", "color": C_ARM, "label": "ARM CPU"},
        {"type": "swatch", "color": C_FLOW, "label": "Pipeline 1: VecKM Flow (~100Hz)"},
        {"type": "swatch", "color": C_CNN, "label": "Pipeline 2: Direct CNN (~1kHz)"},
        {"type": "swatch", "color": C_MOTOR, "label": "Motor Output"},
    ]

    html = wrap_html(
        title="System Architecture: Dual-Pipeline Collision Avoidance",
        label="Fig. 1",
        caption=(
            "Complete drone collision avoidance architecture. Event camera data streams into FPGA fabric "
            "for real-time normal flow estimation (VecKM, d=128). The ARM CPU runs two parallel pipelines: "
            "(1) VecKM flow-based object detection and TTC estimation at ~100Hz, and (2) a direct CNN "
            "action predictor at ~1kHz. Outputs are fused into graded evasion velocity commands for the "
            "quadrotor motors."
        ),
        legend_items=legend,
        svg_content="\n".join(svg_parts)
    )
    return html


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: VecKM Normal Flow Estimation
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig2_normal_flow():
    """VecKM normal flow — geometric illustration of kernel mixture."""
    W, H = 1100, 650

    def ttip(title, detail=""):
        return f'data-tooltip=\'{json.dumps({"title": title, "detail": detail})}\''

    C_EVENT = "#7c3aed"
    C_NEIGHBOR = "#2563eb"
    C_FLOW = "#e67e22"
    C_KERNEL = "#059669"
    C_ENC = "#d94841"

    svg_parts = []

    # SVG open + defs
    svg_parts.append(
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family: Hind, sans-serif;">'
    )
    svg_parts.append(
        '<defs>'
        '<marker id="arr-gray" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#6c757d"/>'
        '</marker>'
        '</defs>'
    )
    svg_parts.append(f'<rect width="{W}" height="{H}" fill="#fafbfc" rx="6"/>')

    # Title
    svg_parts.append(
        f'<text x="{W/2}" y="35" text-anchor="middle" font-size="18" font-weight="700" fill="#222">'
        f'VecKM: Vectorized Kernel Mixture for Normal Flow</text>'
    )

    # ============ STEP 1: Event Neighborhood ============
    svg_parts.append(
        f'<text x="95" y="80" text-anchor="middle" font-size="13" font-weight="600" fill="#6c757d">'
        f'1. Event Neighborhood</text>'
    )

    # Sensor plane
    svg_parts.append(
        f'<rect x="30" y="95" width="130" height="130" rx="4" fill="white" stroke="#dee2e6" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="95" y="112" text-anchor="middle" font-size="10" fill="#adb5bd">Sensor Plane (640×480)</text>'
    )

    # Scatter events
    events = [(55,130),(80,118),(62,150),(95,140),(58,170),(105,155),(70,185),(88,175),(50,160),(110,130),
              (75,145),(100,170),(55,195),(120,150),(65,135),(90,190),(115,170),(45,175),(130,160),(85,200)]
    for ex, ey in events:
        svg_parts.append(f'<circle cx="{ex}" cy="{ey}" r="2.5" fill="{C_EVENT}" opacity="0.6"/>')

    # Query event (highlighted)
    qx, qy = 85, 160
    svg_parts.append(f'<circle cx="{qx}" cy="{qy}" r="5" fill="{C_FLOW}" stroke="white" stroke-width="2"/>')
    svg_parts.append(
        f'<text x="{qx+12}" y="{qy+4}" font-size="10" fill="{C_FLOW}" font-weight="600">'
        f'Query event e\u1d62</text>'
    )

    # k-NN radius
    svg_parts.append(
        f'<circle cx="{qx}" cy="{qy}" r="35" fill="none" stroke="{C_NEIGHBOR}" '
        f'stroke-width="1.5" stroke-dasharray="4,2" opacity="0.6"/>'
    )
    svg_parts.append(
        f'<text x="{qx+30}" y="{qy-42}" font-size="9" fill="{C_NEIGHBOR}">k=32 neighbors</text>'
    )

    # Arrow to step 2
    svg_parts.append(
        f'<line x1="170" y1="160" x2="225" y2="160" stroke="#6c757d" stroke-width="1.5" '
        f'marker-end="url(#arr-gray)"/>'
    )

    # ============ STEP 2: Kernel Mixture ============
    svg_parts.append(
        f'<text x="360" y="80" text-anchor="middle" font-size="13" font-weight="600" fill="#6c757d">'
        f'2. Vectorized Kernel Mixture</text>'
    )

    # Show a small grid of basis functions
    for gx_idx in range(5):
        for gy_idx in range(5):
            gx = 280 + gx_idx * 32
            gy = 100 + gy_idx * 32
            alpha = 0.15 + 0.08 * ((gx_idx + gy_idx) % 3)
            svg_parts.append(
                f'<ellipse cx="{gx}" cy="{gy}" rx="14" ry="14" fill="{C_KERNEL}" '
                f'opacity="{alpha:.2f}" stroke="{C_KERNEL}" stroke-width="0.5" stroke-opacity="0.3"/>'
            )

    # Formula annotation
    svg_parts.append(
        f'<text x="360" y="275" text-anchor="middle" font-size="11" fill="#6c757d" '
        f'font-style="italic">\u03ba(x, y) = \u03a3\u2c7c w\u2c7c \u00b7 exp(\u2212\u03b3||(x,y) \u2212 c\u2c7c||\u00b2)</text>'
    )

    # Highlight encoding vector
    svg_parts.append(
        f'<rect x="295" y="290" width="130" height="22" rx="3" fill="{C_ENC}" fill-opacity="0.12" '
        f'stroke="{C_ENC}" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="360" y="305" text-anchor="middle" font-size="10" font-weight="600" fill="{C_ENC}">'
        f'Encoded: d=128 vector</text>'
    )

    # Arrow to step 3
    svg_parts.append(
        f'<line x1="435" y1="160" x2="490" y2="160" stroke="#6c757d" stroke-width="1.5" '
        f'marker-end="url(#arr-gray)"/>'
    )

    # ============ STEP 3: Flow Regression ============
    svg_parts.append(
        f'<text x="625" y="80" text-anchor="middle" font-size="13" font-weight="600" fill="#6c757d">'
        f'3. Normal Flow Regression</text>'
    )

    # MLP network
    layers = [(520, 120, 40, 80, C_ENC), (580, 140, 30, 60, C_ENC), (635, 130, 30, 70, C_FLOW),
              (695, 145, 25, 50, C_FLOW), (750, 155, 50, 30, C_FLOW)]
    for i, (lx, ly, lw, lh, lc) in enumerate(layers):
        svg_parts.append(
            f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="3" fill="{lc}" '
            f'fill-opacity="0.15" stroke="{lc}" stroke-width="1"/>'
        )
        if i < len(layers) - 1:
            nx, ny = layers[i+1][0], layers[i+1][1] + layers[i+1][3]/2
            px, py = lx + lw, ly + lh/2
            svg_parts.append(
                f'<line x1="{px}" y1="{py}" x2="{nx}" y2="{ny}" '
                f'stroke="#adb5bd" stroke-width="0.8"/>'
            )

    svg_parts.append(
        f'<text x="585" y="210" text-anchor="middle" font-size="9" fill="#6c757d">3-Layer MLP</text>'
    )

    # Output
    svg_parts.append(
        f'<text x="800" y="165" font-size="10" font-weight="600" fill="{C_FLOW}">(v\u2093, v\u1d67)</text>'
    )
    svg_parts.append(
        f'<text x="800" y="180" font-size="9" fill="#6c757d">Normal flow</text>'
    )

    # Uncertainty
    svg_parts.append(
        f'<text x="800" y="200" font-size="10" fill="#d94841">\u03c3\u00b2</text>'
    )
    svg_parts.append(
        f'<text x="800" y="215" font-size="9" fill="#6c757d">Uncertainty</text>'
    )

    # ============ FLOW FIELD VISUALIZATION (bottom half) ============
    svg_parts.append(
        f'<text x="{W/2}" y="310" text-anchor="middle" font-size="14" font-weight="600" fill="#222">'
        f'Normal Flow Field (Simulated)</text>'
    )

    # Simulated diverging flow (looming object)
    center_x, center_y = 550, 460
    for angle_deg in range(0, 360, 15):
        angle = math.radians(angle_deg)
        dist = 30
        sx = center_x + math.cos(angle) * dist
        sy = center_y + math.sin(angle) * dist
        flow_len = 8 + 20 * (dist / 120)
        ex = sx + math.cos(angle) * flow_len
        ey = sy + math.sin(angle) * flow_len
        alpha = 0.3 + 0.5 * (dist / 120)
        svg_parts.append(
            f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="{C_FLOW}" '
            f'stroke-width="1.8" opacity="{alpha:.2f}"/>'
        )
        # Arrowhead
        ah_len = 3
        ax1 = ex - ah_len * math.cos(angle + 2.5 - 0.6)
        ay1 = ey - ah_len * math.sin(angle + 2.5 - 0.6)
        ax2 = ex - ah_len * math.cos(angle + 2.5 + 0.6)
        ay2 = ey - ah_len * math.sin(angle + 2.5 + 0.6)
        svg_parts.append(
            f'<polygon points="{ex},{ey} {ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f}" '
            f'fill="{C_FLOW}" opacity="{alpha:.2f}"/>'
        )

    # Surrounding lateral flow
    for lx in [200, 300, 400, 500, 600, 700, 800, 900]:
        for ly in [350, 390, 430, 470, 510, 550]:
            dx = (lx - center_x)
            dy = (ly - center_y)
            dist_c = math.hypot(dx, dy)
            if dist_c < 50:
                continue
            fx = dx / dist_c * 3 * (1 if dist_c < 180 else -0.3)
            fy = dy / dist_c * 3 * (1 if dist_c < 180 else -0.3)
            svg_parts.append(
                f'<line x1="{lx}" y1="{ly}" x2="{lx+fx}" y2="{ly+fy}" '
                f'stroke="{C_NEIGHBOR}" stroke-width="1" opacity="0.25"/>'
            )

    # Looming object circle
    svg_parts.append(
        f'<circle cx="{center_x}" cy="{center_y}" r="45" fill="none" stroke="#d94841" '
        f'stroke-width="2" stroke-dasharray="8,3" opacity="0.7"/>'
    )
    svg_parts.append(
        f'<text x="{center_x}" y="{center_y-52}" text-anchor="middle" font-size="10" '
        f'fill="#d94841" font-weight="600">Looming Object</text>'
    )

    # Collision cone annotation
    svg_parts.append(
        f'<line x1="{center_x}" y1="{center_y}" x2="{center_x-32}" y2="580" '
        f'stroke="#d94841" stroke-width="1" stroke-dasharray="3,3" opacity="0.4"/>'
    )
    svg_parts.append(
        f'<line x1="{center_x}" y1="{center_y}" x2="{center_x+32}" y2="580" '
        f'stroke="#d94841" stroke-width="1" stroke-dasharray="3,3" opacity="0.4"/>'
    )
    svg_parts.append(
        f'<text x="{center_x}" y="598" text-anchor="middle" font-size="10" fill="#d94841">'
        f'Collision Cone (flow divergence)</text>'
    )

    # Drone position
    svg_parts.append(
        f'<rect x="{center_x-8}" y="595" width="16" height="10" rx="2" fill="#6c757d"/>'
    )
    svg_parts.append(
        f'<text x="{center_x}" y="620" text-anchor="middle" font-size="9" fill="#6c757d">Drone</text>'
    )

    # Performance metrics box
    metrics_x, metrics_y = 30, 350
    svg_parts.append(
        f'<rect x="{metrics_x}" y="{metrics_y}" width="200" height="145" rx="6" '
        f'fill="white" stroke="#dee2e6" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="{metrics_x+100}" y="{metrics_y+22}" text-anchor="middle" font-size="11" '
        f'font-weight="600" fill="#222">Performance Metrics</text>'
    )
    metrics = [
        ("Inference Rate", "100 Hz (FPGA)"),
        ("Latency", "1.0 ms per frame"),
        ("Encoder Dim", "d = 128 (FPGA), 384 (GPU)"),
        ("Neighbors (k)", "32"),
        ("Ensemble Size", "10 members"),
        ("Uncertainty Threshold", "\u03c3\u00b2 > 0.3 \u2192 masked"),
    ]
    for i, (key, val) in enumerate(metrics):
        my = metrics_y + 44 + i * 21
        svg_parts.append(f'<text x="{metrics_x+12}" y="{my}" font-size="10" fill="#6c757d">{key}</text>')
        svg_parts.append(
            f'<text x="{metrics_x+190}" y="{my}" text-anchor="end" font-size="10" '
            f'font-weight="500" fill="#222">{val}</text>'
        )

    # Close SVG
    svg_parts.append('</svg>')

    legend = [
        {"type": "swatch", "color": C_EVENT, "label": "Event data (AER)"},
        {"type": "swatch", "color": C_NEIGHBOR, "label": "Spatial neighborhood (k-NN)"},
        {"type": "swatch", "color": C_KERNEL, "label": "Kernel mixture basis"},
        {"type": "swatch", "color": C_FLOW, "label": "Predicted normal flow"},
        {"type": "line", "color": C_FLOW, "label": "Diverging flow (looming)"},
    ]

    html = wrap_html(
        title="VecKM Normal Flow: From Event Neighborhoods to Flow Vectors",
        label="Fig. 2",
        caption=(
            "The VecKM pipeline transforms raw event data into dense normal flow fields. "
            "(1) Each event's k=32 nearest spatiotemporal neighbors are retrieved via spatial hash. "
            "(2) A vectorized kernel mixture encodes local geometry into a d=128 descriptor. "
            "(3) A 3-layer MLP regresses (vx, vy) flow vectors with per-prediction uncertainty sigma-squared. "
            "Diverging flow patterns (red cone) indicate looming objects for collision prediction."
        ),
        legend_items=legend,
        svg_content="\n".join(svg_parts)
    )
    return html


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Evasion Levels
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig3_evasion_levels():
    """Five graded evasion levels with hysteresis."""
    W, H = 1100, 620

    def ttip(title, detail=""):
        return f'data-tooltip=\'{json.dumps({"title": title, "detail": detail})}\''

    levels = [
        {"name": "NONE", "danger": "< 0.15", "speed": "100%", "behavior": "Normal cruise at set speed",
         "color": "#27ae60", "y": 55, "h": 68},
        {"name": "CAUTION", "danger": "≥ 0.15", "speed": "80%", "behavior": "20% speed reduction, gentle lateral nudge",
         "color": "#f1c40f", "y": 140, "h": 68},
        {"name": "WARNING", "danger": "≥ 0.35", "speed": "50%", "behavior": "50% speed reduction, strong lateral + vertical",
         "color": "#e67e22", "y": 225, "h": 68},
        {"name": "CRITICAL", "danger": "≥ 0.60", "speed": "0% forward", "behavior": "Full stop forward, aggressive dodge",
         "color": "#e74c3c", "y": 310, "h": 68},
        {"name": "EMERGENCY", "danger": "≥ 0.85", "speed": "−30% reverse", "behavior": "Reverse thrust + maximum dodge + hard yaw",
         "color": "#8e0000", "y": 395, "h": 68},
    ]

    svg_parts = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family: Hind, sans-serif;">',
        f'<rect width="{W}" height="{H}" fill="#fafbfc" rx="6"/>',
    ]

    # Danger continuum bar (top)
    bar_x, bar_w = 50, 380
    bar_y = 30
    svg_parts.append(
        f'<linearGradient id="dangerGrad" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="#27ae60"/>'
        f'<stop offset="15%" stop-color="#f1c40f"/>'
        f'<stop offset="35%" stop-color="#e67e22"/>'
        f'<stop offset="60%" stop-color="#e74c3c"/>'
        f'<stop offset="85%" stop-color="#8e0000"/>'
        f'<stop offset="100%" stop-color="#4a0000"/>'
        f'</linearGradient>'
    )
    svg_parts.append(
        f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="16" rx="8" fill="url(#dangerGrad)"/>'
    )
    svg_parts.append(
        f'<text x="{bar_x}" y="{bar_y+30}" text-anchor="middle" font-size="10" fill="#6c757d">'
        f'0.0</text>'
    )
    svg_parts.append(
        f'<text x="{bar_x + bar_w}" y="{bar_y+30}" text-anchor="end" font-size="10" fill="#6c757d">'
        f'1.0</text>'
    )
    for thresh, lbl in [(0.15, "0.15"), (0.35, "0.35"), (0.60, "0.60"), (0.85, "0.85")]:
        tx = bar_x + bar_w * thresh
        svg_parts.append(f'<line x1="{tx}" y1="{bar_y-2}" x2="{tx}" y2="{bar_y+18}" '
                         f'stroke="#222" stroke-width="1"/>')
        svg_parts.append(f'<text x="{tx}" y="{bar_y+30}" text-anchor="middle" font-size="9" fill="#6c757d">'
                         f'{lbl}</text>')
    svg_parts.append(
        f'<text x="{bar_x + bar_w/2}" y="{bar_y-8}" text-anchor="middle" font-size="11" '
        f'font-weight="600" fill="#222">Danger Level Continuum</text>'
    )

    # Level bars (left side)
    bar_start_x = 50
    bar_end_x = 480
    for lv in levels:
        ly = lv["y"]
        lh = lv["h"]
        # Calculate bar width based on danger threshold
        if lv["name"] == "NONE":
            bw = 0.15 * (bar_end_x - bar_start_x)
        elif lv["name"] == "CAUTION":
            bw = (0.35 - 0.15) * (bar_end_x - bar_start_x)
        elif lv["name"] == "WARNING":
            bw = (0.60 - 0.35) * (bar_end_x - bar_start_x)
        elif lv["name"] == "CRITICAL":
            bw = (0.85 - 0.60) * (bar_end_x - bar_start_x)
        else:
            bw = (1.0 - 0.85) * (bar_end_x - bar_start_x)

        bx = bar_start_x
        if lv["name"] == "CAUTION":
            bx = bar_start_x + 0.15 * (bar_end_x - bar_start_x)
        elif lv["name"] == "WARNING":
            bx = bar_start_x + 0.35 * (bar_end_x - bar_start_x)
        elif lv["name"] == "CRITICAL":
            bx = bar_start_x + 0.60 * (bar_end_x - bar_start_x)
        elif lv["name"] == "EMERGENCY":
            bx = bar_start_x + 0.85 * (bar_end_x - bar_start_x)

        svg_parts.append(
            f'<rect x="{bx}" y="{ly}" width="{bw}" height="{lh}" rx="5" '
            f'fill="{lv["color"]}" fill-opacity="0.15" stroke="{lv["color"]}" stroke-width="2" '
            f'class="clickable" {ttip(lv["name"] + " Level", chr(10).join([lv["behavior"], "Danger: " + lv["danger"], "Speed: " + lv["speed"]]))}/>'
        )
        svg_parts.append(
            f'<text x="{bx + bw/2}" y="{ly + 22}" text-anchor="middle" font-size="14" '
            f'font-weight="700" fill="{lv["color"]}">{lv["name"]}</text>'
        )
        svg_parts.append(
            f'<text x="{bx + bw/2}" y="{ly + 44}" text-anchor="middle" font-size="10" '
            f'fill="{lv["color"]}">Danger {lv["danger"]} | Speed {lv["speed"]}</text>'
        )

    # Behavior descriptions on the right
    desc_x = 520
    for lv in levels:
        ly = lv["y"]
        svg_parts.append(
            f'<text x="{desc_x}" y="{ly + 30}" font-size="12" fill="#222">{lv["behavior"]}</text>'
        )
        # Hysteresis arrows
        if lv["name"] != "NONE":
            svg_parts.append(
                f'<text x="{desc_x + 400}" y="{ly + 30}" font-size="9" fill="#6c757d">'
                f'Hysteresis: {["","3 cycles↑","5 cycles↑","10 cycles↑",""][levels.index(lv)]}'
                f'</text>'
            )

    # Hysteresis diagram
    hyst_x, hyst_y = 50, 490
    svg_parts.append(
        f'<text x="{hyst_x}" y="{hyst_y+15}" font-size="12" font-weight="600" fill="#222">'
        f'Hysteresis State Machine</text>'
    )

    states = [
        ("NONE", 70, 530, "#27ae60"),
        ("CAUTION", 220, 530, "#f1c40f"),
        ("WARNING", 370, 530, "#e67e22"),
        ("CRITICAL", 520, 530, "#e74c3c"),
        ("EMERG.", 670, 530, "#8e0000"),
    ]
    for i, (name, sx, sy, sc) in enumerate(states):
        svg_parts.append(
            f'<rect x="{sx}" y="{sy}" width="80" height="36" rx="18" fill="{sc}" '
            f'fill-opacity="0.2" stroke="{sc}" stroke-width="1.5"/>'
        )
        svg_parts.append(
            f'<text x="{sx+40}" y="{sy+23}" text-anchor="middle" font-size="10" '
            f'font-weight="600" fill="{sc}">{name}</text>'
        )
        # Arrow to next state
        if i < len(states) - 1:
            nx = states[i+1][1]
            svg_parts.append(
                f'<line x1="{sx+80}" y1="{sy+18}" x2="{nx}" y2="{sy+18}" '
                f'stroke="#6c757d" stroke-width="1.2" marker-end="url(#arr-gray2)"/>'
            )
            svg_parts.append(
                f'<text x="{(sx+80+nx)/2}" y="{sy+10}" text-anchor="middle" font-size="8" fill="#6c757d">'
                f'danger↑</text>'
            )
            # Back arrow (hysteresis recovery)
            svg_parts.append(
                f'<line x1="{nx}" y1="{sy+26}" x2="{sx+80}" y2="{sy+26}" '
                f'stroke="#adb5bd" stroke-width="0.8" stroke-dasharray="3,2"/>'
            )
            hyst_labels = ["","3cyc","5cyc","10cyc",""]
            svg_parts.append(
                f'<text x="{(sx+80+nx)/2}" y="{sy+38}" text-anchor="middle" font-size="7" fill="#adb5bd">'
                f'{hyst_labels[i+1]}↓</text>'
            )

    # Hysteresis cycles annotation table
    tbl_x, tbl_y = 800, 490
    svg_parts.append(
        f'<rect x="{tbl_x}" y="{tbl_y}" width="240" height="110" rx="4" '
        f'fill="white" stroke="#dee2e6" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="{tbl_x+120}" y="{tbl_y+20}" text-anchor="middle" font-size="10" '
        f'font-weight="600" fill="#222">Recovery Hysteresis</text>'
    )
    hyst_data = [
        ("CRITICAL → WARNING", "3 control cycles"),
        ("WARNING → CAUTION", "5 control cycles"),
        ("CAUTION → NONE", "10 control cycles"),
    ]
    for i, (transition, cycles) in enumerate(hyst_data):
        ry = tbl_y + 38 + i * 22
        svg_parts.append(
            f'<text x="{tbl_x+12}" y="{ry}" font-size="9" fill="#6c757d">{transition}</text>'
        )
        svg_parts.append(
            f'<text x="{tbl_x+228}" y="{ry}" text-anchor="end" font-size="9" '
            f'font-weight="500" fill="#222">{cycles}</text>'
        )

    svg_parts.insert(1,
        '<defs>'
        '<marker id="arr-gray2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#6c757d"/>'
        '</marker>'
        '</defs>'
    )
    svg_parts.append('</svg>')

    legend = [
        {"type": "swatch", "color": "#27ae60", "label": "NONE — Normal cruise"},
        {"type": "swatch", "color": "#f1c40f", "label": "CAUTION — 20% speed reduction"},
        {"type": "swatch", "color": "#e67e22", "label": "WARNING — 50% speed reduction"},
        {"type": "swatch", "color": "#e74c3c", "label": "CRITICAL — Full stop forward"},
        {"type": "swatch", "color": "#8e0000", "label": "EMERGENCY — Reverse thrust"},
    ]

    html = wrap_html(
        title="Five-Level Graded Evasion Response with Hysteresis",
        label="Fig. 3",
        caption=(
            "Graded collision avoidance response across five danger levels. As danger increases "
            "(measured via TTC and flow divergence), the drone transitions from normal cruise through "
            "progressively more aggressive evasion. Hysteresis prevents oscillation: recovery to lower "
            "levels requires sustained safe conditions (3–10 control cycles at 100Hz)."
        ),
        legend_items=legend,
        svg_content="\n".join(svg_parts)
    )
    return html


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: FPGA Pipeline Timing
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig4_fpga_pipeline():
    """FPGA pipeline stages with timing diagram."""
    W, H = 1100, 600

    def ttip(title, detail=""):
        return f'data-tooltip=\'{json.dumps({"title": title, "detail": detail})}\''

    C_STAGE = "#2563eb"
    C_TIME = "#e67e22"
    C_BUS = "#7c3aed"
    C_DDR = "#059669"

    svg_parts = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family: Hind, sans-serif;">',
        f'<rect width="{W}" height="{H}" fill="#fafbfc" rx="6"/>',
    ]

    # Pipeline stages (top)
    stages = [
        ("AER\nReceive", 0.005, "#7c3aed"),
        ("Ring Buffer\nWrite", 0.003, "#2563eb"),
        ("Spatial\nHash k-NN", 0.045, "#2563eb"),
        ("Coordinate\nNormalize", 0.008, "#2563eb"),
        ("Systolic\nEncoder", 0.120, "#2563eb"),
        ("Ensemble\nAggregate", 0.018, "#2563eb"),
        ("Flow\nOutput", 0.003, "#2563eb"),
        ("ARM\nPredict", 0.095, "#059669"),
    ]

    total_time = sum(s[1] for s in stages)  # ~0.297ms
    # Scale to fit
    scale_factor = 950 / total_time  # px per ms
    stage_x = 55
    stage_y = 70
    stage_h = 80

    for name, duration, color in stages:
        sw = duration * scale_factor
        svg_parts.append(
            f'<rect x="{stage_x}" y="{stage_y}" width="{sw}" height="{stage_h}" rx="6" '
            f'fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="1.5" '
            f'class="clickable" {ttip(name.replace(chr(10)," "), f"Duration: {duration*1000:.1f} μs")}/>'
        )
        # Name inside box
        lines = name.split("\n")
        for li, line in enumerate(lines):
            svg_parts.append(
                f'<text x="{stage_x + sw/2}" y="{stage_y + 30 + li*16}" text-anchor="middle" '
                f'font-size="10" font-weight="600" fill="{color}">{line}</text>'
            )
        # Duration below
        svg_parts.append(
            f'<text x="{stage_x + sw/2}" y="{stage_y + stage_h + 16}" text-anchor="middle" '
            f'font-size="9" fill="#6c757d">{duration*1000:.1f} μs</text>'
        )
        stage_x += sw

    # Total latency
    total_ms = total_time * 1000
    svg_parts.append(
        f'<text x="1015" y="{stage_y + stage_h/2}" font-size="10" font-weight="600" fill="#d94841">'
        f'Total: {total_ms:.1f} μs</text>'
    )

    # Clock cycles annotation
    svg_parts.append(
        f'<text x="55" y="45" font-size="12" font-weight="600" fill="#222">'
        f'FPGA Pipeline Stages (100MHz, {int(total_time * 100000)} cycles total)</text>'
    )

    # Resource utilization (bottom left)
    res_x, res_y = 55, 260
    svg_parts.append(
        f'<text x="{res_x}" y="{res_y}" font-size="13" font-weight="600" fill="#222">'
        f'Resource Utilization (XCZU9EG)</text>'
    )

    resources = [
        ("DSP Slices", 720, 2520, "#2563eb"),
        ("BRAM (36Kb)", 412, 912, "#7c3aed"),
        ("LUTs", 89000, 274000, "#059669"),
        ("Flip-Flops", 175000, 548000, "#e67e22"),
    ]

    max_bar_w = 280
    for i, (name, used, total, color) in enumerate(resources):
        ry = res_y + 28 + i * 42
        ratio = used / total
        bw = max_bar_w * ratio
        svg_parts.append(
            f'<text x="{res_x}" y="{ry+6}" font-size="10" fill="#222">{name}</text>'
        )
        svg_parts.append(
            f'<rect x="{res_x+120}" y="{ry}" width="{bw}" height="18" rx="3" '
            f'fill="{color}" fill-opacity="0.7" class="clickable" '
            f'{ttip(name, f"Used: {used:,} / {total:,} ({ratio*100:.1f}%)")}/>'
        )
        svg_parts.append(
            f'<text x="{res_x+128+bw}" y="{ry+13}" font-size="9" fill="#6c757d">'
            f'{used:,} / {total:,} ({ratio*100:.1f}%)</text>'
        )

    # Dataflow diagram (bottom right)
    df_x, df_y = 500, 260
    svg_parts.append(
        f'<text x="{df_x}" y="{df_y}" font-size="13" font-weight="600" fill="#222">'
        f'Dataflow Architecture</text>'
    )

    df_blocks = [
        ("AER Bus\n10M evt/s", df_x, df_y + 30, 110, 55, C_BUS),
        ("Ring Buffer\n4096 events", df_x + 140, df_y + 30, 110, 55, C_STAGE),
        ("Spatial Hash\nGrid 0.15", df_x + 280, df_y + 30, 110, 55, C_STAGE),
        ("Systolic Array\n8 PEs × 16", df_x + 420, df_y + 30, 110, 55, C_STAGE),
    ]
    for label, bx, by, bw, bh, bc in df_blocks:
        svg_parts.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="{bc}" '
            f'fill-opacity="0.12" stroke="{bc}" stroke-width="1.5" class="clickable" '
            f'{ttip(label.split(chr(10))[0], label.split(chr(10))[1] if len(label.split(chr(10)))>1 else "")}/>'
        )
        lines = label.split("\n")
        for li, line in enumerate(lines):
            svg_parts.append(
                f'<text x="{bx+bw/2}" y="{by+22+li*16}" text-anchor="middle" '
                f'font-size="10" font-weight="600" fill="{bc}">{line}</text>'
            )
        # Arrow
        if df_blocks.index((label, bx, by, bw, bh, bc)) < len(df_blocks) - 1:
            nb = df_blocks[df_blocks.index((label, bx, by, bw, bh, bc)) + 1]
            svg_parts.append(
                f'<line x1="{bx+bw}" y1="{by+bh/2}" x2="{nb[1]}" y2="{nb[2]+nb[4]/2}" '
                f'stroke="#6c757d" stroke-width="1.5" marker-end="url(#arr-df)"/>'
            )

    # Throughput table
    tp_x, tp_y = 500, 390
    svg_parts.append(
        f'<rect x="{tp_x}" y="{tp_y}" width="540" height="115" rx="4" '
        f'fill="white" stroke="#dee2e6" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="{tp_x+270}" y="{tp_y+22}" text-anchor="middle" font-size="11" '
        f'font-weight="600" fill="#222">Throughput Analysis</text>'
    )
    tp_data = [
        ("Event Input Rate", "10M events/sec", "Sustained AER parallel bus"),
        ("k-NN Throughput", "4.1M neighbors/sec", "Spatial hash O(k) lookup"),
        ("Encoder Throughput", "1.0M events/sec", "Systolic array, d=128, INT8"),
        ("End-to-End Latency", f"{total_ms:.1f} μs", f"{int(total_time * 100000)} cycles @ 100MHz"),
    ]
    for i, (metric, value, note) in enumerate(tp_data):
        ry = tp_y + 40 + i * 18
        svg_parts.append(
            f'<text x="{tp_x+12}" y="{ry}" font-size="10" fill="#6c757d">{metric}</text>'
        )
        svg_parts.append(
            f'<text x="{tp_x+220}" y="{ry}" font-size="10" font-weight="600" fill="#222">{value}</text>'
        )
        svg_parts.append(
            f'<text x="{tp_x+400}" y="{ry}" font-size="10" fill="#adb5bd">{note}</text>'
        )

    # Clock domain annotation
    svg_parts.append(
        f'<text x="55" y="545" font-size="10" fill="#6c757d">'
        f'⏱ Clock domains: AER (100MHz) → Pipeline (100MHz) → PWM (50Hz) → ARM (100Hz control loop)'
        f'</text>'
    )
    svg_parts.append(
        f'<text x="55" y="565" font-size="10" fill="#adb5bd">'
        f'🔒 Memory: Ring buffer (BRAM, 3-cycle read) | Weights: DDR4 (AXI4-Stream, 32-bit) | '
        f'Quantization: INT8 activations, INT8 weights'
        f'</text>'
    )

    svg_parts.insert(1,
        '<defs>'
        '<marker id="arr-df" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#6c757d"/>'
        '</marker>'
        '</defs>'
    )
    svg_parts.append('</svg>')

    legend = [
        {"type": "swatch", "color": "#7c3aed", "label": "AER Interface"},
        {"type": "swatch", "color": "#2563eb", "label": "FPGA Pipeline Stage"},
        {"type": "swatch", "color": "#059669", "label": "ARM CPU"},
        {"type": "swatch", "color": "#e67e22", "label": "Resource Utilization"},
    ]

    html = wrap_html(
        title="FPGA Pipeline: Stages, Timing, and Resource Utilization",
        label="Fig. 4",
        caption=(
            f"Complete FPGA pipeline for real-time normal flow inference. Total end-to-end latency "
            f"is {total_ms:.1f} μs ({int(total_time * 100000)} clock cycles at 100MHz), enabling sustained "
            f"1kHz inference. The systolic encoder array uses 8 processing elements × 16-wide SIMD "
            f"for INT8 vectorized kernel mixture computation. Resource utilization shown for Xilinx "
            f"Zynq UltraScale+ XCZU9EG."
        ),
        legend_items=legend,
        svg_content="\n".join(svg_parts)
    )
    return html


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: TTC and Collision Prediction
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig5_ttc_prediction():
    """Time-to-collision estimation and collision prediction."""
    W, H = 1100, 620

    def ttip(title, detail=""):
        return f'data-tooltip=\'{json.dumps({"title": title, "detail": detail})}\''

    C_TTC = "#d94841"
    C_SAFE = "#27ae60"
    C_TRACK = "#2563eb"
    C_DIVERGE = "#e67e22"

    svg_parts = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family: Hind, sans-serif;">',
        f'<rect width="{W}" height="{H}" fill="#fafbfc" rx="6"/>',
    ]

    # === TTC Formula (top left) ===
    form_x, form_y = 40, 30
    svg_parts.append(
        f'<rect x="{form_x}" y="{form_y}" width="340" height="130" rx="6" '
        f'fill="white" stroke="#dee2e6" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="{form_x+170}" y="{form_y+22}" text-anchor="middle" font-size="12" '
        f'font-weight="600" fill="#222">Lee\'s τ — Time-to-Collision</text>'
    )
    svg_parts.append(
        f'<text x="{form_x+170}" y="{form_y+55}" text-anchor="middle" font-size="14" '
        f'font-style="italic" fill="{C_TTC}">τ = R / Ṙ</text>'
    )
    svg_parts.append(
        f'<text x="{form_x+170}" y="{form_y+78}" text-anchor="middle" font-size="10" fill="#6c757d">'
        f'where R = object distance, Ṙ = rate of closure</text>'
    )
    svg_parts.append(
        f'<text x="{form_x+170}" y="{form_y+100}" text-anchor="middle" font-size="10" fill="#6c757d">'
        f'Flow divergence ∝ 1/τ for looming objects</text>'
    )
    svg_parts.append(
        f'<text x="{form_x+170}" y="{form_y+120}" text-anchor="middle" font-size="9" fill="#adb5bd">'
        f'Thresholds: τ < 1.0s → evade | τ < 0.5s → hard evade</text>'
    )

    # === TTC vs Distance Graph (top right) ===
    graph_x, graph_y = 420, 30
    graph_w, graph_h = 640, 220
    svg_parts.append(
        f'<rect x="{graph_x}" y="{graph_y}" width="{graph_w}" height="{graph_h}" rx="6" '
        f'fill="white" stroke="#dee2e6" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="{graph_x+graph_w/2}" y="{graph_y+20}" text-anchor="middle" font-size="11" '
        f'font-weight="600" fill="#222">TTC vs. Object Distance (various approach speeds)</text>'
    )

    # Axes
    pad_l, pad_r, pad_t, pad_b = 55, 20, 40, 40
    gx = graph_x + pad_l
    gy = graph_y + pad_t
    gw = graph_w - pad_l - pad_r
    gh = graph_h - pad_t - pad_b

    svg_parts.append(f'<line x1="{gx}" y1="{gy+gh}" x2="{gx+gw}" y2="{gy+gh}" stroke="#222" stroke-width="1"/>')
    svg_parts.append(f'<line x1="{gx}" y1="{gy}" x2="{gx}" y2="{gy+gh}" stroke="#222" stroke-width="1"/>')
    svg_parts.append(
        f'<text x="{gx+gw/2}" y="{gy+gh+22}" text-anchor="middle" font-size="9" fill="#6c757d">'
        f'Distance (m)</text>'
    )
    svg_parts.append(
        f'<text x="{gx-10}" y="{gy+gh/2}" text-anchor="middle" font-size="9" fill="#6c757d" '
        f'transform="rotate(-90, {gx-10}, {gy+gh/2})">TTC (seconds)</text>'
    )

    # Grid lines & labels
    for d in [0, 5, 10, 15, 20]:
        dx = gx + (d / 20) * gw
        svg_parts.append(f'<line x1="{dx}" y1="{gy}" x2="{dx}" y2="{gy+gh}" stroke="#e9ecef" stroke-width="0.5"/>')
        svg_parts.append(f'<text x="{dx}" y="{gy+gh+12}" text-anchor="middle" font-size="8" fill="#adb5bd">{d}</text>')

    for ttc_val in [0, 1, 2, 3, 4, 5]:
        ty = gy + (1 - ttc_val / 5) * gh
        svg_parts.append(f'<line x1="{gx}" y1="{ty}" x2="{gx+gw}" y2="{ty}" stroke="#e9ecef" stroke-width="0.5"/>')
        svg_parts.append(f'<text x="{gx-6}" y="{ty+3}" text-anchor="end" font-size="8" fill="#adb5bd">{ttc_val}</text>')

    # TTC curves for different approach speeds
    speeds = [(1, "#27ae60"), (3, "#f1c40f"), (5, "#e67e22"), (10, "#d94841"), (20, "#8e0000")]
    for speed, color in speeds:
        path_parts = []
        for d in range(1, 21):
            ttc = d / speed
            px = gx + (d / 20) * gw
            py = gy + (1 - min(ttc, 5) / 5) * gh
            path_parts.append(f'{px:.1f},{py:.1f}')
        svg_parts.append(
            f'<polyline points="{" ".join(path_parts)}" fill="none" stroke="{color}" '
            f'stroke-width="2" class="clickable" '
            f'{ttip(f"{speed} m/s approach", f"At {speed} m/s, TTC < 1s at {speed}m distance")}/>'
        )

    # Danger thresholds
    for ttc_thresh, thresh_color, label in [(1.0, "#e67e22", "τ=1.0s"), (0.5, "#d94841", "τ=0.5s")]:
        yth = gy + (1 - ttc_thresh / 5) * gh
        svg_parts.append(
            f'<line x1="{gx}" y1="{yth}" x2="{gx+gw}" y2="{yth}" stroke="{thresh_color}" '
            f'stroke-width="1" stroke-dasharray="6,3"/>'
        )
        svg_parts.append(
            f'<text x="{gx+gw+4}" y="{yth+3}" font-size="8" fill="{thresh_color}">{label}</text>'
        )

    # Speed legend
    for i, (speed, color) in enumerate(speeds):
        lx = graph_x + graph_w - 130
        ly = graph_y + 40 + i * 16
        svg_parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+20}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        svg_parts.append(f'<text x="{lx+24}" y="{ly+3}" font-size="8" fill="#6c757d">{speed} m/s</text>')

    # === Kalman Tracking Box (bottom left) ===
    kf_x, kf_y = 40, 190
    kf_w, kf_h = 340, 150

    svg_parts.append(
        f'<rect x="{kf_x}" y="{kf_y}" width="{kf_w}" height="{kf_h}" rx="6" '
        f'fill="white" stroke="#dee2e6" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="{kf_x+kf_w/2}" y="{kf_y+20}" text-anchor="middle" font-size="11" '
        f'font-weight="600" fill="#222">4D Kalman Filter Tracker</text>'
    )

    kf_states = [
        ("State Vector", "x = [x, y, vₓ, vᵧ]ᵀ"),
        ("Process Model", "Constant velocity"),
        ("Measurement", "Flow centroid (u, v)"),
        ("Data Association", "Mahalanobis distance"),
        ("Track Lifecycle", "Init → Confirm → Coast → Delete"),
    ]
    for i, (key, val) in enumerate(kf_states):
        ky = kf_y + 42 + i * 20
        svg_parts.append(
            f'<text x="{kf_x+10}" y="{ky}" font-size="9" font-weight="600" fill="#6c757d">{key}</text>'
        )
        svg_parts.append(
            f'<text x="{kf_x+180}" y="{ky}" font-size="9" fill="#222">{val}</text>'
        )

    # === Safe Bearing Diagram (bottom right) ===
    sb_x, sb_y = 420, 270
    svg_parts.append(
        f'<text x="{sb_x}" y="{sb_y+15}" font-size="13" font-weight="600" fill="#222">'
        f'Safe Bearing Computation</text>'
    )

    # Drone at center
    drone_cx, drone_cy = 700, 420
    svg_parts.append(
        f'<circle cx="{drone_cx}" cy="{drone_cy}" r="30" fill="{C_SAFE}" fill-opacity="0.1" '
        f'stroke="{C_SAFE}" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="{drone_cx}" y="{drone_cy+4}" text-anchor="middle" font-size="9" '
        f'font-weight="600" fill="{C_SAFE}">Drone</text>'
    )

    # Obstacles
    obstacles = [
        (drone_cx - 80, drone_cy - 90, 18, "Obj A\nτ=0.3s"),
        (drone_cx + 100, drone_cy - 60, 14, "Obj B\nτ=1.2s"),
        (drone_cx - 50, drone_cy + 80, 20, "Obj C\nτ=0.6s"),
    ]
    for ox, oy, orad, olabel in obstacles:
        svg_parts.append(
            f'<circle cx="{ox}" cy="{oy}" r="{orad}" fill="#d94841" fill-opacity="0.3" '
            f'stroke="#d94841" stroke-width="1.5"/>'
        )
        svg_parts.append(
            f'<text x="{ox}" y="{oy+2}" text-anchor="middle" font-size="7" fill="#d94841">{olabel}</text>'
        )

    # Bearing lines to obstacles
    for ox, oy, orad, olabel in obstacles:
        angle = math.atan2(oy - drone_cy, ox - drone_cx)
        dist = math.hypot(ox - drone_cx, oy - drone_cy)
        ex = drone_cx + math.cos(angle) * (dist - orad)
        ey = drone_cy + math.sin(angle) * (dist - orad)
        svg_parts.append(
            f'<line x1="{drone_cx}" y1="{drone_cy}" x2="{ex}" y2="{ey}" '
            f'stroke="#adb5bd" stroke-width="1" stroke-dasharray="4,3"/>'
        )

    # Safe bearing arc
    safe_angles = []
    for angle_deg in range(-60, 61, 3):
        rad = math.radians(angle_deg + 270)  # pointing "up"
        safe_angles.append((drone_cx + 55 * math.cos(rad), drone_cy + 55 * math.sin(rad)))

    if safe_angles:
        path_str = f'M{safe_angles[0][0]:.1f},{safe_angles[0][1]:.1f}'
        for sx, sy in safe_angles[1:]:
            path_str += f' L{sx:.1f},{sy:.1f}'
        svg_parts.append(
            f'<path d="{path_str}" fill="none" stroke="{C_SAFE}" stroke-width="2.5"/>'
        )
        svg_parts.append(
            f'<text x="{drone_cx-10}" y="{drone_cy-48}" font-size="9" fill="{C_SAFE}" font-weight="600">'
            f'Safe Sector</text>'
        )

    # Blocked sector
    blocked_angles = []
    for angle_deg in range(61, 120, 3):
        rad = math.radians(angle_deg + 270)
        blocked_angles.append((drone_cx + 65 * math.cos(rad), drone_cy + 65 * math.sin(rad)))
    if blocked_angles:
        path_str = f'M{blocked_angles[0][0]:.1f},{blocked_angles[0][1]:.1f}'
        for bx, by in blocked_angles[1:]:
            path_str += f' L{bx:.1f},{by:.1f}'
        svg_parts.append(
            f'<path d="{path_str}" fill="none" stroke="{C_TTC}" stroke-width="2" stroke-dasharray="4,3"/>'
        )
        svg_parts.append(
            f'<text x="{drone_cx+70}" y="{drone_cy-55}" font-size="8" fill="{C_TTC}">Blocked</text>'
        )

    # Evasion vector
    svg_parts.append(
        f'<line x1="{drone_cx}" y1="{drone_cy}" x2="{drone_cx-90}" y2="{drone_cy-70}" '
        f'stroke="{C_SAFE}" stroke-width="3" marker-end="url(#arr-evade)"/>'
    )
    svg_parts.append(
        f'<text x="{drone_cx-95}" y="{drone_cy-75}" font-size="9" font-weight="600" fill="{C_SAFE}">'
        f'Evasion</text>'
    )

    # === Threat summary table (bottom center) ===
    tbl_x, tbl_y = 40, 365
    svg_parts.append(
        f'<rect x="{tbl_x}" y="{tbl_y}" width="340" height="200" rx="4" '
        f'fill="white" stroke="#dee2e6" stroke-width="1"/>'
    )
    svg_parts.append(
        f'<text x="{tbl_x+170}" y="{tbl_y+20}" text-anchor="middle" font-size="11" '
        f'font-weight="600" fill="#222">Threat Summary</text>'
    )
    threat_data = [
        ("Object A", "0.3s", "CRITICAL", "N 42°W", "dodge right"),
        ("Object B", "1.2s", "CAUTION", "N 28°E", "gentle left"),
        ("Object C", "0.6s", "WARNING", "S 15°E", "climb"),
    ]
    headers = ["Object", "TTC", "Level", "Bearing", "Action"]
    for hi, hdr in enumerate(headers):
        svg_parts.append(
            f'<text x="{tbl_x+12+hi*63}" y="{tbl_y+38}" font-size="8" font-weight="600" fill="#6c757d">'
            f'{hdr}</text>'
        )
    for ri, (obj, ttc, level, bearing, action) in enumerate(threat_data):
        ry = tbl_y + 52 + ri * 18
        level_colors = {"CRITICAL": "#e74c3c", "WARNING": "#e67e22", "CAUTION": "#f1c40f"}
        vals = [obj, ttc, level, bearing, action]
        for vi, val in enumerate(vals):
            fill = level_colors.get(val, "#222") if vi == 2 else "#222"
            svg_parts.append(
                f'<text x="{tbl_x+12+vi*63}" y="{ry}" font-size="8" fill="{fill}">{val}</text>'
            )

    svg_parts.insert(1,
        '<defs>'
        '<marker id="arr-evade" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#27ae60"/>'
        '</marker>'
        '</defs>'
    )
    svg_parts.append('</svg>')

    legend = [
        {"type": "line", "color": "#27ae60", "label": "Safe trajectory"},
        {"type": "line", "color": "#d94841", "label": "Collision threat / blocked bearing"},
        {"type": "swatch", "color": C_TTC, "label": "TTC threshold (τ < 1.0s)"},
        {"type": "swatch", "color": C_TRACK, "label": "Kalman filter tracking"},
    ]

    html = wrap_html(
        title="Time-to-Collision Estimation and Collision Prediction",
        label="Fig. 5",
        caption=(
            "Collision prediction pipeline. Lee's τ (TTC = R/Ṙ) is computed from flow divergence "
            "for looming objects. A 4D Kalman filter tracks multiple objects with Mahalanobis "
            "data association. Safe bearing sectors are computed, and graded evasion vectors "
            "are generated for all threats exceeding τ=1.0s."
        ),
        legend_items=legend,
        svg_content="\n".join(svg_parts)
    )
    return html


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figures = {
        "fig1_architecture.html": generate_fig1_architecture,
        "fig2_normal_flow.html": generate_fig2_normal_flow,
        "fig3_evasion_levels.html": generate_fig3_evasion_levels,
        "fig4_fpga_pipeline.html": generate_fig4_fpga_pipeline,
        "fig5_ttc_prediction.html": generate_fig5_ttc_prediction,
    }

    for filename, generator in figures.items():
        filepath = OUTPUT_DIR / filename
        html = generator()
        with open(filepath, "w") as f:
            f.write(html)
        size_kb = filepath.stat().st_size / 1024
        print(f"  ✅ {filename}  ({size_kb:.1f} KB)")

    print(f"\nGenerated {len(figures)} interactive figures in {OUTPUT_DIR}/")
    print("Open any .html file in a browser to view.")
    print("  Scroll to zoom | Drag to pan | Hover for tooltips")


if __name__ == "__main__":
    main()