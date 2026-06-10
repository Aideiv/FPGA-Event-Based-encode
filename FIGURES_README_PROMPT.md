## Task: Recreate Interactive Figures + README Gallery

This repo uses a self-contained figure pipeline:
1. `figures/generate_figures.py` — produces 5 interactive HTML figures (SVG + hover tooltips + zoom/pan)
2. `figures/screenshot_figures.js` — screenshots each HTML figure to a PNG preview via Puppeteer
3. `assets/fig{1..5}_*.png` — the PNGs displayed as clickable thumbnails in `README.md`
4. `README.md` lines 30–62 — "Interactive Figures" gallery section linking PNGs to HTMLs

The HTML figures and PNG previews may be missing, deleted, or stale. Recreate them from scratch.

### Step 1: Regenerate the Interactive HTML Figures
```bash
python figures/generate_figures.py
```
This writes 5 files to `figures/`:
- `fig1_architecture.html` — System architecture diagram
- `fig2_normal_flow.html` — VecKM normal flow estimation pipeline
- `fig3_evasion_levels.html` — Five-level graded evasion response
- `fig4_fpga_pipeline.html` — FPGA pipeline timing & resource utilization
- `fig5_ttc_prediction.html` — TTC estimation & Kalman tracking

### Step 2: Screenshot PNG Previews
```bash
node figures/screenshot_figures.js
```
Requires Puppeteer (`npm install puppeteer` if missing). This loads each HTML file in a headless browser at 1200×900 viewport (2× device scale), clips the `.figure-container` element, and saves to `assets/`:
- `assets/fig1_architecture.png`
- `assets/fig2_normal_flow.png`
- `assets/fig3_evasion_levels.png`
- `assets/fig4_fpga_pipeline.png`
- `assets/fig5_ttc_prediction.png`

### Step 3: Add/Verify the README Gallery Section
Ensure `README.md` contains the "Interactive Figures" section (between the demo GIF and the API section) with clickable PNG thumbnails linking to the HTML files. The section should:
- Use `<a href="figures/figN_....html"><img src="assets/figN_....png" width="100%"></a>` for each figure
- Include a one-sentence italic caption per figure
- Note interaction instructions: scroll = zoom, drag = pan, hover = tooltips
- Credit the generator and screenshot scripts

### Step 4: Verify
```bash
ls -la figures/fig{1,2,3,4,5}_*.html
ls -la assets/fig{1,2,3,4,5}_*.png
```
All 10 files should exist. Open any `.html` file in a browser to confirm interactivity works.

### Architecture Notes for Repurposing
- `generate_figures.py` is pure Python with zero dependencies (stdlib only). It inlines all CSS/JS/SVG.
- Each figure is a standalone HTML file — no server needed. They can be opened directly with `file://`.
- To swap the figure content for a different project, edit the 5 builder functions at the bottom of `generate_figures.py` (builders create SVG elements specific to this project's domain — architecture, flow estimation, evasion, FPGA pipeline, TTC).
- The shared CSS framework and zoom/pan JS are reusable as-is for any scientific figure.
- `screenshot_figures.js` uses Puppeteer; the figure list and paths are defined in a config array at the top — update those if figures change.