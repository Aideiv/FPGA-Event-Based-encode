/**
 * Screenshot all interactive figure HTML files to PNG thumbnails.
 * Uses Puppeteer to render each figure at 1200px wide and capture.
 *
 * Usage: node figures/screenshot_figures.js
 * Output: assets/fig1_architecture.png ... assets/fig5_ttc_prediction.png
 */
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const FIGURES_DIR = path.resolve(__dirname);
const ASSETS_DIR = path.resolve(__dirname, '..', 'assets');

const figures = [
  { file: 'fig1_architecture.html', name: 'System Architecture' },
  { file: 'fig2_normal_flow.html', name: 'VecKM Normal Flow' },
  { file: 'fig3_evasion_levels.html', name: 'Evasion Levels' },
  { file: 'fig4_fpga_pipeline.html', name: 'FPGA Pipeline Timing' },
  { file: 'fig5_ttc_prediction.html', name: 'TTC Prediction' },
];

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  for (const { file, name } of figures) {
    const htmlPath = path.join(FIGURES_DIR, file);
    const pngPath = path.join(ASSETS_DIR, file.replace('.html', '.png'));

    console.log(`📸 Screenshotting: ${name}...`);
    const page = await browser.newPage();
    await page.setViewport({ width: 1200, height: 900, deviceScaleFactor: 2 });

    const fileUrl = `file://${htmlPath}`;
    await page.goto(fileUrl, { waitUntil: 'networkidle0', timeout: 15000 });

    // Wait for fonts and rendering
    await page.waitForSelector('.figure-container', { timeout: 10000 });
    await new Promise(r => setTimeout(r, 1000));

    // Get the actual figure-container height and clip to it
    const container = await page.$('.figure-container');
    const box = await container.boundingBox();

    await page.screenshot({
      path: pngPath,
      clip: {
        x: box.x,
        y: box.y,
        width: box.width,
        height: Math.min(box.height, 1800), // Cap height for very long figures
      },
    });

    const sizeKb = (fs.statSync(pngPath).size / 1024).toFixed(1);
    console.log(`  ✅ ${file.replace('.html', '.png')} (${sizeKb} KB)`);
    await page.close();
  }

  await browser.close();
  console.log(`\nDone! ${figures.length} PNG previews saved to assets/`);
})();