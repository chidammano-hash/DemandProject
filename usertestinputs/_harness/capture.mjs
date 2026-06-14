/**
 * UX capture harness — drives the live app with Playwright as a demand planner would,
 * screenshots each tab, and dumps visible text + console errors for critique.
 *
 * Usage (run from frontend/):
 *   node ../usertestinputs/_harness/capture.mjs <cycleDir>
 * where <cycleDir> is an absolute path like
 *   /Users/.../usertestinputs/cycle1
 *
 * Requires: API on :8000 and Vite on :5173 already running (they are).
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { createRequire } from "node:module";

// Resolve Playwright from the frontend install regardless of this script's cwd/location.
const require = createRequire("/Users/manoharchidambaram/projects/DemandProject/frontend/package.json");
const { chromium } = require("@playwright/test");

const cycleDir = process.argv[2];
if (!cycleDir) {
  console.error("usage: node capture.mjs <absolute-cycle-dir>");
  process.exit(1);
}
const screensDir = join(cycleDir, "screens");
mkdirSync(screensDir, { recursive: true });

const BASE = "http://localhost:5173";

// Demand-planner-relevant tabs (key -> friendly label). Keys match App.tsx activeTab + ?tab=.
const TABS = [
  ["commandCenter", "Command Center"],
  ["aggregateAnalysis", "Aggregate / Accuracy"],
  ["demandHistory", "Demand History"],
  ["invPlanning", "Inventory Planning"],
  ["controlTower", "Control Tower"],
  ["sop", "S&OP"],
  ["fva", "FVA & ROI"],
  ["aiPlanner", "AI Planner"],
  ["aiPlannerFva", "AI Planner FVA"],
  ["dataQuality", "Data Quality"],
  ["itemAnalysis", "Item Analysis"],
  ["explorer", "Explorer"],
  ["customerAnalytics", "Customer Map"],
  ["clusters", "Clusters"],
];

const results = [];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

for (const [key, label] of TABS) {
  const consoleErrors = [];
  const onMsg = (m) => {
    if (m.type() === "error") consoleErrors.push(m.text().slice(0, 300));
  };
  page.on("console", onMsg);
  page.on("pageerror", (e) => consoleErrors.push("PAGEERROR: " + e.message.slice(0, 300)));

  let text = "";
  let ok = true;
  let err = "";
  try {
    await page.goto(`${BASE}/?tab=${key}`, { waitUntil: "networkidle", timeout: 30000 });
    await page.waitForSelector('[role="navigation"]', { timeout: 15000 });
    await page.waitForTimeout(2500); // let lazy panels + data settle
    const content = page.locator("#tab-content");
    text = (await content.innerText().catch(() => "")).slice(0, 6000);
    await page.screenshot({ path: join(screensDir, `${key}.png`), fullPage: true });
  } catch (e) {
    ok = false;
    err = String(e).slice(0, 400);
    try { await page.screenshot({ path: join(screensDir, `${key}-ERROR.png`), fullPage: true }); } catch {}
  }
  page.off("console", onMsg);
  results.push({ key, label, ok, err, consoleErrors, textLen: text.length, text });
  console.log(`[${ok ? "OK " : "ERR"}] ${key} (${label}) text=${text.length} consoleErrors=${consoleErrors.length}`);
}

await browser.close();

// Write a machine-readable dump the planner agent reads alongside screenshots.
writeFileSync(join(cycleDir, "capture-dump.json"), JSON.stringify(results, null, 2));

// Write a compact text digest (text + errors per tab) for quick reading.
const digest = results
  .map(
    (r) =>
      `### ${r.label} (?tab=${r.key})  [${r.ok ? "loaded" : "FAILED: " + r.err}]\n` +
      (r.consoleErrors.length ? `CONSOLE ERRORS:\n${r.consoleErrors.map((e) => "  - " + e).join("\n")}\n` : "") +
      `VISIBLE TEXT (first 6k):\n${r.text}\n`
  )
  .join("\n\n");
writeFileSync(join(cycleDir, "capture-digest.md"), digest);

console.log(`\nWrote ${results.length} tabs -> ${screensDir}`);
console.log(`Dump: ${join(cycleDir, "capture-dump.json")}`);
console.log(`Digest: ${join(cycleDir, "capture-digest.md")}`);
