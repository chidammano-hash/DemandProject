export const meta = {
  name: 'ux-improvement-loop',
  description: 'Loop-until-dry UX hardening: per cycle a demand-planner + usability reviewer critique the live app, a fixer applies & tests fixes. Cap 100 cycles.',
  whenToUse: 'Drive a supply-chain planning UI toward best-in-class via repeated critique→fix cycles against the live app.',
  phases: [
    { title: 'Capture' },
    { title: 'Critique' },
    { title: 'Fix' },
  ],
}

// ---- config (from args, with safe defaults) ----
const ROOT = '/Users/manoharchidambaram/projects/DemandProject'
const UTI = `${ROOT}/tests/Automated_tests`   // fresh output home (reset from usertestinputs)
const HARNESS = `${UTI}/_harness`
// ─────────────────────────────────────────────────────────────────────────────
// CYCLE COUNT — edit DEFAULT_N_CYCLES to control how many cycles a launch runs.
// Args (nCycles/maxCycles) are honored WHEN they arrive, but this harness has seen
// the Workflow runtime silently drop args, so the editable constant is the reliable
// lever. The loop always runs EXACTLY this many cycles from the probed start
// (START_CYCLE .. START_CYCLE + N_CYCLES - 1), unless dry-stop ends it sooner.
// ─────────────────────────────────────────────────────────────────────────────
const DEFAULT_N_CYCLES = 5
const DEFAULT_DRY_STOP = 2

const DRY_STOP = (args && args.dryStop) || DEFAULT_DRY_STOP   // stop after this many consecutive dry cycles (no new P0/P1/P2)
const N_CYCLES = (args && (args.nCycles || args.maxCycles)) || DEFAULT_N_CYCLES // cycles to run this launch
// START_CYCLE / MAX_CYCLES are resolved AFTER probing the output dir below — never trusted to args alone,
// because args plumbing can silently drop values and clobber existing cycles.

// ---- shared command reference baked into every agent prompt ----
const ENV = `
ENVIRONMENT (memorize — \`make\` does NOT work here):
- Live app: React UI http://localhost:5173 (Vite, HMR on) -> proxies to FastAPI http://localhost:8000 (host uvicorn --reload, serves current host code). Postgres on localhost:5440 (db demand_mvp, user/pass demand/demand) via docker. Redis localhost:6379.
- Backend tests (run from ${ROOT}): ~/.local/bin/uv run pytest tests/ -q   (target specific files for speed, e.g. ~/.local/bin/uv run pytest tests/api/test_x.py -q)
- Run a python one-liner against the DB: ~/.local/bin/uv run python -c "..."
- Frontend tests (run from ${ROOT}/frontend): PATH="/opt/homebrew/bin:$PATH" /opt/homebrew/bin/node node_modules/.bin/vitest run --reporter=dot
- Hit an endpoint: curl -s -m 10 'http://localhost:8000/<path>' | head
- Psql: docker exec demandproject-postgres-1 psql -U demand -d demand_mvp -c "..."
- Backend edits hot-reload (uvicorn --reload); frontend edits HMR. So applied fixes go LIVE for the next cycle's scan.
`

const RULES = `
PROJECT RULES (from ${ROOT}/CLAUDE.md — obey exactly):
- psycopg3 uses %s placeholders, NEVER $1/$2. Identifier interpolation via psycopg.sql.Identifier only.
- inv_planning_* routers use get_conn(), NOT Depends(_get_pool).
- NO bare \`except Exception\`: catch specific (psycopg.Error, ValueError, ...) + logger.exception(). A \`# noqa: BLE001 — reason\` needs justification.
- 5xx HTTPException detail = short verb-phrase, NEVER interpolate exception text / str(exc).
- Every write endpoint (post/put/delete/patch) needs dependencies=[Depends(require_api_key)].
- Frontend: all HTTP via fetchJson in src/api/queries/<module>.ts (never raw fetch in tabs); NO \`: any\`/\`as any\` in queries; charts read theme from useThemeContext()/useChartColors() (no theme prop, no inline hex); tab files < 600 lines (split into subpanels).
- New API prefix: add to BOTH frontend/vite.config.ts API_PATH_PREFIXES AND frontend/src/api/queries/index.ts barrel.
- date.today() forbidden outside common/core/planning_date.py — use get_planning_date().
- Tests mandatory: API tests use make_pool/make_async_pool from tests/api/conftest.py + httpx AsyncClient + ASGITransport; async tests patch api.core._get_async_pool.
- Do NOT recreate deleted configs. Do NOT add backward-compat shims. Do NOT commit.
`

// ---- schemas ----
const PLANNER_SCHEMA = {
  type: 'object',
  required: ['summary', 'findings', 'newActionableCount'],
  properties: {
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'title', 'severity', 'isNew'],
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['P0', 'P1', 'P2', 'P3'] },
          workflow: { type: 'string' },
          evidence: { type: 'string' },
          rootCause: { type: 'string' },
          acceptance: { type: 'string' },
          isNew: { type: 'boolean' },
        },
      },
    },
    newActionableCount: { type: 'integer', description: 'count of NEW, unresolved P0/P1/P2 findings this cycle' },
  },
}

const USABILITY_SCHEMA = {
  type: 'object',
  required: ['summary', 'items', 'newActionableCount'],
  properties: {
    summary: { type: 'string' },
    items: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'title', 'category', 'severity', 'isNew'],
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          category: { type: 'string', enum: ['simplification', 'usability', 'consistency', 'accessibility', 'performance', 'information-architecture'] },
          severity: { type: 'string', enum: ['P0', 'P1', 'P2', 'P3'] },
          evidence: { type: 'string' },
          file: { type: 'string' },
          recommendation: { type: 'string' },
          acceptance: { type: 'string' },
          isNew: { type: 'boolean' },
        },
      },
    },
    newActionableCount: { type: 'integer' },
  },
}

const FIXER_SCHEMA = {
  type: 'object',
  required: ['cycle', 'fixed', 'deferred', 'testsPass', 'remainingActionable'],
  properties: {
    cycle: { type: 'integer' },
    fixed: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'what', 'files', 'acceptanceMet', 'verification', 'redGreenEvidence'],
        properties: {
          id: { type: 'string' },
          what: { type: 'string' },
          files: { type: 'string' },
          redGreenEvidence: { type: 'string', description: 'TDD proof: the test written FIRST + that it FAILED (red) before the fix, then PASSED (green) after. e.g. "wrote test_action_feed_returns_critical; ran red: AssertionError 0!=20; implemented; green: 1 passed"' },
          verification: { type: 'string', description: 'live re-verify: curl before->after / UI behavior change' },
          acceptanceMet: { type: 'boolean' },
        },
      },
    },
    deferred: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'why'],
        properties: { id: { type: 'string' }, why: { type: 'string' } },
      },
    },
    testsPass: { type: 'boolean' },
    remainingActionable: { type: 'integer', description: 'NEW P0/P1/P2 issues found this cycle that were NOT fixed' },
    notes: { type: 'string' },
  },
}

// ---- resolve the cycle range from the OUTPUT DIR itself (ground truth, not fragile args) ----
const probe = await agent(
  `You are a harness operator. Determine the next cycle number for the UX loop.
Run EXACTLY this and report ONLY the resulting integer:
  mkdir -p ${UTI}; n=$(ls -d ${UTI}/cycle*/ 2>/dev/null | grep -oE 'cycle[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1); if [ -z "$n" ]; then echo 1; else echo $((n+1)); fi
That integer is (highest existing cycleN under ${UTI}) + 1, or 1 if none exist. Return it as startCycle.`,
  { label: 'probe-start-cycle', phase: 'Capture', schema: { type: 'object', required: ['startCycle'], properties: { startCycle: { type: 'integer', minimum: 1 } } } }
)
const START_CYCLE = (probe && probe.startCycle) || 1
const MAX_CYCLES = START_CYCLE + Math.max(1, N_CYCLES) - 1
log(`Resolved cycle range: ${START_CYCLE}..${MAX_CYCLES} (${N_CYCLES} cycle(s), dryStop=${DRY_STOP})`)

// ---- the loop ----
let dryStreak = 0
let aborted = false
const ledgerLine = []
const cycleSummaries = []

for (let c = START_CYCLE; c <= MAX_CYCLES; c++) {
  const cycleDir = `${UTI}/cycle${c}`
  log(`=== Cycle ${c}/${MAX_CYCLES} (dryStreak=${dryStreak}) ===`)

  // --- Phase 1: capture the live app ---
  const cap = await agent(
    `You are a test harness operator. Create the cycle dir and capture the live UI.
${ENV}
Run EXACTLY:
  mkdir -p ${cycleDir}/screens
  cd ${ROOT}/frontend && node ${HARNESS}/capture.mjs ${cycleDir}
This drives Playwright over 14 planner tabs, screenshots each to ${cycleDir}/screens/, and writes ${cycleDir}/capture-digest.md + capture-dump.json.
If the API appears down (many 500s on EVERY tab or connection refused), wait 8 seconds and retry once (a prior fix may have triggered a uvicorn reload).
Return a one-line status per tab: which loaded, which had console errors / non-2xx, and the text length. Keep it terse.`,
    { label: `capture-c${c}`, phase: 'Capture' }
  )

  // --- Phase 2: two reviewers in parallel (both READ-ONLY on code) ---
  const [planner, usability] = await parallel([
    () =>
      agent(
        `You are a SENIOR DEMAND PLANNER doing acceptance testing of the "Supply Chain Command Center". You judge the product ONLY by whether it supports real demand-planning workflows: morning portfolio triage, forecast accuracy & FVA review, exception/control-tower triage, inventory planning actions, demand history, S&OP prep, customer/item drill-down.
${ENV}
This is CYCLE ${c}. Earlier cycles already fixed issues — do NOT re-report resolved ones.
1. READ the running ledger ${UTI}/LEDGER.md (if it exists) and the previous findings ${UTI}/testinput${c - 1}.md to learn what is already fixed/known.
2. READ this cycle's evidence: ${cycleDir}/capture-digest.md and ${cycleDir}/capture-dump.json. View the most relevant screenshots in ${cycleDir}/screens/ with the Read tool (PNGs) — especially any *-ERROR.png and the key tabs (commandCenter, aggregateAnalysis, invPlanning, controlTower, fva).
3. INVESTIGATE root causes (read-only): for any 500/404/empty-where-data-should-exist, identify the failing endpoint (curl it), check API logs if useful (docker logs --tail 60 demandproject-api-1 is STOPPED now — instead the host uvicorn logs to its own terminal; rely on curl + reading code in api/routers and frontend/src). Trace to the file/component. DO NOT edit code — you are the planner.
4. Distinguish broken (error/empty-where-data-expected) vs genuinely-empty (no data; should show honest empty state) vs pure UX friction.
WRITE ${UTI}/testinput${c}.md with prioritized, evidence-backed findings. For each: id (F${c}.n), title, [SEV P0|P1|P2|P3], Workflow blocked, Evidence (tab/screenshot/console error/curl status), Root cause (endpoint+file), Acceptance criterion (concrete/testable), Planner impact. P0=blocks a daily workflow; P1=major friction/misleading; P2=clarity/efficiency; P3=polish.
Focus on the 6-12 highest-value items, NEW ones first. If the product is in good shape and you find few/no new issues, SAY SO honestly (do not invent issues).
Return the structured object: summary, findings[] (set isNew=true only for genuinely new/unresolved items), and newActionableCount = number of NEW unresolved P0/P1/P2 findings.`,
        { label: `planner-c${c}`, phase: 'Critique', schema: PLANNER_SCHEMA }
      ),
    () =>
      agent(
        `You are a PRINCIPAL UX + FRONTEND ENGINEER focused on SIMPLIFICATION and USABILITY of the "Supply Chain Command Center" React app. Your mandate: make the product simpler, more consistent, more usable, and more maintainable — toward best-in-class. You look at BOTH the rendered app and the code.
${ENV}
${RULES}
This is CYCLE ${c}. Do NOT re-report items already resolved.
1. READ ${UTI}/LEDGER.md (if present) and ${UTI}/usability${c - 1}.md (if present) to avoid duplicates.
2. READ this cycle's evidence: ${cycleDir}/capture-digest.md, capture-dump.json, and view key screenshots in ${cycleDir}/screens/ (Read tool on PNGs).
3. INSPECT code (read-only) for simplification/usability wins: ${ROOT}/frontend/src/tabs and src/components. Look for: redundant/cluttered UI, inconsistent labels/units/empty-states/loading-states, confusing information architecture, tabs > 600 lines that should split, duplicated query logic, missing affordances (no clear next action, no tooltips on jargon), inconsistent number/date/currency formatting, accessibility gaps (roles/labels), and slow/heavy panels. Prefer high-leverage simplifications a planner will feel.
WRITE ${UTI}/usability${c}.md with prioritized items. For each: id (U${c}.n), title, Category (simplification|usability|consistency|accessibility|performance|information-architecture), [SEV], Evidence (screenshot/text/file:line), File, Recommendation (concrete), Acceptance criterion (testable).
Aim for 6-12 high-value items, NEW first. Be honest if there's little left to improve.
Return the structured object with items[] (isNew flag) and newActionableCount = NEW P0/P1/P2 items.`,
        { label: `usability-c${c}`, phase: 'Critique', schema: USABILITY_SCHEMA }
      ),
  ])

  const plannerNew = (planner && planner.newActionableCount) || 0
  const usabilityNew = (usability && usability.newActionableCount) || 0
  const newTotal = plannerNew + usabilityNew
  log(`Cycle ${c}: planner=${plannerNew} new, usability=${usabilityNew} new (total ${newTotal})`)

  // Guard: if BOTH reviewers failed to produce findings (API/session-limit/death), this is an
  // ABORTED cycle, NOT a clean app. Bail out without polluting the dry-stop streak — otherwise
  // dead agents returning 0 get misread as "product is stable".
  if (!planner && !usability) {
    log(`Cycle ${c}: both reviewers failed to return findings — aborting loop (environment/limit issue, not a dry cycle).`)
    aborted = true
    break
  }

  // --- Phase 3: fixer applies + TESTS the top items from both reviewers ---
  const fixer = await agent(
    `You are a SENIOR FULL-STACK ENGINEER (React/TS + FastAPI/psycopg3/Postgres) on the "Supply Chain Command Center". Two reviewers filed findings this cycle. Fix the highest-value items using STRICT TEST-DRIVEN DEVELOPMENT on the current git branch, and document.
${ENV}
${RULES}
This is CYCLE ${c}.
1. READ the planner findings ${UTI}/testinput${c}.md and the usability findings ${UTI}/usability${c}.md. Also skim ${UTI}/LEDGER.md to avoid redoing prior work.
2. Pick the highest-VALUE items you can fix CORRECTLY and SAFELY this cycle (bias: planner P0/P1 first, then high-leverage usability/simplification). Quality over quantity — 3 to 8 solid, complete fixes beat many shaky ones. Genuine root-cause fixes over band-aids. Verify DB schema before writing SQL (don't guess columns: docker exec demandproject-postgres-1 psql -U demand -d demand_mvp -c "\\d <table>"). For missing-MV 500s, degrade gracefully (return empty/neutral + honest warning, never 500) OR apply the real DDL from sql/ if it exists; never fabricate data.
3. STRICT TDD — NON-NEGOTIABLE. For EACH item, in this order:
   a. RED: FIRST write a test that encodes the item's acceptance criterion (backend: tests/api/ with make_pool/make_async_pool, httpx AsyncClient + ASGITransport; frontend: vitest). Run it and CONFIRM IT FAILS for the right reason — capture the red output (assertion/error). A test that passes before you've written the fix does not encode the bug — make it tighter.
   b. GREEN: write the MINIMAL implementation to make that test pass. Re-run: confirm green.
      - Backend: ~/.local/bin/uv run pytest tests/<file>::<test> -q   (then the touched file, then a broader run if quick)
      - Frontend: from ${ROOT}/frontend -> PATH="/opt/homebrew/bin:$PATH" /opt/homebrew/bin/node node_modules/.bin/vitest run --reporter=dot
   c. REFACTOR: clean up impl + test while green (naming, dedup, CLAUDE.md compliance). Re-run to stay green.
   d. LIVE-VERIFY: curl the previously-failing endpoint (500->200 / empty->data) or describe the UI behavior change. Backend edits hot-reload; allow uvicorn ~3s after a .py save before curling.
   e. Self-review your diff per CLAUDE.md. Only set acceptanceMet=true if red-then-green happened AND live-verify passed. In redGreenEvidence, record the test name, the RED failure message, and the GREEN result — this is required per fixed item; an item without a real red phase is NOT done.
   Do not skip the red phase to save time. If you cannot write a failing test first for an item, DEFER it (note why) rather than fix it untested.
4. DOCUMENT: write ${cycleDir}/fixes-applied.md (include the red→green evidence per item) (per item: id, what was wrong, fix = files + 1-line each, verification = test/curl before->after, acceptance met?; plus Deferred and Risk/notes sections). Then APPEND a concise block to ${UTI}/LEDGER.md titled "## Cycle ${c}" listing each item id -> FIXED/DEFERRED with a 1-line note (create the file with a "# Improvement Ledger" header if it does not exist).
Do NOT commit — leave changes in the working tree.
Return the structured object: cycle=${c}, fixed[], deferred[], testsPass (did your test runs pass, ignoring pre-existing unrelated failures), remainingActionable = count of NEW P0/P1/P2 issues you did NOT fix this cycle, notes.`,
    { label: `fixer-c${c}`, phase: 'Fix', schema: FIXER_SCHEMA }
  )

  const fixedCount = fixer && fixer.fixed ? fixer.fixed.filter((f) => f.acceptanceMet).length : 0
  const remaining = (fixer && fixer.remainingActionable) || 0
  cycleSummaries.push({ cycle: c, plannerNew, usabilityNew, fixedCount, remaining, testsPass: !!(fixer && fixer.testsPass) })
  ledgerLine.push(`Cycle ${c}: ${newTotal} new issues, ${fixedCount} fixed & verified, ${remaining} remaining, tests ${fixer && fixer.testsPass ? 'PASS' : 'CHECK'}`)
  log(`Cycle ${c} done: ${fixedCount} fixed/verified, ${remaining} remaining.`)

  // --- dry-stop: no new substantive issues from EITHER reviewer ---
  if (newTotal === 0) {
    dryStreak++
    log(`No new actionable issues — dryStreak=${dryStreak}/${DRY_STOP}.`)
    if (dryStreak >= DRY_STOP) {
      log(`Stopping: ${DRY_STOP} consecutive dry cycles. Product is stable.`)
      break
    }
  } else {
    dryStreak = 0
  }
}

return {
  cyclesRun: cycleSummaries.length,
  stoppedEarly: dryStreak >= DRY_STOP,
  aborted,
  perCycle: cycleSummaries,
  ledger: ledgerLine,
}
