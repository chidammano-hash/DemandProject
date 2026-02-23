# Feature 27 — Figma MCP Integration: Design-to-Code & Code-to-Design Workflow

## Overview

Integrate the **Figma MCP (Model Context Protocol) server** with the Planthium development workflow to establish a bidirectional design ↔ code pipeline. This enables:
- **Figma → Code:** Select any frame in Figma and generate production-ready React + Tailwind components via Claude Code
- **Code → Figma:** Capture the running Planthium UI and push it back into Figma as editable design frames
- **Design system sync:** Connect Figma components to codebase components via Code Connect, ensuring design tokens, colors, typography, and spacing stay in sync

## Problem

The current Planthium frontend was built entirely in code with no design artifacts:
- No Figma file exists for the UI — designers cannot review, iterate, or contribute to the interface
- The design system (3 themes, periodic table navigation, KPI cards, chart styles) lives only in CSS variables and Tailwind classes — not documented visually
- UI improvements require reading 2,728 lines of `App.tsx` — no visual reference for the intended layout
- Component inconsistencies (e.g., hardcoded colors, duplicated patterns) could have been caught earlier with a shared design source of truth
- Non-technical stakeholders (product, business) have no way to review or propose UI changes

## Architecture

### Connection Methods

Figma offers two MCP server modes. Both can be configured simultaneously.

#### Option A: Desktop Server (Local — Recommended for Development)

Runs locally via the Figma desktop app. No cloud auth required. Best for active development.

```
Figma Desktop App (Dev Mode enabled)
        ↓ serves
http://127.0.0.1:3845/mcp (local HTTP)
        ↓ consumed by
Claude Code (via `claude mcp add`)
```

**Requirements:**
- Figma desktop app (latest version)
- Dev or Full seat on a paid Figma plan
- Dev Mode enabled (Shift+D in Figma)

#### Option B: Remote Server (Cloud — Works Without Desktop App)

Connects to Figma's hosted endpoint. Requires OAuth authentication. Best for CI/CD or remote workflows.

```
Figma Cloud (https://mcp.figma.com/mcp)
        ↓ OAuth
Claude Code (via `claude mcp add`)
```

**Requirements:**
- Any Figma plan (including free)
- OAuth authentication flow on first use

### MCP Server Tools Available

Once connected, Claude Code gains access to these Figma MCP tools:

| Tool | Purpose | Server |
|------|---------|--------|
| `get_design_context` | Extract styling, layout, and component info from a Figma selection or link. Default output: React + Tailwind. | Both |
| `get_variable_defs` | Retrieve design variables (colors, spacing, typography tokens) from a Figma file | Both |
| `get_metadata` | Sparse XML of layer IDs, names, types, positions, sizes (low token cost) | Both |
| `get_screenshot` | Visual screenshot of a frame for layout accuracy during code generation | Both |
| `get_code_connect_map` | Retrieve existing Code Connect mappings (Figma node → code component) | Both |
| `add_code_connect_map` | Create new Code Connect mappings | Both |
| `get_code_connect_suggestions` | AI-suggested mappings of Figma components to code components | Both |
| `send_code_connect_mappings` | Confirm suggested Code Connect mappings | Both |
| `create_design_system_rules` | Generate a rules file for design-to-code translation context | Both |
| `generate_figma_design` | Push UI from Claude Code into Figma as editable frames (Code → Canvas) | Remote only |
| `get_figjam` | Convert FigJam diagrams to XML for architectural context | Both |
| `generate_diagram` | Create FigJam diagrams from Mermaid syntax | Both |
| `whoami` | Return authenticated user identity | Remote only |

---

## Implementation Plan

### Phase 1: MCP Server Setup

#### Step 1.1: Configure Desktop MCP Server

1. Open Figma desktop app → open/create the Planthium design file
2. Toggle to Dev Mode (Shift+D)
3. In the inspect panel, click **"Enable desktop MCP server"**
4. Server starts at `http://127.0.0.1:3845/mcp`

Register with Claude Code:
```bash
claude mcp add --transport http figma-desktop http://127.0.0.1:3845/mcp
```

#### Step 1.2: Configure Remote MCP Server (Optional)

```bash
claude mcp add --transport http --scope user figma https://mcp.figma.com/mcp
```

On first use, Claude Code will prompt OAuth authentication. Follow the redirect to authorize Figma access.

#### Step 1.3: Verify Connection

```bash
claude mcp list
# Should show: figma-desktop (or figma) with status "connected"
```

Inside Claude Code, run `/mcp` to verify the Figma server is listed and active.

### Phase 2: Capture Current UI into Figma (Code → Canvas)

Use the `generate_figma_design` tool (remote server) to push the running Planthium UI into Figma as editable frames. This creates the initial design artifact.

**Workflow:**
1. Start the Planthium dev server: `make ui` (localhost:5173)
2. In Claude Code (with remote MCP connected), prompt:
   - "Capture the current Planthium UI at localhost:5173 and create Figma frames for each tab"
3. Claude Code uses `generate_figma_design` to push the live UI into a Figma file
4. Repeat for each theme (Light, Dark, Midnight) to document all visual states

**Target frames to capture:**
| Frame | Tab | Theme | Content |
|-------|-----|-------|---------|
| Explorer - Light | Explorer | Light | Data table with filters, pagination, periodic table nav |
| Explorer - Dark | Explorer | Dark | Same layout, dark theme |
| Explorer - Midnight | Explorer | Midnight | Same layout, midnight theme |
| Accuracy - Light | Accuracy | Light | Model comparison table, lag curve chart, champion panel |
| DFU Analysis - Light | DFU Analysis | Light | Sales vs forecast overlay chart, KPI cards |
| Market Intel - Light | Intel | Light | Search results grid, narrative panel |
| Clusters - Light | Clusters | Light | Cluster table, PCA visualization |

### Phase 3: Establish Design Tokens in Figma

Map the existing CSS variables to Figma variables so that design tokens stay in sync.

#### Step 3.1: Create Figma Variable Collections

Create variable collections in Figma mirroring `index.css`:

**Color Primitives** (per-mode: Light / Dark / Midnight):
| Variable | Light | Dark | Midnight |
|----------|-------|------|----------|
| `background` | `hsl(220, 15%, 96%)` | `hsl(220, 15%, 10%)` | `hsl(230, 25%, 9%)` |
| `foreground` | `hsl(230, 30%, 14%)` | `hsl(220, 10%, 90%)` | `hsl(210, 20%, 88%)` |
| `card` | `hsl(0, 0%, 100%)` | `hsl(220, 15%, 13%)` | `hsl(230, 22%, 12%)` |
| `primary` | `hsl(230, 65%, 28%)` | `hsl(230, 55%, 55%)` | `hsl(200, 80%, 55%)` |
| `secondary` | `hsl(45, 80%, 92%)` | `hsl(45, 50%, 20%)` | `hsl(260, 40%, 20%)` |
| `muted` | `hsl(225, 18%, 93%)` | `hsl(220, 12%, 18%)` | `hsl(230, 20%, 15%)` |
| `accent` | `hsl(43, 90%, 55%)` | `hsl(43, 80%, 50%)` | `hsl(175, 70%, 45%)` |
| `border` | `hsl(225, 14%, 82%)` | `hsl(220, 12%, 22%)` | `hsl(230, 18%, 20%)` |
| `destructive` | `hsl(0, 72%, 51%)` | `hsl(0, 62%, 60%)` | `hsl(0, 55%, 65%)` |

**Chart Colors** (per-mode):
| Variable | Light | Dark | Midnight |
|----------|-------|------|----------|
| `chart-1` | `#4f46e5` | `#818cf8` | `#93c5fd` |
| `chart-2` | `#0d9488` | `#2dd4bf` | `#5eead4` |
| `chart-3` | `#d97706` | `#fbbf24` | `#fde68a` |
| `chart-4` | `#7c3aed` | `#a78bfa` | `#c4b5fd` |
| `chart-5` | `#dc2626` | `#fca5a5` | `#fda4af` |
| `chart-6` | `#0284c7` | `#7dd3fc` | `#a5f3fc` |

**Spacing** (single mode):
| Variable | Value |
|----------|-------|
| `radius-sm` | `0.3rem` |
| `radius-md` | `0.5rem` |
| `radius-lg` | `0.7rem` |

**Typography** (from `tailwind.config.ts`):
| Variable | Value |
|----------|-------|
| `font-sans` | `Avenir Next, Trebuchet MS, Segoe UI, sans-serif` |

#### Step 3.2: Validate Token Sync

Use `get_variable_defs` to retrieve Figma variables and compare against `index.css` values:
```
Claude: "Use the Figma MCP to get all variable definitions and compare them against our CSS variables in src/index.css"
```

### Phase 4: Code Connect Setup

Link Figma design components to the actual React source files so the MCP server can reference real code during design-to-code translation.

#### Step 4.1: Install Code Connect

```bash
cd mvp/demand/frontend
npm install --save-dev @figma/code-connect
```

#### Step 4.2: Create `figma.config.json`

```json
{
  "codeConnect": {
    "include": ["src/components/**/*.tsx"],
    "parser": "react",
    "paths": {
      "src/*": ["src/*"]
    }
  }
}
```

#### Step 4.3: Create Code Connect Files

For each reusable component, create a `.figma.tsx` file that maps Figma component properties to React props:

**`src/components/ElementTab.figma.tsx`**
```tsx
import figma from "@figma/code-connect";
import { ElementTab } from "./ElementTab";

figma.connect(ElementTab, "<FIGMA_COMPONENT_URL>", {
  props: {
    config: figma.enum("Element", {
      Explorer: 'ELEMENT_CONFIG["explorer"]',
      Clusters: 'ELEMENT_CONFIG["clusters"]',
      Accuracy: 'ELEMENT_CONFIG["accuracy"]',
    }),
    isActive: figma.boolean("Active"),
  },
  example: (props) => <ElementTab config={props.config} isActive={props.isActive} onClick={() => {}} />,
});
```

**`src/components/KpiCard.figma.tsx`**
```tsx
import figma from "@figma/code-connect";
import { KpiCard } from "./KpiCard";

figma.connect(KpiCard, "<FIGMA_COMPONENT_URL>", {
  props: {
    label: figma.string("Label"),
    value: figma.string("Value"),
    sublabel: figma.string("Sublabel"),
  },
  example: (props) => <KpiCard label={props.label} value={props.value} sublabel={props.sublabel} />,
});
```

**`src/components/LoadingElement.figma.tsx`**
```tsx
import figma from "@figma/code-connect";
import { LoadingElement } from "./LoadingElement";

figma.connect(LoadingElement, "<FIGMA_COMPONENT_URL>", {
  props: {
    overlay: figma.boolean("Overlay"),
    size: figma.enum("Size", { Small: "sm", Medium: "md" }),
  },
  example: (props) => <LoadingElement config={ELEMENT_CONFIG.explorer} overlay={props.overlay} size={props.size} message="Loading..." />,
});
```

Similarly for shadcn/ui primitives: `Button`, `Card`, `Badge`, `Input`, `Table`.

#### Step 4.4: Publish Mappings

```bash
npx figma connect publish
```

This uploads the Code Connect mappings to Figma. When designers inspect a component in Dev Mode, they'll see the exact React import and usage.

### Phase 5: Generate Design System Rules

Use the MCP tool to create a rules file that gives Claude Code persistent context about how to translate Figma designs into Planthium code:

```
Claude: "Use create_design_system_rules to generate rules for translating Figma designs into React + Tailwind code using our existing components and design tokens"
```

Save the output to `mvp/demand/frontend/.figma/rules/planthium.md` (this path is automatically read by the MCP agent on future prompts).

**Expected rules content should cover:**
- Use shadcn/ui primitives (`Button`, `Card`, `Badge`, `Input`, `Table`) for standard UI elements
- Use `ElementTab` for periodic table navigation tiles
- Use `KpiCard` for metric display cards
- Use `LoadingElement` for loading states
- Import colors from `@/constants/colors` and elements from `@/constants/elements`
- Use `cn()` from `@/lib/utils` for className merging
- Theme-aware styling: use semantic tokens (`bg-background`, `text-foreground`, `border-border`) not hardcoded colors
- Charts use Recharts (`LineChart`) with `CHART_COLORS[theme]` for grid/axis/tooltip colors
- All components must support 3 themes (light, dark, midnight) via CSS variable cascade

### Phase 6: Design-to-Code Workflow (Ongoing)

With all infrastructure in place, the day-to-day workflow becomes:

#### New Feature from Figma Design

```
1. Designer creates/updates frames in Figma
2. Developer selects the frame in Figma (or copies the link)
3. In Claude Code:
   "Implement the selected Figma frame as a React component.
    Use our existing design system (ElementTab, KpiCard, etc.)
    and follow the Planthium design rules."
4. Claude Code calls get_design_context → reads layout + styles
5. Claude Code calls get_code_connect_map → finds matching components
6. Claude Code generates React + Tailwind code using existing components
7. Developer reviews and commits
```

#### Updating Figma After Code Changes

```
1. Developer implements UI changes in code
2. Start dev server: make ui
3. In Claude Code:
   "Capture the updated DFU Analysis tab and push it to the
    Planthium Figma file as a new frame"
4. Claude Code calls generate_figma_design → creates editable Figma frame
5. Designer reviews the frame in Figma
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `mvp/demand/frontend/figma.config.json` | Create | Code Connect configuration |
| `mvp/demand/frontend/src/components/ElementTab.figma.tsx` | Create | Code Connect mapping for ElementTab |
| `mvp/demand/frontend/src/components/KpiCard.figma.tsx` | Create | Code Connect mapping for KpiCard |
| `mvp/demand/frontend/src/components/LoadingElement.figma.tsx` | Create | Code Connect mapping for LoadingElement |
| `mvp/demand/frontend/src/components/ui/button.figma.tsx` | Create | Code Connect mapping for Button |
| `mvp/demand/frontend/src/components/ui/card.figma.tsx` | Create | Code Connect mapping for Card |
| `mvp/demand/frontend/src/components/ui/badge.figma.tsx` | Create | Code Connect mapping for Badge |
| `mvp/demand/frontend/.figma/rules/planthium.md` | Create | Design system rules for MCP agent context |
| `mvp/demand/frontend/package.json` | Modify | Add `@figma/code-connect` dev dependency |
| `docs/design-specs/feature27.md` | Create | This spec |
| `docs/design-specs/feature1.md` | Modify | Add Feature 27 to implemented features list |
| `CLAUDE.md` | Modify | Add Figma MCP commands and workflow docs |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `@figma/code-connect` | `^1.x` | Code Connect CLI + React bindings |

**External requirements:**
- Figma desktop app (latest) with Dev Mode enabled
- Figma account with Dev or Full seat (for desktop server)
- Figma file for Planthium (to be created)

## Common Commands (to add to Makefile)

```bash
# Figma Code Connect
make figma-publish     # Publish Code Connect mappings to Figma
make figma-validate    # Validate Code Connect files without publishing
```

**Makefile targets:**
```makefile
figma-publish:
	cd frontend && npx figma connect publish

figma-validate:
	cd frontend && npx figma connect parse --dry-run
```

## MCP Setup Commands

```bash
# Desktop server (requires Figma app running with Dev Mode MCP enabled)
claude mcp add --transport http figma-desktop http://127.0.0.1:3845/mcp

# Remote server (cloud, OAuth on first use)
claude mcp add --transport http --scope user figma https://mcp.figma.com/mcp

# Verify
claude mcp list

# Inside Claude Code
/mcp  # Check server status
```

## Figma File Structure (Recommended)

```
Planthium Design System
├── 📄 Cover Page
├── 🎨 Design Tokens
│   ├── Colors (Light / Dark / Midnight modes)
│   ├── Typography
│   ├── Spacing & Radius
│   └── Shadows & Effects
├── 🧩 Components
│   ├── ElementTab (periodic table tile)
│   ├── KpiCard (metric display)
│   ├── LoadingElement (chemistry loader)
│   ├── Button (shadcn variants)
│   ├── Card (shadcn)
│   ├── Badge (shadcn)
│   ├── Input (shadcn)
│   └── Table (shadcn)
├── 📱 Pages
│   ├── Explorer (Light / Dark / Midnight)
│   ├── Clusters (Light / Dark / Midnight)
│   ├── DFU Analysis (Light / Dark / Midnight)
│   ├── Accuracy (Light / Dark / Midnight)
│   ├── Market Intel (Light / Dark / Midnight)
│   └── Chat Panel (Light / Dark / Midnight)
└── 🔄 Flows
    ├── Tab Navigation
    ├── Data Filtering
    └── Theme Switching
```

## Testing & Validation

### MCP Connection
1. Run `claude mcp list` — verify `figma-desktop` or `figma` shows as connected
2. Inside Claude Code, run `/mcp` and confirm the Figma server appears
3. Select a frame in Figma and prompt: "Describe the selected Figma frame" — should return layout data

### Code Connect
1. Run `npx figma connect parse --dry-run` — should parse all `.figma.tsx` files without errors
2. Run `npx figma connect publish` — should upload mappings to Figma
3. In Figma Dev Mode, inspect a connected component — should show React import + usage code

### Design-to-Code Round Trip
1. Select a simple frame (e.g., a KpiCard) in Figma
2. Prompt Claude Code: "Implement this Figma selection as a React component"
3. Verify the generated code uses `KpiCard` from `@/components/KpiCard` (not a new custom component)
4. Verify it uses semantic Tailwind classes (`bg-card`, `text-foreground`) not hardcoded colors

### Code-to-Design Round Trip
1. Start `make ui` (localhost:5173)
2. Prompt Claude Code: "Capture the Explorer tab and push it to Figma"
3. Open the Figma file — verify a new frame was created with the Explorer layout
4. Verify the frame is editable (layers are selectable, text is editable, not a flat image)

### Token Sync Validation
1. Prompt: "Use get_variable_defs to list all Figma color variables"
2. Compare output against `src/index.css` `:root` / `.dark` / `.midnight` values
3. All values should match — any drift indicates a sync issue

## Limitations & Known Constraints

1. **`generate_figma_design` is remote-only** — the Code → Canvas feature requires the remote MCP server (cloud), not the desktop server
2. **Recharts charts render as images** — the `generate_figma_design` tool captures live UI, but interactive chart internals (SVG paths) may flatten into images rather than editable vector shapes
3. **Code Connect requires component URLs** — each `.figma.tsx` file needs the actual Figma component URL, which can only be obtained after the Figma file is created
4. **No hot-reload sync** — changes in Figma are not automatically reflected in code (or vice versa). The workflow is manual: select frame → prompt → generate
5. **Desktop server requires Figma open** — the local MCP server only runs while the Figma desktop app is open with Dev Mode enabled
6. **Paid Figma plan required for desktop server** — the local server requires a Dev or Full seat. The remote server works on all plans (including free) but with OAuth

## Future Enhancements (Out of Scope for Feature 27)

1. **Automated token sync pipeline** — GitHub Action to diff Figma variables vs CSS variables on every PR
2. **Storybook integration** — Code Connect supports Storybook; adding Storybook to the project would provide interactive component documentation alongside Figma
3. **Figma plugin for theme preview** — custom plugin to preview all 3 themes side-by-side in Figma
4. **Design lint CI** — validate that new components use Figma variables (not hardcoded colors) before merge
5. **FigJam architecture diagrams** — use `generate_diagram` to push Mermaid architecture diagrams from code docs into FigJam automatically
