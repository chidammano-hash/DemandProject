# Feature 35 — Configurable Multi-Theme System

**Status:** Removed (motifs stripped; simplified to single professional theme with light/dark modes)
**Priority:** Enhancement
**Dependencies:** Feature 22 (dark mode + midnight theme), Feature 28 (UI architecture)

> **Note (2026-02):** The 5 motif themes (periodic, spirits, space, f1, zen) and the extra product themes (Wine & Spirits, Obsidian) were removed after critic feedback identified them as unnecessary complexity. The system was simplified to a single "Demand Studio" professional theme with light/dark modes only. All motif-related files, hooks, context, types, and test files were deleted. The `ThemeSelector` was reduced to a light/dark toggle. See `CLAUDE.md` for current architecture.

---

## 1. Overview

Demand Studio's current "Periodic Table of Elements" theme gives each tab a chemistry-inspired tile with symbol, atomic number, and name. This feature introduces a **configurable multi-motif system** that allows users to switch between 5 distinct visual themes while preserving the existing light/dark/midnight color modes as an orthogonal axis.

### 5 Themes

| # | Motif ID | Display Name | Concept |
|---|----------|-------------|---------|
| 1 | `periodic` | Periodic Table | Chemistry-inspired element tiles (current default) |
| 2 | `spirits` | Cellar Collection | Hospitality — spirits & wine tasting cards |
| 3 | `space` | Deep Space | NASA Mission Control — celestial navigation |
| 4 | `f1` | Formula 1 | Motorsport telemetry & race engineering |
| 5 | `zen` | Zen Garden | Japanese karesansui — contemplative minimalism |

Each theme provides: per-tab tile identity (symbol, label, colors, glow), a custom loading animation, app chrome (name, tagline, logo), and theme-aware adaptations for light/dark/midnight modes.

---

## 2. Current System Analysis

### 2.1 Element Configuration

The single source of truth is `src/constants/elements.ts`:

```typescript
export const ELEMENT_CONFIG: Record<string, {
  symbol: string; number: number; name: string;
  color: string; activeColor: string; glow: string;
}> = {
  explorer: { symbol: "Dx", number: 1,  name: "Explorer",  color: "bg-pink-50/90 ...", ... },
  clusters: { symbol: "Cl", number: 2,  name: "Clusters",  color: "bg-emerald-50/90 ...", ... },
  // ... 12 entries total (5 main tabs + 7 domain sub-tabs)
};
```

### 2.2 Components That Consume Theme Data

| Component | File | Usage |
|---|---|---|
| `ElementTab` | `src/components/ElementTab.tsx` | Tab buttons with symbol/number/name |
| `LoadingElement` | `src/components/LoadingElement.tsx` | Loading overlay tile with pulse-glow |
| `App.tsx` | `src/App.tsx` | Header title "Planthium", tab bar |
| `useTheme` | `src/hooks/useTheme.ts` | Color mode (light/dark/midnight) — orthogonal |

### 2.3 Loading Animation

The existing `pulse-glow` animation in `tailwind.config.ts`:
```javascript
animation: { 'pulse-glow': 'pulseGlow 2s ease-in-out infinite' }
```

### 2.4 Key Insight

Color mode (`useTheme`) and visual motif are **orthogonal** — any of 5 themes works with any of 3 color modes = 15 combinations. Both persist independently to localStorage.

---

## 3. TypeScript Type System

### 3.1 Core Tile Config (`src/types/motif.ts`)

```typescript
export type TileVariant = "periodic" | "card" | "badge" | "emblem";

export interface TileConfigBase {
  /** Big centered glyph — element symbol, emoji, kanji, letter */
  primary: string;
  /** Top-right superscript — atomic number, year, position */
  superscript: string | number;
  /** Bottom label below the glyph */
  label: string;
  /** Tailwind classes for inactive state (bg, text, border) */
  restClasses: string;
  /** Tailwind classes for active state */
  activeClasses: string;
  /** Tailwind shadow class for glow effect */
  glowClass: string;
  /** One-line flavor text for tooltip/subtitle */
  tagline?: string;
}

export interface TileConfig extends TileConfigBase {
  variant: TileVariant;
}
```

### 3.2 Loading Animation Config

```typescript
export type AnimationName =
  | "pulse-glow"     // periodic: indigo box-shadow pulse
  | "orbit-spin"     // space: orbiting ring
  | "flame-flicker"  // f1: red/orange brightness flicker
  | "zen-breathe"    // zen: slow opacity scale
  | "pour-shimmer"   // spirits: shimmer sweep
  ;

export interface LoadingAnimationConfig {
  animationName: AnimationName;
  wrapperClasses: string;
  statusLabel?: string;
}
```

### 3.3 App Chrome Config

```typescript
export interface MotifChrome {
  appName: string;       // "Planthium" | "Stellarium" | "The Cellar" | etc.
  appTagline: string;
  logoSvgPath: string | null;
  bgOverlay: string;     // CSS gradient or "none"
  tileRadius: string;    // "rounded-xl" | "rounded-full" | "rounded-sm"
}
```

### 3.4 Top-Level Motif Config

```typescript
export type MotifId = "periodic" | "spirits" | "space" | "f1" | "zen";

export interface MotifThemeConfig {
  id: MotifId;
  displayName: string;
  description: string;
  previewTile: TileConfig;
  tiles: Record<string, TileConfig>;
  loading: LoadingAnimationConfig;
  chrome: MotifChrome;
}
```

---

## 4. Theme Registry Architecture

### 4.1 Registry Pattern (`src/constants/motifRegistry.ts`)

```typescript
const registry = new Map<MotifId, MotifThemeConfig>();

export function registerMotif(config: MotifThemeConfig): void {
  registry.set(config.id, config);
}

export function getMotif(id: MotifId): MotifThemeConfig { ... }
export function getAllMotifs(): MotifThemeConfig[] { ... }
export const DEFAULT_MOTIF_ID: MotifId = "periodic";
```

### 4.2 Boot File (`src/constants/motifs/index.ts`)

```typescript
// Side-effect imports — each calls registerMotif() at module evaluation time
import "./periodicMotif";
import "./spaceMotif";
import "./spiritsMotif";
import "./f1Motif";
import "./zenMotif";

export { getMotif, getAllMotifs, DEFAULT_MOTIF_ID } from "@/constants/motifRegistry";
```

Adding a 6th theme in the future: create a new motif file, add one import line.

---

## 5. Theme 1 — Periodic Table (Default)

**Concept:** Chemistry-inspired element tiles. Classic Demand Studio look. Backward-compatible — the existing ELEMENT_CONFIG maps 1:1 to the new TileConfig shape.

**App Chrome:**
- App name: "Planthium"
- Tagline: "Periodic Analytics for Demand Forecasting"
- Animation: `pulse-glow` (existing)
- Tile radius: `rounded-xl`

**Tab Mappings (unchanged from current):**

| Tab | Symbol | Number | Color Family |
|-----|--------|--------|-------------|
| Explorer | Dx | 1 | Pink |
| Clusters | Cl | 2 | Emerald |
| DFU Analysis | Da | 7 | Teal |
| Accuracy | Ac | 5 | Purple |
| Market Intel | Mi | 6 | Cyan |
| Chat | Ch | 8 | Blue |
| Inventory | Iv | 9 | Amber |

---

## 6. Theme 2 — Cellar Collection (Hospitality / Spirits & Wine)

### 6.1 Concept & Philosophy

The "Cellar" theme transforms tabs into a curated spirits collection — each tab is a bottle pulled from a premium cellar, presented on a tasting card. The visual language borrows from high-end spirits packaging: embossed labels, wax seals, hand-lettered numerals, and warm amber of aged oak.

Where the periodic table conveys precision and scientific authority, the Cellar theme conveys *connoisseurship* — demand analysis as an art, requiring discernment, patience, and deep knowledge.

**App Chrome:**
- App name: "The Cellar"
- Tagline: "Curated Analytics, Aged to Perfection"
- Animation: `pour-shimmer` (liquid fill from bottom)
- Tile radius: `rounded-sm` (clean label corners)

### 6.2 Tab Mappings

#### Tab 1 — Explorer → Bourbon
*"Discovering new territory, one barrel at a time"*

| Property | Value |
|---|---|
| Symbol | `BRN` |
| Superscript | `86` (proof) |
| Background | `#2C1810` (deep charred oak) |
| Text | `#F5CBA7` (pale amber) |
| Border | `#8B4513` (saddle brown) |
| Glow | `rgba(210, 105, 30, 0.6)` (cognac glow) |

```
+─────────────────────+
│  86 proof            │
│                      │
│       B R N          │
│     ─────────        │
│      Bourbon         │
│                      │
│  "Discovering new    │
│   territory..."      │
+─────────────────────+
```

#### Tab 2 — Accuracy → Champagne
*"Every bubble, precisely placed"*

| Property | Value |
|---|---|
| Symbol | `CMG` |
| Superscript | `2019` (vintage year) |
| Background | `#1A1A2E` (midnight cellar) |
| Text | `#F7E98E` (pale gold) |
| Border | `#C9A84C` (aged gold foil) |
| Glow | `rgba(218, 165, 32, 0.55)` (golden effervescence) |

#### Tab 3 — DFU Analysis → Islay Scotch
*"Complexity rewarded by patience"*

| Property | Value |
|---|---|
| Symbol | `SCT` |
| Superscript | `18 yr` |
| Background | `#1C1C1C` (dark peat) |
| Text | `#C8B89A` (aged parchment) |
| Border | `#4A5240` (weathered sage) |
| Glow | `rgba(100, 120, 80, 0.5)` (peaty green smoke) |

#### Tab 4 — Clusters → Wine Flight
*"Grouped by character, distinguished by nuance"*

| Property | Value |
|---|---|
| Symbol | `FLT` |
| Superscript | `5 pours` |
| Background | `#2A0D1A` (deep burgundy) |
| Text | `#F4A4C0` (rosé pink) |
| Border | `#8B1A4A` (claret red) |
| Glow | `rgba(180, 40, 90, 0.55)` (ruby glow) |

#### Tab 5 — Market Intel → Hendrick's Gin
*"Botanical clarity in a complex world"*

| Property | Value |
|---|---|
| Symbol | `GIN` |
| Superscript | `44` (ABV) |
| Background | `#0D1A14` (deep forest) |
| Text | `#A8D8B8` (gin-clear pale green) |
| Border | `#2A6B4A` (botanicals green) |
| Glow | `rgba(80, 180, 120, 0.5)` (botanical glow) |

#### Tab 6 — Chat → Absinthe
*"The green fairy answers all questions"*

| Property | Value |
|---|---|
| Symbol | `ABS` |
| Superscript | `68` (ABV) |
| Background | `#0A1A0F` (dark wormwood) |
| Text | `#7AE87A` (electric green) |
| Border | `#1A6B2A` (bright emerald) |
| Glow | `rgba(80, 220, 80, 0.65)` (lurid green — intentionally intense) |

#### Tab 7 — Inventory → Cellar Reserve
*"What is stored, aged, and awaiting its moment"*

| Property | Value |
|---|---|
| Symbol | `RSV` |
| Superscript | `1982` (legendary vintage) |
| Background | `#1A1208` (dark mahogany) |
| Text | `#E8C870` (dusty gold) |
| Border | `#6B4C1A` (antique brass) |
| Glow | `rgba(180, 140, 40, 0.5)` (candlelight) |

### 6.3 Loading Animation — The Pour

```css
@keyframes cellar-pour {
  0%   { transform: translateY(100%); opacity: 0; }
  10%  { opacity: 1; }
  85%  { transform: translateY(8%); opacity: 1; }
  100% { transform: translateY(5%); opacity: 0.9; }
}

@keyframes pour-shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
```

Liquid fills the tile from bottom up. Each spirit has its own liquid gradient color. Champagne adds rising bubble dots.

### 6.4 Color Mode Adaptations

| Mode | Adaptation |
|---|---|
| **Light** | Background: warm parchment `#f8f4ec`. Tiles float on cream. Text warmth increases. |
| **Dark** | Deep cellar. Mahogany and charred oak. Glow from candlelight amber. Native mode. |
| **Midnight** | Pure black. Only spirit labels and glow visible. Like tasting in a cave. |

### 6.5 Typography

| Role | Font |
|---|---|
| Symbol (BRN, CMG) | `Playfair Display` (serif, old-world) |
| Spirit Name | `Cormorant Garamond` (elegant, wine-label) |
| Tab Label | `Inter` |
| Tagline | `Cormorant Garamond` italic, 0.6 opacity |

---

## 7. Theme 3 — Deep Space (NASA Mission Control)

### 7.1 Concept

Demand Studio becomes Mission Control — a high-stakes operations center where each tab is a celestial body or space program mission. The palette shifts from warm NASA amber to electric SpaceX blues. Data feels cosmic in scale. Every percentage point of accuracy is orbital mechanics.

**App Chrome:**
- App name: "Stellarium"
- Tagline: "Deep Space Demand Navigation"
- Animation: `orbit-spin` (ring orbiting tile)
- Tile radius: `rounded-xl`
- Background overlay: `radial-gradient(ellipse at 50% 0%, rgba(56,189,248,0.06) 0%, transparent 60%)`

### 7.2 Tab Mappings

| Tab | Celestial Body | Symbol | Designation | Tagline |
|-----|---------------|--------|------------|---------|
| Explorer | Mars | ♂ | 04 | "Every dataset is unexplored terrain" |
| Accuracy | Saturn | ♄ | 06 | "Perfect rings require perfect math" |
| DFU Analysis | Europa | Eu | J2 | "The signal lives beneath the surface" |
| Clusters | Pleiades | ✦✦ | M45 | "Gravity finds what logic misses" |
| Market Intel | Voyager | V1 | AU | "Signal from 23.9 billion km out" |
| Chat | Apollo | ⌘ | 11 | "Houston, we have a question" |
| Inventory | Ceres | ⚳ | 01 | "Everything stored. Nothing wasted." |

**Colors:** All tiles use `bg-slate-900/90` base with distinct accent colors (sky, emerald, teal, purple, cyan, amber, orange) and `0.35` opacity glows.

```
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ ♂    04 │ │ ♄    06 │ │ Eu   J2 │ │ ✦✦  M45 │ │ V1   AU │ │ ⌘    11 │ │ ⚳    01 │
│  MARS   │ │ SATURN  │ │ EUROPA  │ │PLEIADES │ │ VOYAGER │ │ APOLLO  │ │  CERES  │
│Explorer │ │Accuracy │ │   DFU   │ │Clusters │ │ Market  │ │  Chat   │ │Inventory│
│229M km  │ │1.4B km  │ │628M km  │ │ 444 ly  │ │23.9B km │ │  1969   │ │414M km  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 7.3 Loading Animation — Orbital Ring

```css
@keyframes orbital-spin {
  0%   { transform: rotate(0deg) scale(1); opacity: 0.3; }
  50%  { transform: rotate(180deg) scale(1.15); opacity: 0.9; }
  100% { transform: rotate(360deg) scale(1); opacity: 0.3; }
}

@keyframes starfield-warp {
  0%   { background-size: 1px 1px; opacity: 0; }
  50%  { background-size: 3px 300px; opacity: 0.6; }
  100% { background-size: 1px 1px; opacity: 0; }
}
```

A dashed circular border orbits the active tile. Tile glow intensifies during load. Starfield warp background on tab switch.

### 7.4 Color Mode Adaptations

| Mode | Adaptation |
|---|---|
| **Light** | Star chart paper cream `#f0f4f8`. Tiles remain dark — space is always dark. |
| **Dark** | Full deep space. Pitch black. Glows at 0.7 opacity. Star particle CSS background. Native mode. |
| **Midnight** | Pure black `#000000`. Maximum glow 0.9 opacity. Orbital rings doubled brightness. |

### 7.5 Typography

| Role | Font |
|---|---|
| Body Symbol | `Space Mono` (monospaced, NASA telemetry) |
| Planet Name | `Orbitron` (geometric aerospace) |
| Tab Label | `Inter` |
| Distance | `Space Mono` (telemetry) |

---

## 8. Theme 4 — Formula 1 (Motorsport Racing)

### 8.1 Concept

Demand Studio becomes a Race Engineer's console — the nerve center where strategy, telemetry, and split-second decisions converge. The aesthetic draws from carbon fiber brutalism, real-time telemetry dashboards, and iconic F1 livery colors. Every percentage point of accuracy is a tenth on the timing sheet.

**App Chrome:**
- App name: "Podium"
- Tagline: "Race to Peak Forecast Performance"
- Animation: `flame-flicker` (red/orange brightness flicker)
- Tile radius: `rounded-sm` (sharp carbon fiber edges)
- Background texture: CSS-only carbon fiber weave pattern

### 8.2 Tab Mappings

| Tab | Racing Element | Symbol | Designation | Tagline |
|-----|---------------|--------|------------|---------|
| Explorer | Qualifying | Q3 | Lap | "One lap to find the limit" |
| Accuracy | Telemetry | TEL | kHz | "Zero tolerance for imprecision" |
| DFU Analysis | Pit Wall | PW | Eng | "The call that wins races" |
| Clusters | Starting Grid | P1 | Grid | "Position is earned, not assigned" |
| Market Intel | Race Strategy | STR | Lap | "The undercut starts 3 laps before you see it" |
| Chat | Team Radio | RDO | Ch1 | "Copy that. Box this lap." |
| Inventory | Tire Compound | TCp | C3 | "The right compound wins the race" |

```
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Q3  Lap │ │TEL  kHz │ │ PW  Eng │ │ P1 Grid │ │STR  Lap │ │RDO  Ch1 │ │TCp   C3 │
│QUALIFYING│ │TELEMETRY│ │ PIT WALL│ │GRID     │ │STRATEGY │ │  RADIO  │ │TIRE CPD │
│Explorer │ │Accuracy │ │DFU Anlys│ │Clusters │ │Mkt Intel│ │  Chat   │ │Inventory│
│Sec1·107%│ │1.2kHz   │ │Box Box  │ │20 Cars  │ │Soft·12Δ │ │900 MHz  │ │Soft·28°C│
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 8.3 Colors

| Tab | Background | Text | Border | Glow |
|-----|-----------|------|--------|------|
| Qualifying | `#1a0000` | `#ff1801` (Ferrari red) | `#8b0000` | `#ff1801` 0.65 |
| Telemetry | `#050a14` | `#00d4ff` (screen cyan) | `#0077aa` | `#00d4ff` 0.55 |
| Pit Wall | `#0d0d08` | `#f5c842` (McLaren process yellow) | `#b8860b` | `#ffd700` 0.5 |
| Starting Grid | `#0a0a0a` | `#e8e8e8` (grid white) | `#555555` | `#ffffff` 0.3 |
| Strategy | `#0d0a00` | `#ff7700` (McLaren papaya) | `#993d00` | `#ff6600` 0.6 |
| Team Radio | `#081408` | `#00ff41` (radio green) | `#006600` | `#00ff41` 0.5 |
| Tire Compound | `#100010` | `#ffe000` (Pirelli yellow) | `#aa9900` | `#ffdd00` 0.5 |

### 8.4 Loading Animation — Starting Lights

Five red circles appear one by one (like the 5 Halo start lights), then all extinguish simultaneously — "LIGHTS OUT, GO!"

```css
@keyframes start-light-on {
  0%   { background: #111; box-shadow: none; }
  100% { background: #ff1100; box-shadow: 0 0 12px 4px #ff0000; }
}

@keyframes start-light-off {
  0%   { background: #ff1100; box-shadow: 0 0 12px 4px #ff0000; }
  100% { background: #111; box-shadow: none; }
}

@keyframes carbon-shimmer {
  0%   { background-position: 0% 0%; }
  100% { background-position: 100% 100%; }
}
```

```
● ● ● ● ●   ← all 5 red (loading)
○ ○ ○ ○ ○   ← LIGHTS OUT → content loaded
```

### 8.5 Color Mode Adaptations

| Mode | Adaptation |
|---|---|
| **Light** | Pit lane concrete `#f5f5f5`. Tiles remain dark. Team color glows pop against light. |
| **Dark** | Full garage-at-night. Carbon fiber pattern visible. Native mode. |
| **Midnight** | Pure black. Maximum glow opacity. Radial gradient like headcam in a tunnel. |

### 8.6 Typography

| Role | Font |
|---|---|
| Symbol (Q3, TEL) | `Orbitron` Bold (timing board) |
| Racing Name | `Barlow Condensed` ExtraBold (F1 livery) |
| Sub-label | `Space Mono` (telemetry readout) |

---

## 9. Theme 5 — Zen Garden (Japanese Karesansui)

### 9.1 Concept

Demand Studio becomes a contemplative digital Karesansui (dry rock garden). Data flows like raked sand patterns, insights emerge like cherry blossoms. Three Japanese aesthetic principles guide the design: **Ma** (negative space), **Wabi-sabi** (beauty in imperfection), and **Shibui** (quiet sophistication). Colors are never saturated. Animation is never abrupt — it arrives like mist.

**App Chrome:**
- App name: "Ensō"
- Tagline: "Demand, Observed in Stillness"
- Animation: `zen-breathe` (slow opacity scale)
- Tile radius: `rounded-xl`
- Background overlay: subtle raked-sand CSS pattern

### 9.2 Tab Mappings

| Tab | Garden Element | Kanji | Number | Tagline |
|-----|---------------|-------|--------|---------|
| Explorer | Ishi (石) — Stone | 石 | 一 (1) | "The first stone sets the path" |
| Accuracy | Mizu (水) — Water | 水 | 二 (2) | "Water never argues with level" |
| DFU Analysis | Take (竹) — Bamboo | 竹 | 三 (3) | "Every node connects root to sky" |
| Clusters | Koke (苔) — Moss | 苔 | 四 (4) | "Groupings that emerge, not imposed" |
| Market Intel | Kaze (風) — Wind | 風 | 五 (5) | "The wind knows before the market does" |
| Chat | Tsuru (鶴) — Crane | 鶴 | 六 (6) | "One call. Precisely enough." |
| Inventory | Kura (蔵) — Storehouse | 蔵 | 七 (7) | "Everything in its season, its place" |

```
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ 石    一 │ │ 水    二 │ │ 竹    三 │ │ 苔    四 │ │ 風    五 │ │ 鶴    六 │ │ 蔵    七 │
│  ISHI   │ │  MIZU   │ │  TAKE   │ │  KOKE   │ │  KAZE   │ │  TSURU  │ │  KURA   │
│Explorer │ │Accuracy │ │   DFU   │ │Clusters │ │ Market  │ │  Chat   │ │Inventory│
│Stone · 1│ │Flow · 2 │ │Node · 3 │ │Patch · 4│ │Wind · 5 │ │Call · 6 │ │Hold · 7 │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 9.3 Colors

| Tab | Background | Text | Border | Glow |
|-----|-----------|------|--------|------|
| Ishi (Stone) | `#1a1814` | `#c8b89a` (limestone) | `#6b5c44` | `#c8b89a` 0.35 |
| Mizu (Water) | `#091820` | `#7eb8d4` (shallow water) | `#2a6080` | `#5ba3c9` 0.4 |
| Take (Bamboo) | `#0a1a0a` | `#8fbc5a` (fresh bamboo) | `#4a7a20` | `#7ab840` 0.4 |
| Koke (Moss) | `#0a1508` | `#5c8c50` (old moss) | `#3a5c30` | `#6aab58` 0.35 |
| Kaze (Wind) | `#18100a` | `#e8c4a0` (warm breeze) | `#9a6840` | `#d4a060` 0.4 |
| Tsuru (Crane) | `#0d0a12` | `#c8b4e8` (feather lavender) | `#7855a8` | `#a070d0` 0.4 |
| Kura (Storehouse) | `#14100a` | `#d4b896` (rice paper) | `#8c6840` | `#c89860` 0.35 |

### 9.4 Loading Animation — Enso Brush Stroke

An ink brush draws a circular Enso (禅) around the active tile symbol using SVG stroke-dashoffset animation. Cherry blossom petals drift during longer loads.

```css
@keyframes enso-draw {
  0%   { stroke-dashoffset: 314; opacity: 0; }
  20%  { opacity: 1; }
  100% { stroke-dashoffset: 0; opacity: 0.8; }
}

@keyframes stone-ripple {
  0%   { width: 20px; height: 20px; opacity: 0.8; }
  100% { width: 200px; height: 200px; opacity: 0; }
}

@keyframes petal-fall {
  0%   { transform: translateY(-20px) rotate(0deg); opacity: 0.9; }
  50%  { transform: translateY(40px) rotate(45deg) translateX(15px); opacity: 0.7; }
  100% { transform: translateY(100px) rotate(90deg) translateX(-5px); opacity: 0; }
}

@keyframes symbol-breathe {
  0%, 100% { opacity: 0.7; transform: scale(1); }
  50%      { opacity: 1.0; transform: scale(1.04); }
}
```

**Petal shape:** CSS `clip-path` ellipse — no images needed.

### 9.5 Color Mode Adaptations

| Mode | Adaptation |
|---|---|
| **Light** | Warm rice-paper `#f5f0e8`. Tiles become subtle ink-on-paper. Glows are mist-like. |
| **Dark** | Evening garden. Deep shadows. Moonlight glows. Raked sand visible as subtle lines. |
| **Midnight** | Near-black with only kanji and glow visible. Meditative void. |

### 9.6 Typography

| Role | Font |
|---|---|
| Kanji Symbol | `Noto Serif JP` (traditional brush aesthetic) |
| Element Name | `Cormorant Garamond` Light (elegant Roman for Romaji) |
| Tab Label | `Inter` Light (minimal, airy) |
| Tagline | `Cormorant Garamond` italic, 0.5 opacity |

---

## 10. Settings UI Panel

### 10.1 Placement

The existing Settings gear dropdown expands from `w-64` to `w-80` with two sections:
- **Section 1 (existing):** "Color mode" — Light / Dark / Midnight buttons
- **Section 2 (new):** "Theme style" — 5 motif cards in a 2-column grid

### 10.2 Layout Mockup

```
┌────────────────────────────────────┐  w-80 (320px)
│  Color mode                        │
│  ┌────────┐ ┌────────┐ ┌────────┐  │
│  │☀ Light │ │🌙 Dark │ │🌊 Midn.│  │
│  └────────┘ └────────┘ └────────┘  │
│                                    │
│  Theme style                       │
│  ┌──────────────┐ ┌──────────────┐ │
│  │ [TILE PREV]  │ │ [TILE PREV]  │ │
│  │ Periodic     │ │ Deep Space   │ │
│  │ Chemistry-…  │ │ Stellar nav… │ │
│  └──────────────┘ └──────────────┘ │
│  ┌──────────────┐ ┌──────────────┐ │
│  │ [TILE PREV]  │ │ [TILE PREV]  │ │
│  │ Cellar       │ │ Formula 1    │ │
│  │ Spirits and… │ │ Race to peak │ │
│  └──────────────┘ └──────────────┘ │
│  ┌──────────────┐                  │
│  │ [TILE PREV]  │                  │
│  │ Zen Garden   │                  │
│  │ Calm, minim… │                  │
│  └──────────────┘                  │
└────────────────────────────────────┘

Active card: ring-2 ring-primary border-primary
Each [TILE PREV] = read-only 48×48px version of previewTile
```

### 10.3 Interaction

- **Instant switch** — no apply button (consistent with existing color mode behavior)
- Motif selection persists to `localStorage` key `ds-motif`
- Switching motif triggers immediate re-render of all tiles, header, and loading overlays
- Color mode and motif are independent — any combination works

---

## 11. State Management

### 11.1 New `useMotifTheme` Hook

```typescript
// src/hooks/useMotifTheme.ts
const STORAGE_KEY = "ds-motif";

export function useMotifTheme(): {
  motifId: MotifId;
  motifConfig: MotifThemeConfig;
  setMotif: (id: MotifId) => void;
  getTile: (tabKey: string) => TileConfig;
}
```

- Reads from `localStorage` on mount, validates against registry
- Persists immediately on change
- Sets `data-motif` attribute on `<html>` for CSS selectors
- Falls back to `periodic` if stored value is invalid

### 11.2 Relationship with Existing `useTheme`

```
useTheme()      → controls light/dark/midnight (CSS classes on <html>)
useMotifTheme() → controls which tile/animation config is active (data-motif attr)
```

Completely independent. Both called at `App` level. No merging.

### 11.3 Context Provider

`MotifProvider` wraps the app so deeply-nested components (`LoadingElement` inside Suspense/ErrorBoundary) can access motif config without prop drilling:

```tsx
// App.tsx
<MotifProvider value={useMotifTheme()}>
  {/* ... tabs, suspense, error boundaries ... */}
</MotifProvider>
```

---

## 12. Component Architecture

### 12.1 Component Tree

```
App
├── MotifProvider (value = useMotifTheme())
│   ├── Header
│   │   ├── ElementTab × N   (reads tile from useMotif().getTile(tabKey))
│   │   └── SettingsDropdown
│   │       ├── ColorModePicker (existing)
│   │       └── MotifSettingsPanel (NEW)
│   │           └── MotifTilePreview × 5  (NEW, read-only)
│   │
│   ├── Suspense (fallback: LoadingElement reads motif from context)
│   ├── ErrorBoundary (fallback: LoadingElement from context)
│   │
│   └── Tab Content...
```

### 12.2 Modified Components

| Component | Change |
|---|---|
| `ElementTab` | Reads tile config from `useMotif()` context instead of `ELEMENT_CONFIG` |
| `LoadingElement` | Reads animation config from `useMotif()` context |
| `App.tsx` | Wraps with `MotifProvider`, reads `chrome.appName` for header |
| Settings dropdown | Adds `MotifSettingsPanel` section |

### 12.3 New Components

| Component | Purpose |
|---|---|
| `MotifSettingsPanel` | 2-column grid of motif selection cards |
| `MotifTilePreview` | Read-only 48x48 miniature tile for settings panel |
| `MotifProvider` | React Context provider for motif config |

### 12.4 New Files

| File | Purpose |
|---|---|
| `src/types/motif.ts` | Type definitions for motif system |
| `src/constants/motifRegistry.ts` | Map-based registry with register/get/getAll |
| `src/constants/motifs/index.ts` | Boot file importing all motif modules |
| `src/constants/motifs/periodicMotif.ts` | Periodic table motif config |
| `src/constants/motifs/spiritsMotif.ts` | Spirits & wine motif config |
| `src/constants/motifs/spaceMotif.ts` | Deep space motif config |
| `src/constants/motifs/f1Motif.ts` | Formula 1 motif config |
| `src/constants/motifs/zenMotif.ts` | Zen garden motif config |
| `src/hooks/useMotifTheme.ts` | Motif state management hook |
| `src/context/MotifContext.tsx` | React Context for motif |
| `src/components/MotifSettingsPanel.tsx` | Settings panel UI |
| `src/components/MotifTilePreview.tsx` | Preview tile component |

---

## 13. CSS & Tailwind Configuration

### 13.1 New Animations in `tailwind.config.ts`

```javascript
animation: {
  'pulse-glow':    'pulseGlow 2s ease-in-out infinite',      // existing
  'orbit-spin':    'orbitSpin 3s linear infinite',             // space
  'flame-flicker': 'flameFlicker 0.8s ease-in-out infinite',  // f1
  'zen-breathe':   'zenBreathe 4s ease-in-out infinite',       // zen
  'pour-shimmer':  'pourShimmer 2s ease-in-out infinite',      // spirits
}
```

### 13.2 CSS Custom Properties

```css
[data-motif="spirits"] {
  --motif-bg-overlay: linear-gradient(180deg, rgba(44,24,16,0.1) 0%, transparent 50%);
}
[data-motif="space"] {
  --motif-bg-overlay: radial-gradient(ellipse at 50% 0%, rgba(56,189,248,0.06), transparent 60%);
}
[data-motif="f1"] {
  --motif-bg-overlay: none;
}
[data-motif="zen"] {
  --motif-bg-overlay: none;
}
```

### 13.3 Google Fonts

Add to `index.html` or dynamically load per motif:
- **Spirits:** Playfair Display, Cormorant Garamond
- **Space:** Space Mono, Orbitron
- **F1:** Orbitron, Barlow Condensed, Space Mono
- **Zen:** Noto Serif JP, Cormorant Garamond

Lazy-load non-periodic fonts only when the motif is selected (avoid loading 4 extra font families upfront).

---

## 14. Keyboard Shortcut

Add `Ctrl+M` (or `Cmd+M` on Mac) to cycle through motifs, following the existing pattern for keyboard shortcuts in `useKeyboardShortcuts.ts`.

---

## 15. URL State

Add optional `motif` URL param via `useUrlState.ts` for shareable links:
```
?motif=spirits&theme=midnight
```

---

## 16. Testing Strategy

### 16.1 Unit Tests

| Test File | Tests |
|---|---|
| `motifRegistry.test.ts` | registerMotif, getMotif, getAllMotifs, unknown ID throws |
| `useMotifTheme.test.ts` | Default motif, localStorage persistence, invalid stored value fallback |
| `motifs/*.test.ts` | Each motif file registers correctly, has all required tile keys |

### 16.2 Component Tests

| Test File | Tests |
|---|---|
| `MotifSettingsPanel.test.tsx` | Renders all 5 motifs, click switches active, aria-pressed |
| `MotifTilePreview.test.tsx` | Renders primary/superscript/label, applies active classes |
| `ElementTab.test.tsx` | Reads from motif context, renders correct tile for each motif |
| `LoadingElement.test.tsx` | Uses motif animation config, displays correct status label |

### 16.3 Integration Tests

- Theme × Motif matrix: verify all 15 combinations render without errors
- Settings panel state synchronization with localStorage
- URL param `?motif=space` sets correct motif on load

### 16.4 Visual Regression

- Screenshot each of the 15 combinations (5 motifs × 3 color modes)
- Screenshot loading state for each motif's animation
- Screenshot settings panel with each motif active

---

## 17. Accessibility

- All motif tiles maintain WCAG AA contrast ratios
- `aria-label` on tab buttons includes tile label text
- `aria-pressed` on settings panel motif cards
- Keyboard navigation through settings panel via arrow keys
- `prefers-reduced-motion`: disable glow animations, use static borders instead
- Screen readers announce motif name on switch: "Theme changed to Deep Space"

---

## 18. Performance Considerations

- Motif configs are static data — zero runtime cost, tree-shakeable
- Only one motif's font families loaded at a time (lazy font loading)
- CSS animations use `transform` and `opacity` only (GPU-composited, no layout thrash)
- No image assets — all visual effects are CSS-only (gradients, clip-path, box-shadow)
- Settings panel renders 5 × 48px preview tiles — negligible overhead

---

## 19. Migration Strategy

1. **Phase 1:** Create type system and registry. Register `periodic` motif as exact translation of current `ELEMENT_CONFIG`. No visible change.
2. **Phase 2:** Add `MotifProvider` and refactor `ElementTab`/`LoadingElement` to read from context. Still only `periodic` available. Verify zero regression.
3. **Phase 3:** Add settings panel UI with motif selector. Still only `periodic` card, but UI is functional.
4. **Phase 4:** Add `spirits`, `space`, `f1`, `zen` motif configs one at a time. Each is a single file addition + one import line.
5. **Phase 5:** Add custom animations to `tailwind.config.ts` and `index.css`.
6. **Phase 6:** Add Google Fonts lazy loading.
7. **Phase 7:** Tests for all new components and hooks.

---

## 20. Future Extensions

- **Custom motif builder:** UI to create user-defined themes with color picker
- **Seasonal motifs:** Auto-switch based on date (Halloween, holidays)
- **Team/org themes:** Shared motifs via API for enterprise deployments
- **Animated transitions:** Smooth morph animation when switching between motifs
- **Sound effects:** Optional subtle audio on tab switch (per motif)

---

## Implementation Details

### Additional Types (`types/motif.ts`)
- `ColorMode = "light" | "dark" | "midnight"`
- `MotifPalette` interface with 31 CSS variable fields
- `MotifThemeConfig.palette` optional field: `Partial<Record<ColorMode, Partial<MotifPalette>>>`

### Hook Enhancements
- `useMotifTheme` returns `cycleMotif()` method
- `useMotifOptional()` safe variant in `MotifContext.tsx` — returns null instead of throwing outside provider

### Component Notes
- `MotifTilePreview` is inlined inside `MotifSettingsPanel.tsx` (not a separate component file)
- Ctrl+M cycles through motifs (not opens panel as stated in some docs)

### Test Files (implemented)
- `frontend/src/constants/__tests__/motifRegistry.test.ts`
- `frontend/src/hooks/__tests__/useMotifTheme.test.ts`
- `frontend/src/components/__tests__/MotifSettingsPanel.test.tsx`


---

## Examples

### Example: Single Demand Studio theme with light/dark

```typescript
// context/ThemeContext.tsx
import { useThemeContext } from '@/context/ThemeContext'

function MyComponent() {
  const { colorMode, toggleColorMode, theme } = useThemeContext()
  // theme.name = "Demand Studio"  (only one theme now)
  // colorMode = "light" | "dark"
  console.log(theme.palette.light['--bg-primary'])  // "#ffffff"
  console.log(theme.palette.dark['--bg-primary'])   // "#0f172a"
}
```

### Example: ThemeSelector component in sidebar footer

```typescript
// components/ThemeSelector.tsx — light/dark toggle only
import { useThemeContext } from '@/context/ThemeContext'
import { Sun, Moon } from 'lucide-react'

export function ThemeSelector() {
  const { colorMode, toggleColorMode } = useThemeContext()
  return (
    <button onClick={toggleColorMode} title="Toggle dark mode (d)">
      {colorMode === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  )
}
// Keyboard shortcut: press 'd' to toggle
```

### Example: CSS variable application at runtime

```typescript
// hooks/useTheme.ts — applies palette on colorMode change
function applyPalette(palette: Record<string, string>) {
  Object.entries(palette).forEach(([key, value]) => {
    document.documentElement.style.setProperty(key, value)
  })
}
// Dark mode: document.documentElement.classList.add('dark')
// Light mode: document.documentElement.classList.remove('dark')
```
