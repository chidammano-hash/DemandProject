// Per-page chat customization for the global chat drawer.
// Each tab maps to a focus blurb (sent to the agent as page context so answers
// are tailored to that page) and a few page-relevant suggested prompts.
// Spec: docs/specs/06-ai-platform/07-sku-chatbot.md

export interface PageChatConfig {
  title: string;
  focus: string; // sent to the backend as page_focus
  suggestions: string[];
}

const DEFAULT_CONFIG: PageChatConfig = {
  title: "Supply Chain Assistant",
  focus: "the Supply Chain Command Center (general assistance).",
  suggestions: ["What can you tell me about this SKU?", "Summarize the key risks here."],
};

// Keyed by the tab key used in App.tsx / useUrlState VALID_TABS.
const PAGE_CHAT_CONFIG: Record<string, PageChatConfig> = {
  commandCenter: {
    title: "Command Center",
    focus: "the Command Center — supply-chain health KPIs and the day's biggest risks.",
    suggestions: ["What are the biggest risks right now?", "Summarize portfolio health."],
  },
  aggregateAnalysis: {
    title: "Portfolio Analysis",
    focus:
      "the Portfolio / Aggregate Analysis page — forecast accuracy (WAPE, bias) across segments.",
    suggestions: ["Which segments own the most forecast error?", "Is bias trending up anywhere?"],
  },
  itemAnalysis: {
    title: "Item Analysis",
    focus:
      "the Item Analysis page — forecast vs actuals, accuracy, and inventory for the selected SKU.",
    suggestions: [
      "Why did the forecast miss last quarter?",
      "Is the safety stock right for this SKU?",
      "Compare this SKU to its cluster peers.",
    ],
  },
  fva: {
    title: "FVA & ROI",
    focus: "the FVA & ROI page — forecast value-add vs the naive baseline.",
    suggestions: ["Is the champion beating the baseline here?", "Where is FVA negative?"],
  },
  lgbmTuning: {
    title: "Forecasting",
    focus:
      "the Forecasting page — clustering, backtests, tuning, champion assignment, forecast release, and period roll.",
    suggestions: ["Why didn't the last tune improve WAPE?", "Summarize the best run."],
  },
  customerAnalytics: {
    title: "Customer Analytics",
    focus: "the Customer Analytics page — customer demand, segments, and concentration.",
    suggestions: [
      "Which customers drive this item's demand?",
      "Is there customer concentration risk?",
    ],
  },
  demandHistory: {
    title: "Demand History",
    focus: "the Demand History page — customer-level demand history and decomposition.",
    suggestions: ["Break down this SKU's demand by customer.", "Any demand trend break?"],
  },
  skuFeatures: {
    title: "SKU Features",
    focus:
      "the SKU Features page — demand-behavior features (seasonality, CV, intermittency, cluster).",
    suggestions: ["Describe this SKU's demand pattern.", "Why is it in this cluster?"],
  },
  invPlanning: {
    title: "Inventory Planning",
    focus: "the Inventory Planning page — safety stock, reorder points, EOQ, and exceptions.",
    suggestions: [
      "What's driving the stockout risk?",
      "Explain this SKU's safety stock.",
      "Is there excess inventory?",
    ],
  },
  invBacktest: {
    title: "Inventory Backtest",
    focus: "the Inventory Backtest page — how inventory policies performed historically.",
    suggestions: ["How did the policy perform?", "Where did service level dip?"],
  },
  sop: {
    title: "S&OP",
    focus: "the S&OP page — demand/supply review, gaps, and approvals.",
    suggestions: ["Summarize the open S&OP gaps.", "What still needs sign-off?"],
  },
  dataQuality: {
    title: "Data Quality",
    focus: "the Data Quality page — checks, exceptions, and scoring.",
    suggestions: ["What are the top data-quality issues?", "What should I fix first?"],
  },
};

/** Resolve the chat config for a tab (falls back to a sensible default). */
export function getPageChatConfig(tab: string): PageChatConfig {
  return PAGE_CHAT_CONFIG[tab] ?? DEFAULT_CONFIG;
}

// Tabs where the global chat drawer should NOT appear (the standalone SKU Chat
// tab is already a full-page chat).
export const CHAT_HIDDEN_TABS = new Set<string>(["skuChat", "customerAnalytics"]);
