import { useEffect } from "react";

const VALID_TABS = ["commandCenter", "aggregateAnalysis", "overview", "explorer", "clusters", "itemAnalysis", "skuAnalysis", "accuracy", "inventory", "invBacktest", "intel", "jobs", "chat", "settings", "aiPlanner", "controlTower", "invPlanning", "storyboard", "exceptions", "sop", "customerMap", "fva", "dataQuality", "lgbmTuning", "sqlRunner"];
const ANALYTICS_TAB_DOMAINS = new Set(["sales", "forecast"]);
const DIMENSION_DOMAINS = ["item", "location", "customer", "time", "sku", "sales", "forecast"];

export function getInitialDomain(): string {
  const queryDomain = new URLSearchParams(window.location.search).get("domain");
  return (queryDomain || "item").toLowerCase();
}

const TAB_REDIRECTS: Record<string, string> = {
  skuAnalysis: "itemAnalysis",
  inventory: "itemAnalysis",
  overview: "aggregateAnalysis",
  accuracy: "aggregateAnalysis",
  aiPlanner: "commandCenter",
  controlTower: "commandCenter",
  storyboard: "commandCenter",
};

export function getInitialTab(): string {
  const params = new URLSearchParams(window.location.search);
  const urlTab = params.get("tab");
  if (urlTab && TAB_REDIRECTS[urlTab]) return TAB_REDIRECTS[urlTab];
  if (urlTab && VALID_TABS.includes(urlTab)) return urlTab;
  const d = getInitialDomain();
  if (ANALYTICS_TAB_DOMAINS.has(d)) return d;
  return DIMENSION_DOMAINS.includes(d) ? "commandCenter" : d;
}

export function updateUrlState(domain: string, tab: string) {
  const url = new URL(window.location.href);
  url.searchParams.set("domain", domain.toLowerCase());
  url.searchParams.set("tab", tab);
  window.history.pushState(null, "", url);
}

export function usePopstateSync(
  setActiveTab: (tab: string) => void,
  setDomain: (domain: string) => void,
) {
  useEffect(() => {
    const handler = () => {
      const params = new URLSearchParams(window.location.search);
      const urlTab = params.get("tab");
      const urlDomain = params.get("domain");
      if (urlTab && VALID_TABS.includes(urlTab)) setActiveTab(urlTab);
      if (urlDomain) setDomain(urlDomain.toLowerCase());
    };
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, [setActiveTab, setDomain]);
}

export function getScenarioJobParam(): string | null {
  return new URLSearchParams(window.location.search).get("scenario_job");
}

export function setScenarioJobParam(jobId: string | null) {
  const url = new URL(window.location.href);
  if (jobId) url.searchParams.set("scenario_job", jobId);
  else url.searchParams.delete("scenario_job");
  window.history.replaceState(null, "", url);
}

export { VALID_TABS, ANALYTICS_TAB_DOMAINS, DIMENSION_DOMAINS };
