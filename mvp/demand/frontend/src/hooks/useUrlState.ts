import { useCallback, useEffect, useState } from "react";

const VALID_TABS = ["explorer", "clusters", "dfuAnalysis", "accuracy", "intel", "inventory"];
const ANALYTICS_TAB_DOMAINS = new Set(["sales", "forecast"]);
const DIMENSION_DOMAINS = ["item", "location", "customer", "time", "dfu", "sales", "forecast"];

export function getInitialDomain(): string {
  const queryDomain = new URLSearchParams(window.location.search).get("domain");
  return (queryDomain || "item").toLowerCase();
}

export function getInitialTab(): string {
  const params = new URLSearchParams(window.location.search);
  const urlTab = params.get("tab");
  if (urlTab && VALID_TABS.includes(urlTab)) return urlTab;
  const d = getInitialDomain();
  if (ANALYTICS_TAB_DOMAINS.has(d)) return d;
  return DIMENSION_DOMAINS.includes(d) ? "explorer" : d;
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

export { VALID_TABS, ANALYTICS_TAB_DOMAINS, DIMENSION_DOMAINS };
