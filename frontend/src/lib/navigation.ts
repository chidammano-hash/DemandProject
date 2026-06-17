/**
 * Helpers for deep-linking between tabs with context.
 */

/** Navigate to Item Analysis with item context via URL state */
export function navigateToItem(
  onNavigate: (tab: string) => void,
  itemNo: string,
  location?: string
) {
  const url = new URL(window.location.href);
  url.searchParams.set("tab", "itemAnalysis");
  if (itemNo) url.searchParams.set("item", itemNo);
  if (location) url.searchParams.set("loc", location);
  window.history.pushState({}, "", url.toString());
  onNavigate("itemAnalysis");
}

/**
 * Navigate to a top-level tab (optionally switching domain) without a callback.
 * Updates the URL and dispatches `popstate`, which the app's `usePopstateSync`
 * picks up to switch tab + domain. Use when the caller has no `onNavigate` prop.
 */
export function navigateToTab(tab: string, domain?: string) {
  const url = new URL(window.location.href);
  if (domain) url.searchParams.set("domain", domain);
  url.searchParams.set("tab", tab);
  window.history.pushState({}, "", url.toString());
  window.dispatchEvent(new PopStateEvent("popstate"));
}
