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
