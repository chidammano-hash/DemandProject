/**
 * Breadcrumb label for the selected DFU on the Item Analysis tab (U3.5).
 *
 * The product is human-readable everywhere else (Demand History, the action
 * feeds), but the breadcrumb used to render a bare numeric code ("Item 185690").
 * When the description (`dim_item.item_desc`, surfaced by /sku/analysis) is
 * loaded, render "Item <id> — <desc>"; fall back to the bare code while it is
 * still loading or unavailable.
 */
export function itemBreadcrumbLabel(itemId: string, itemDesc?: string | null): string {
  const desc = itemDesc?.trim();
  if (!desc) return `Item ${itemId}`;
  // Some feeds prefix the id into the description; strip it to avoid
  // "Item 15502 — 15502 ...".
  const deduped = desc.startsWith(`${itemId} `) ? desc.slice(itemId.length + 1) : desc;
  return `Item ${itemId} — ${deduped}`;
}
