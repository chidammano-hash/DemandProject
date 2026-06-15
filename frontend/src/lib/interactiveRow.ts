/**
 * U6.2 — keyboard-accessible drill-in table rows.
 *
 * Many data tables make a whole `<tr>` clickable to drill into a detail pane,
 * but a bare `<tr onClick>` with only `cursor-pointer` is invisible to keyboard
 * and screen-reader users: it is not focusable and announces as a plain row.
 *
 * `interactiveRowProps(onActivate)` returns the props to spread onto such a row
 * so it becomes operable: `role="button"`, `tabIndex={0}`, and an `onKeyDown`
 * that fires the same handler as a click on Enter or Space (mirroring native
 * button activation). Use it like:
 *
 *   <tr {...interactiveRowProps(() => select(id))} className="cursor-pointer …">
 *
 * Shared helper in the `togglePillClass` / `severityBadgeClass` family.
 */
export interface InteractiveRowProps {
  role: "button";
  tabIndex: 0;
  onClick: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
}

export function interactiveRowProps(onActivate: () => void): InteractiveRowProps {
  return {
    role: "button",
    tabIndex: 0,
    onClick: onActivate,
    onKeyDown: (e: React.KeyboardEvent) => {
      // Enter and Space activate, matching native <button> semantics. Other
      // keys (arrows, Tab, typing) pass through so normal navigation works.
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onActivate();
      }
    },
  };
}
