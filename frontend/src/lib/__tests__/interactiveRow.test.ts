import { describe, it, expect, vi } from "vitest";
import { interactiveRowProps } from "@/lib/interactiveRow";

/**
 * U6.2 — clickable drill-in rows must be keyboard-operable. `interactiveRowProps`
 * mirrors the `togglePillClass`/`severityBadgeClass` shared-helper pattern: it
 * returns the accessibility props (role/tabIndex) plus click + key handlers so a
 * keyboard user can activate the row with Enter or Space, same as a mouse click.
 */
function fakeKey(key: string) {
  return {
    key,
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent;
}

describe("interactiveRowProps (U6.2)", () => {
  it("exposes button role and tab focusability", () => {
    const props = interactiveRowProps(vi.fn());
    expect(props.role).toBe("button");
    expect(props.tabIndex).toBe(0);
  });

  it("onClick fires the activate handler", () => {
    const onActivate = vi.fn();
    interactiveRowProps(onActivate).onClick();
    expect(onActivate).toHaveBeenCalledTimes(1);
  });

  it("Enter activates the same handler as click and prevents default", () => {
    const onActivate = vi.fn();
    const ev = fakeKey("Enter");
    interactiveRowProps(onActivate).onKeyDown(ev);
    expect(onActivate).toHaveBeenCalledTimes(1);
    expect(ev.preventDefault).toHaveBeenCalled();
  });

  it("Space activates the handler", () => {
    const onActivate = vi.fn();
    interactiveRowProps(onActivate).onKeyDown(fakeKey(" "));
    expect(onActivate).toHaveBeenCalledTimes(1);
  });

  it("other keys do NOT activate (so arrow/tab navigation still works)", () => {
    const onActivate = vi.fn();
    const props = interactiveRowProps(onActivate);
    props.onKeyDown(fakeKey("ArrowDown"));
    props.onKeyDown(fakeKey("a"));
    props.onKeyDown(fakeKey("Tab"));
    expect(onActivate).not.toHaveBeenCalled();
  });
});
