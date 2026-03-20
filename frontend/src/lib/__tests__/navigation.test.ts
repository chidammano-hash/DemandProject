import { describe, it, expect, vi, beforeEach } from "vitest";
import { navigateToItem } from "../navigation";

describe("navigateToItem", () => {
  beforeEach(() => {
    // Reset URL
    window.history.replaceState({}, "", "/");
  });

  it("sets tab=itemAnalysis and item param", () => {
    const onNav = vi.fn();
    navigateToItem(onNav, "100320");
    expect(onNav).toHaveBeenCalledWith("itemAnalysis");
    expect(window.location.search).toContain("tab=itemAnalysis");
    expect(window.location.search).toContain("item=100320");
  });

  it("includes location param when provided", () => {
    const onNav = vi.fn();
    navigateToItem(onNav, "100320", "1401-BULK");
    expect(window.location.search).toContain("item=100320");
    expect(window.location.search).toContain("loc=1401-BULK");
  });

  it("does not include loc param when location is undefined", () => {
    const onNav = vi.fn();
    navigateToItem(onNav, "100320");
    expect(window.location.search).not.toContain("loc=");
  });

  it("calls onNavigate exactly once", () => {
    const onNav = vi.fn();
    navigateToItem(onNav, "200100", "5501-MAIN");
    expect(onNav).toHaveBeenCalledTimes(1);
  });
});
