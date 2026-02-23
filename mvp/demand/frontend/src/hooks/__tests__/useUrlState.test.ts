import { describe, it, expect, beforeEach } from "vitest";
import {
  getInitialDomain,
  getInitialTab,
  updateUrlState,
  VALID_TABS,
  DIMENSION_DOMAINS,
} from "@/hooks/useUrlState";

describe("useUrlState", () => {
  beforeEach(() => {
    // Reset URL to clean state
    window.history.replaceState(null, "", "/");
  });

  describe("getInitialDomain", () => {
    it("defaults to item when no query param", () => {
      expect(getInitialDomain()).toBe("item");
    });

    it("reads domain from URL", () => {
      window.history.replaceState(null, "", "/?domain=location");
      expect(getInitialDomain()).toBe("location");
    });

    it("lowercases domain", () => {
      window.history.replaceState(null, "", "/?domain=SALES");
      expect(getInitialDomain()).toBe("sales");
    });
  });

  describe("getInitialTab", () => {
    it("defaults to explorer when no params", () => {
      expect(getInitialTab()).toBe("explorer");
    });

    it("reads tab from URL if valid", () => {
      window.history.replaceState(null, "", "/?tab=accuracy");
      expect(getInitialTab()).toBe("accuracy");
    });

    it("ignores invalid tab", () => {
      window.history.replaceState(null, "", "/?tab=bogus");
      expect(getInitialTab()).toBe("explorer");
    });

    it("returns domain for analytics domains", () => {
      window.history.replaceState(null, "", "/?domain=sales");
      expect(getInitialTab()).toBe("sales");
    });
  });

  describe("updateUrlState", () => {
    it("sets domain and tab in URL", () => {
      updateUrlState("location", "clusters");
      const params = new URLSearchParams(window.location.search);
      expect(params.get("domain")).toBe("location");
      expect(params.get("tab")).toBe("clusters");
    });

    it("lowercases domain", () => {
      updateUrlState("ITEM", "explorer");
      const params = new URLSearchParams(window.location.search);
      expect(params.get("domain")).toBe("item");
    });
  });

  describe("exported constants", () => {
    it("VALID_TABS has 5 entries", () => {
      expect(VALID_TABS).toHaveLength(5);
      expect(VALID_TABS).toContain("explorer");
      expect(VALID_TABS).toContain("accuracy");
    });

    it("DIMENSION_DOMAINS has 7 entries", () => {
      expect(DIMENSION_DOMAINS).toHaveLength(7);
    });
  });
});
