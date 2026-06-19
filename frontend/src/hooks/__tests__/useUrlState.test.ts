import { describe, it, expect, beforeEach } from "vitest";
import {
  getInitialDomain,
  getInitialTab,
  resolveTab,
  updateUrlState,
  getScenarioJobParam,
  setScenarioJobParam,
  VALID_TABS,
  DIMENSION_DOMAINS,
} from "@/hooks/useUrlState";
import { NAV_ITEMS } from "@/components/AppSidebar";

describe("useUrlState", () => {
  beforeEach(() => {
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
    it("defaults to commandCenter when no params", () => {
      expect(getInitialTab()).toBe("commandCenter");
    });

    it("redirects overview to aggregateAnalysis", () => {
      window.history.replaceState(null, "", "/?tab=overview");
      expect(getInitialTab()).toBe("aggregateAnalysis");
    });

    it("redirects accuracy to aggregateAnalysis", () => {
      window.history.replaceState(null, "", "/?tab=accuracy");
      expect(getInitialTab()).toBe("aggregateAnalysis");
    });

    it("ignores invalid tab", () => {
      window.history.replaceState(null, "", "/?tab=bogus");
      expect(getInitialTab()).toBe("commandCenter");
    });

    it("returns domain for analytics domains", () => {
      window.history.replaceState(null, "", "/?domain=sales");
      expect(getInitialTab()).toBe("sales");
    });

    it("accepts aggregateAnalysis tab from URL", () => {
      window.history.replaceState(null, "", "/?tab=aggregateAnalysis");
      expect(getInitialTab()).toBe("aggregateAnalysis");
    });

    it("accepts demandHistory tab from URL (U2.2 deep-link / refresh)", () => {
      window.history.replaceState(null, "", "/?tab=demandHistory");
      expect(getInitialTab()).toBe("demandHistory");
    });
  });

  // U5.1 — browser Back/Forward (popstate) must apply TAB_REDIRECTS the same
  // way getInitialTab does, otherwise navigating to a retired tab key resolves
  // to a dead, still-bundled tab branch (non-deterministic for the same URL).
  describe("resolveTab", () => {
    it("redirects retired tab keys to their consolidated tab", () => {
      expect(resolveTab("controlTower")).toBe("commandCenter");
      expect(resolveTab("aiPlanner")).toBe("commandCenter");
      expect(resolveTab("storyboard")).toBe("commandCenter");
    });

    it("passes through valid non-redirected tabs unchanged", () => {
      expect(resolveTab("aggregateAnalysis")).toBe("aggregateAnalysis");
      expect(resolveTab("customerAnalytics")).toBe("customerAnalytics");
    });

    it("returns null for invalid tabs", () => {
      expect(resolveTab("bogus")).toBeNull();
      expect(resolveTab(null)).toBeNull();
    });

    it("resolves controlTower via popstate to the same tab as a fresh load", () => {
      window.history.replaceState(null, "", "/?tab=controlTower");
      const initial = getInitialTab();
      // popstate must reach the identical resolved tab, not the raw URL value
      expect(resolveTab("controlTower")).toBe(initial);
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

  describe("getScenarioJobParam", () => {
    it("returns null when no scenario_job param", () => {
      expect(getScenarioJobParam()).toBeNull();
    });

    it("reads scenario_job from URL", () => {
      window.history.replaceState(null, "", "/?scenario_job=job_abc123");
      expect(getScenarioJobParam()).toBe("job_abc123");
    });
  });

  describe("setScenarioJobParam", () => {
    it("sets scenario_job param in URL", () => {
      setScenarioJobParam("job_xyz789");
      const params = new URLSearchParams(window.location.search);
      expect(params.get("scenario_job")).toBe("job_xyz789");
    });

    it("removes scenario_job param when null", () => {
      setScenarioJobParam("job_abc123");
      expect(new URLSearchParams(window.location.search).get("scenario_job")).toBe("job_abc123");
      setScenarioJobParam(null);
      expect(new URLSearchParams(window.location.search).get("scenario_job")).toBeNull();
    });
  });

  describe("exported constants", () => {
    it("VALID_TABS has 28 entries including commandCenter, aggregateAnalysis, itemAnalysis, invBacktest, jobs, aiPlanner, sop, lgbmTuning, sqlRunner, skuFeatures, and demandHistory", () => {
      expect(VALID_TABS).toHaveLength(28);  // aiPlannerFva removed
      expect(VALID_TABS).toContain("commandCenter");
      expect(VALID_TABS).toContain("aggregateAnalysis");
      expect(VALID_TABS).toContain("overview");
      expect(VALID_TABS).toContain("accuracy");
      expect(VALID_TABS).toContain("itemAnalysis");
      expect(VALID_TABS).toContain("inventory");
      expect(VALID_TABS).toContain("invBacktest");
      expect(VALID_TABS).toContain("jobs");
      expect(VALID_TABS).toContain("lgbmTuning");
      expect(VALID_TABS).toContain("sqlRunner");
      expect(VALID_TABS).toContain("customerAnalytics");
      expect(VALID_TABS).toContain("skuFeatures");
      expect(VALID_TABS).toContain("demandHistory");
    });

    it("DIMENSION_DOMAINS has 9 entries", () => {
      expect(DIMENSION_DOMAINS).toHaveLength(9);
    });

    // U2.9 — guard against the U2.2 class of bug: every clickable sidebar tab
    // must be deep-linkable. A sidebar key missing from VALID_TABS silently
    // bounces a refresh/bookmark back to Command Center.
    it("every NAV_ITEMS sidebar key is in VALID_TABS (deep-linkable)", () => {
      const missing = NAV_ITEMS.map((i) => i.key).filter(
        (key) => !VALID_TABS.includes(key),
      );
      expect(missing).toEqual([]);
    });
  });
});
