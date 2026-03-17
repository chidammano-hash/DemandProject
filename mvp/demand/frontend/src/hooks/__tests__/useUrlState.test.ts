import { describe, it, expect, beforeEach } from "vitest";
import {
  getInitialDomain,
  getInitialTab,
  updateUrlState,
  getScenarioJobParam,
  setScenarioJobParam,
  VALID_TABS,
  DIMENSION_DOMAINS,
} from "@/hooks/useUrlState";

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
    it("defaults to aggregateAnalysis when no params", () => {
      expect(getInitialTab()).toBe("aggregateAnalysis");
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
      expect(getInitialTab()).toBe("aggregateAnalysis");
    });

    it("returns domain for analytics domains", () => {
      window.history.replaceState(null, "", "/?domain=sales");
      expect(getInitialTab()).toBe("sales");
    });

    it("accepts aggregateAnalysis tab from URL", () => {
      window.history.replaceState(null, "", "/?tab=aggregateAnalysis");
      expect(getInitialTab()).toBe("aggregateAnalysis");
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
    it("VALID_TABS has 20 entries including aggregateAnalysis, itemAnalysis, invBacktest, jobs, aiPlanner, sop, and customerMap", () => {
      expect(VALID_TABS).toHaveLength(20);
      expect(VALID_TABS).toContain("aggregateAnalysis");
      expect(VALID_TABS).toContain("overview");
      expect(VALID_TABS).toContain("accuracy");
      expect(VALID_TABS).toContain("itemAnalysis");
      expect(VALID_TABS).toContain("inventory");
      expect(VALID_TABS).toContain("invBacktest");
      expect(VALID_TABS).toContain("jobs");
    });

    it("DIMENSION_DOMAINS has 7 entries", () => {
      expect(DIMENSION_DOMAINS).toHaveLength(7);
    });
  });
});
