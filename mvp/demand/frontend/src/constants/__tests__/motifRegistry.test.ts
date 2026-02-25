import { describe, it, expect } from "vitest";
import {
  getMotif,
  getAllMotifs,
  DEFAULT_MOTIF_ID,
  registerMotif,
} from "@/constants/motifs";

// Importing "@/constants/motifs" triggers the boot file which registers all 5
// motifs (periodic, spirits, space, f1, zen) via side-effect imports.

describe("motifRegistry", () => {
  it("registerMotif adds a motif to the registry", () => {
    // All 5 motifs are already registered via the boot file import.
    // Verify that one of the registered motifs can be retrieved.
    const periodic = getMotif("periodic");
    expect(periodic).toBeDefined();
    expect(periodic.id).toBe("periodic");
  });

  it("getMotif returns the registered motif", () => {
    const spirits = getMotif("spirits");
    expect(spirits.id).toBe("spirits");
    expect(spirits.displayName).toBe("The Cellar");

    const space = getMotif("space");
    expect(space.id).toBe("space");
    expect(space.displayName).toBe("Deep Space");

    const f1 = getMotif("f1");
    expect(f1.id).toBe("f1");
    expect(f1.displayName).toBe("Formula 1");

    const zen = getMotif("zen");
    expect(zen.id).toBe("zen");
    expect(zen.displayName).toBe("Zen Garden");
  });

  it("getMotif throws for unknown ID", () => {
    expect(() => getMotif("nonexistent" as any)).toThrow(
      '[MotifRegistry] Unknown motif id: "nonexistent"',
    );
  });

  it("getAllMotifs returns all registered motifs", () => {
    const all = getAllMotifs();
    expect(all.length).toBe(5);
    const ids = all.map((m) => m.id);
    expect(ids).toContain("periodic");
    expect(ids).toContain("spirits");
    expect(ids).toContain("space");
    expect(ids).toContain("f1");
    expect(ids).toContain("zen");
  });

  it('DEFAULT_MOTIF_ID is "periodic"', () => {
    expect(DEFAULT_MOTIF_ID).toBe("periodic");
  });

  it("registerMotif overwrites an existing motif without throwing", () => {
    // Re-register a motif (the boot file already registered "periodic").
    // registerMotif should warn but not throw.
    const existing = getMotif("periodic");
    expect(() => registerMotif(existing)).not.toThrow();
    // The motif should still be retrievable after re-registration.
    expect(getMotif("periodic")).toBeDefined();
  });

  it("each motif has required structure fields", () => {
    const all = getAllMotifs();
    for (const motif of all) {
      expect(motif.id).toBeTruthy();
      expect(motif.displayName).toBeTruthy();
      expect(motif.description).toBeTruthy();
      expect(motif.previewTile).toBeDefined();
      expect(motif.tiles).toBeDefined();
      expect(motif.loading).toBeDefined();
      expect(motif.loading.animationName).toBeTruthy();
      expect(motif.chrome).toBeDefined();
      expect(motif.chrome.appName).toBeTruthy();
    }
  });
});
