import { describe, it, expect } from "vitest";
import { ICON_SIZE } from "@/constants/layout";

describe("ICON_SIZE", () => {
  it("has sm, md, lg keys", () => {
    expect(ICON_SIZE).toHaveProperty("sm");
    expect(ICON_SIZE).toHaveProperty("md");
    expect(ICON_SIZE).toHaveProperty("lg");
  });

  it("sm contains h-4 w-4 classes", () => {
    expect(ICON_SIZE.sm).toBe("h-4 w-4");
  });

  it("md contains h-[18px] w-[18px] classes", () => {
    expect(ICON_SIZE.md).toBe("h-[18px] w-[18px]");
  });

  it("lg contains h-5 w-5 classes", () => {
    expect(ICON_SIZE.lg).toBe("h-5 w-5");
  });
});
