import { describe, it, expect } from "vitest";
import { cn } from "@/lib/utils";

describe("cn", () => {
  it("returns a single class name unchanged", () => {
    expect(cn("foo")).toBe("foo");
  });

  it("merges multiple class names with a space", () => {
    expect(cn("foo", "bar")).toBe("foo bar");
  });

  it("ignores falsy values (false)", () => {
    expect(cn("foo", false && "bar", "baz")).toBe("foo baz");
  });

  it("ignores undefined", () => {
    expect(cn("foo", undefined)).toBe("foo");
  });

  it("ignores null", () => {
    expect(cn("foo", null)).toBe("foo");
  });

  it("handles an empty string input", () => {
    expect(cn("")).toBe("");
  });

  it("handles all falsy inputs, returning empty string", () => {
    expect(cn(undefined, null, false as unknown as string)).toBe("");
  });

  it("deduplicates conflicting Tailwind utility classes (last wins)", () => {
    // twMerge should drop p-2 in favour of p-4
    expect(cn("p-2", "p-4")).toBe("p-4");
  });

  it("deduplicates conflicting text-color classes", () => {
    expect(cn("text-red-500", "text-blue-500")).toBe("text-blue-500");
  });

  it("preserves non-conflicting classes together", () => {
    const result = cn("flex", "items-center", "gap-2");
    expect(result).toBe("flex items-center gap-2");
  });

  it("supports array inputs", () => {
    // clsx accepts arrays
    expect(cn(["foo", "bar"])).toBe("foo bar");
  });

  it("supports object inputs with boolean values", () => {
    // clsx object syntax: { className: condition }
    expect(cn({ foo: true, bar: false, baz: true })).toBe("foo baz");
  });

  it("merges conditional and unconditional classes correctly", () => {
    const isActive = true;
    const isDisabled = false;
    const result = cn("base", isActive && "active", isDisabled && "disabled");
    expect(result).toBe("base active");
  });
});
