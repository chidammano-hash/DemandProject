import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { Card } from "@/components/ui/card";

describe("Card", () => {
  it("a static card carries no hover-lift affordance", () => {
    const { container } = render(<Card>body</Card>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).not.toMatch(/-translate-y/);
    expect(el.className).not.toMatch(/hover:shadow-card-hover/);
  });

  it("an interactive card lifts and deepens its shadow on hover", () => {
    const { container } = render(<Card interactive>body</Card>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toMatch(/hover:-translate-y-0\.5/);
    expect(el.className).toMatch(/hover:shadow-card-hover/);
  });

  it("an interactive card suppresses the lift under reduced motion", () => {
    const { container } = render(<Card interactive>body</Card>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toMatch(/motion-reduce:hover:translate-y-0/);
    expect(el.className).toMatch(/motion-reduce:transition-none/);
  });
});
