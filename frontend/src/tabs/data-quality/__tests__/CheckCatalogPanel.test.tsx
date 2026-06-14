import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { CheckCatalogPanel } from "../CheckCatalogPanel";
import type { DQCheck } from "@/api/queries/platform";

function makeCheck(over: Partial<DQCheck>): DQCheck {
  return {
    check_id: 1,
    check_name: "completeness_forecast_item_id",
    check_type: "completeness",
    domain: "forecasting",
    table_name: "fact_external_forecast_monthly",
    severity: "critical",
    enabled: true,
    last_status: "pass",
    last_value: 0,
    last_run: "2026-06-14T01:00:00Z",
    ...over,
  };
}

describe("CheckCatalogPanel (F6.1 — outcome must outrank configured severity)", () => {
  // The severity badge is the uppercase pill in the table row (a <span>), not the
  // filter <option>; match it by its pill styling.
  const severityPill = () =>
    Array.from(document.querySelectorAll("span")).find((el) => el.textContent === "critical")!;

  it("does NOT paint a passing check's severity badge with the alarming critical style", () => {
    render(<CheckCatalogPanel checkList={[makeCheck({ last_status: "pass", severity: "critical" })]} domainFilter={null} />);
    // The severity pill still shows the word, but for a PASSING check it must be
    // de-emphasized (muted), not the bold red "critical" alarm style.
    const pill = severityPill();
    expect(pill.className).not.toContain("text-red-700");
    expect(pill.className).toContain("text-muted-foreground");
  });

  it("KEEPS the alarming critical style when the check actually fails", () => {
    render(<CheckCatalogPanel checkList={[makeCheck({ check_id: 2, last_status: "fail", severity: "critical" })]} domainFilter={null} />);
    const pill = severityPill();
    expect(pill.className).toContain("text-red-700");
  });

  it("labels the Last Value cell as a violation/defect metric so 0.00 on a pass does not read as broken", () => {
    render(<CheckCatalogPanel checkList={[makeCheck({ last_status: "pass", last_value: 0 })]} domainFilter={null} />);
    const cell = screen.getByText("0.00");
    // The clarifying context lives in a title tooltip on the cell.
    const titled = cell.closest("[title]");
    expect(titled).not.toBeNull();
    expect((titled as HTMLElement).getAttribute("title")?.toLowerCase()).toContain("violation");
  });
});
