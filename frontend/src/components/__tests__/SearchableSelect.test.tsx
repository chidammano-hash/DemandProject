import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SearchableSelect, demoteJunkOptions } from "../SearchableSelect";

describe("SearchableSelect (U5.11 — searchable Store Type control)", () => {
  const options = [
    "**OBSOLETE **",
    "Chain Grocery Store",
    "Independent Package Store",
    "ALL",
    "UNKNOWN NO/SS/TR",
    "Wine/Gourmet Shop",
  ];

  it("exposes a searchable input with role=combobox", () => {
    render(
      <SearchableSelect
        value=""
        options={options}
        placeholder="All types"
        ariaLabel="Store Type"
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole("combobox", { name: "Store Type" })).toBeInTheDocument();
  });

  it("narrows the option list as the user types", async () => {
    render(
      <SearchableSelect
        value=""
        options={options}
        placeholder="All types"
        ariaLabel="Store Type"
        onChange={() => {}}
      />,
    );
    const input = screen.getByRole("combobox", { name: "Store Type" });
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "groc" } });
    await waitFor(() =>
      expect(screen.getByRole("option", { name: /Chain Grocery Store/ })).toBeInTheDocument(),
    );
    // A non-matching option must not be in the filtered list.
    expect(screen.queryByRole("option", { name: /Wine\/Gourmet Shop/ })).toBeNull();
  });

  it("calls onChange with the selected option value", async () => {
    const onChange = vi.fn();
    render(
      <SearchableSelect
        value=""
        options={options}
        placeholder="All types"
        ariaLabel="Store Type"
        onChange={onChange}
      />,
    );
    const input = screen.getByRole("combobox", { name: "Store Type" });
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "groc" } });
    const opt = await screen.findByRole("option", { name: /Chain Grocery Store/ });
    fireEvent.mouseDown(opt);
    expect(onChange).toHaveBeenCalledWith("Chain Grocery Store");
  });
});

describe("demoteJunkOptions (U5.11 — junk sorts after real values)", () => {
  it("sorts obsolete/ALL/UNKNOWN sentinels after real values", () => {
    const sorted = demoteJunkOptions([
      "**OBSOLETE **",
      "Chain Grocery Store",
      "ALL",
      "UNKNOWN NO/SS/TR",
      "Wine/Gourmet Shop",
    ]);
    const grocIdx = sorted.indexOf("Chain Grocery Store");
    const obsoleteIdx = sorted.indexOf("**OBSOLETE **");
    const allIdx = sorted.indexOf("ALL");
    const unknownIdx = sorted.indexOf("UNKNOWN NO/SS/TR");
    expect(grocIdx).toBeLessThan(obsoleteIdx);
    expect(grocIdx).toBeLessThan(allIdx);
    expect(grocIdx).toBeLessThan(unknownIdx);
  });

  it("keeps real values in their original relative order", () => {
    const sorted = demoteJunkOptions(["Wine/Gourmet Shop", "Chain Grocery Store"]);
    expect(sorted).toEqual(["Wine/Gourmet Shop", "Chain Grocery Store"]);
  });
});
