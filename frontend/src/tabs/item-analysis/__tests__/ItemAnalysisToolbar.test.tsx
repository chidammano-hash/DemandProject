import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ItemAnalysisToolbar } from "../ItemAnalysisToolbar";

describe("ItemAnalysisToolbar", () => {
  it("toggles panels and changes the SHAP model", () => {
    const onToggle = vi.fn();
    const onSelectedModelChange = vi.fn();
    render(
      <ItemAnalysisToolbar
        panels={{
          overlay: true,
          shap: true,
          forecastKpis: true,
          dqCorrections: false,
          aiChampion: true,
        }}
        allOn={false}
        onSetAll={vi.fn()}
        onToggle={onToggle}
        shapModels={["champion", "mstl"]}
        selectedModel={null}
        onSelectedModelChange={onSelectedModelChange}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Toggle Chart" }));
    fireEvent.change(screen.getByRole("combobox", { name: "SHAP model" }), {
      target: { value: "mstl" },
    });

    expect(onToggle).toHaveBeenCalledWith("overlay");
    expect(onSelectedModelChange).toHaveBeenCalledWith("mstl");
  });
});
