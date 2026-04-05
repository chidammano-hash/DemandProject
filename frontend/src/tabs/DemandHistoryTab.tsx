import { useState, useCallback, createContext, useContext } from "react";

import { DemandWorkbenchPanel } from "./demand-history/WorkbenchPanel";
import { DecompositionPanel } from "./demand-history/DecompositionPanel";
import { ComparisonPanel } from "./demand-history/ComparisonPanel";
import { MatrixPanel } from "./demand-history/MatrixPanel";

// ---------------------------------------------------------------------------
// Shared item+loc context across sub-panels
// ---------------------------------------------------------------------------

interface DemandHistorySelection {
  itemId: string;
  loc: string;
  setSelection: (itemId: string, loc: string) => void;
}

const SelectionContext = createContext<DemandHistorySelection>({
  itemId: "",
  loc: "",
  setSelection: () => {},
});

export function useDemandHistorySelection() {
  return useContext(SelectionContext);
}

// ---------------------------------------------------------------------------
// Sub-panel definitions
// ---------------------------------------------------------------------------

type PanelId = "workbench" | "decomposition" | "comparison" | "matrix";

const PANELS: { id: PanelId; label: string }[] = [
  { id: "workbench", label: "Workbench" },
  { id: "decomposition", label: "Decomposition" },
  { id: "comparison", label: "Comparison" },
  { id: "matrix", label: "Matrix" },
];

// ---------------------------------------------------------------------------
// Main tab
// ---------------------------------------------------------------------------

export default function DemandHistoryTab() {
  const [activePanel, setActivePanel] = useState<PanelId>("workbench");
  const [itemId, setItemId] = useState("");
  const [loc, setLoc] = useState("");

  const setSelection = useCallback((newItem: string, newLoc: string) => {
    setItemId(newItem);
    setLoc(newLoc);
  }, []);

  return (
    <SelectionContext.Provider value={{ itemId, loc, setSelection }}>
      <div className="flex flex-col h-full">
        {/* Sub-navigation */}
        <div className="flex items-center gap-1 px-4 pt-3 pb-2 border-b dark:border-gray-700">
          {PANELS.map((p) => (
            <button
              key={p.id}
              onClick={() => setActivePanel(p.id)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                activePanel === p.id
                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                  : "text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800"
              }`}
            >
              {p.label}
            </button>
          ))}

          {/* Active selection badge */}
          {itemId && loc && (
            <span className="ml-auto text-xs bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-gray-600 dark:text-gray-400">
              {itemId} @ {loc}
            </span>
          )}
        </div>

        {/* Panel content */}
        <div className="flex-1 overflow-auto p-4">
          {activePanel === "workbench" && <DemandWorkbenchPanel />}
          {activePanel === "decomposition" && <DecompositionPanel />}
          {activePanel === "comparison" && <ComparisonPanel />}
          {activePanel === "matrix" && <MatrixPanel />}
        </div>
      </div>
    </SelectionContext.Provider>
  );
}