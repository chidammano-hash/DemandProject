import { createContext, useContext, useReducer, type ReactNode, type Dispatch } from "react";

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export interface DashboardFilterState {
  selectedState: string;
  selectedChannel: string;
  selectedCustomer: string;
  selectedSegment: string;
}

const initialState: DashboardFilterState = {
  selectedState: "",
  selectedChannel: "",
  selectedCustomer: "",
  selectedSegment: "",
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

type Action =
  | { type: "SET_STATE"; payload: string }
  | { type: "SET_CHANNEL"; payload: string }
  | { type: "SET_CUSTOMER"; payload: string }
  | { type: "SET_SEGMENT"; payload: string }
  | { type: "CLEAR_ALL" };

function reducer(state: DashboardFilterState, action: Action): DashboardFilterState {
  switch (action.type) {
    case "SET_STATE":
      return { ...state, selectedState: action.payload };
    case "SET_CHANNEL":
      return { ...state, selectedChannel: action.payload };
    case "SET_CUSTOMER":
      return { ...state, selectedCustomer: action.payload };
    case "SET_SEGMENT":
      return { ...state, selectedSegment: action.payload };
    case "CLEAR_ALL":
      return initialState;
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

interface DashboardFilterContextValue {
  state: DashboardFilterState;
  dispatch: Dispatch<Action>;
}

const DashboardFilterContext = createContext<DashboardFilterContextValue | null>(null);

export function DashboardFilterProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <DashboardFilterContext.Provider value={{ state, dispatch }}>
      {children}
    </DashboardFilterContext.Provider>
  );
}

export function useDashboardFilter(): DashboardFilterContextValue {
  const ctx = useContext(DashboardFilterContext);
  if (!ctx) {
    throw new Error("useDashboardFilter must be used within DashboardFilterProvider");
  }
  return ctx;
}
