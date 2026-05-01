import { createContext, useContext } from "react";

// Decoupling layer for cross-panel navigation inside InvPlanningTab. The
// active sub-panel state lives in the tab itself; without this context any
// child panel that wanted to deep-link elsewhere would force prop-drilling
// through every panel render. The provider is wired in InvPlanningTab.
export interface InvPlanningNavContextValue {
  navigateTo: (panel: string) => void;
}

const InvPlanningNavContext = createContext<InvPlanningNavContextValue | null>(null);

export function InvPlanningNavProvider({
  value,
  children,
}: {
  value: InvPlanningNavContextValue;
  children: React.ReactNode;
}) {
  return (
    <InvPlanningNavContext.Provider value={value}>
      {children}
    </InvPlanningNavContext.Provider>
  );
}

// Returns null outside the provider so panels can be rendered standalone
// (e.g. in tests or storybook) without crashing — callers should null-check.
export function useInvPlanningNav(): InvPlanningNavContextValue | null {
  return useContext(InvPlanningNavContext);
}
