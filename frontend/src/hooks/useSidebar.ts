import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "ds-sidebar";

function getInitialCollapsed(): boolean {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored !== null) return stored === "collapsed";
  } catch { /* ignore */ }
  // Default: collapsed on screens < 1440px
  return typeof window !== "undefined" ? window.innerWidth < 1440 : true;
}

export function useSidebar() {
  const [collapsed, setCollapsed] = useState(getInitialCollapsed);
  const [mobileOpen, setMobileOpen] = useState(false);

  // Persist to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? "collapsed" : "expanded");
    } catch { /* ignore */ }
  }, [collapsed]);

  // Auto-collapse on resize below 1024px
  useEffect(() => {
    const handler = () => {
      if (window.innerWidth < 1024) {
        setCollapsed(true);
      }
    };
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, []);

  const toggle = useCallback(() => setCollapsed((c) => !c), []);
  const openMobile = useCallback(() => setMobileOpen(true), []);
  const closeMobile = useCallback(() => setMobileOpen(false), []);

  return {
    collapsed,
    setCollapsed,
    toggle,
    mobileOpen,
    openMobile,
    closeMobile,
  };
}
