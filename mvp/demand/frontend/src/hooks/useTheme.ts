import { useCallback, useEffect, useState } from "react";
import type { Theme } from "@/types";
import { TREND_COLORS_BY_THEME, CHART_COLORS } from "@/constants/colors";

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("ds-theme") as Theme | null;
    return saved && ["light", "dark", "midnight"].includes(saved) ? saved : "light";
  });

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-transitioning", "true");
    root.classList.remove("light", "dark", "midnight");
    root.classList.add(theme);
    localStorage.setItem("ds-theme", theme);
    const timer = setTimeout(() => root.removeAttribute("data-transitioning"), 300);
    return () => clearTimeout(timer);
  }, [theme]);

  const changeTheme = useCallback((t: Theme) => setTheme(t), []);

  return {
    theme,
    setTheme: changeTheme,
    trendColors: TREND_COLORS_BY_THEME[theme],
    chartColors: CHART_COLORS[theme],
  };
}
