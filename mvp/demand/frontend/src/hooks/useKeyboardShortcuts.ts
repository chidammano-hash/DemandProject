import { useEffect, useState, useCallback } from "react";

interface ShortcutConfig {
  onTabSwitch: (tab: string) => void;
  onFocusSearch?: () => void;
  onClosePanel?: () => void;
  onPrevPage?: () => void;
  onNextPage?: () => void;
  onToggleFields?: () => void;
  onToggleSidebar?: () => void;
  onToggleColorMode?: () => void;
}

const TAB_MAP: Record<string, string> = {
  "1": "aiPlanner",
  "2": "controlTower",
  "3": "overview",
  "4": "accuracy",
  "5": "itemAnalysis",
  "6": "exceptions",
  "7": "invPlanning",
  "8": "jobs",
};

export function useKeyboardShortcuts(config: ShortcutConfig) {
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      const isInput = target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable;

      // ? always toggles help (even from input)
      if (e.key === "?" && !e.ctrlKey && !e.metaKey) {
        if (!isInput) {
          e.preventDefault();
          setShowHelp((v) => !v);
          return;
        }
      }

      // Escape closes help or panels
      if (e.key === "Escape") {
        if (showHelp) {
          setShowHelp(false);
          return;
        }
        config.onClosePanel?.();
        return;
      }

      // Skip remaining shortcuts when in input
      if (isInput) return;

      // [ toggle sidebar
      if (e.key === "[") {
        e.preventDefault();
        config.onToggleSidebar?.();
        return;
      }

      // d toggle dark/light mode
      if (e.key === "d" && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        config.onToggleColorMode?.();
        return;
      }

      // 1-8: tab switching
      if (TAB_MAP[e.key]) {
        e.preventDefault();
        config.onTabSwitch(TAB_MAP[e.key]);
        return;
      }

      // / focus search
      if (e.key === "/") {
        e.preventDefault();
        config.onFocusSearch?.();
        return;
      }

      // Arrow keys for pagination
      if (e.key === "ArrowLeft") {
        config.onPrevPage?.();
        return;
      }
      if (e.key === "ArrowRight") {
        config.onNextPage?.();
        return;
      }

      // Ctrl+E toggle fields
      if (e.key === "e" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        config.onToggleFields?.();
        return;
      }
    }

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [config, showHelp]);

  const closeHelp = useCallback(() => setShowHelp(false), []);

  return { showHelp, closeHelp };
}
