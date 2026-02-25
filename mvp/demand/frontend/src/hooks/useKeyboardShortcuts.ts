import { useEffect, useState, useCallback } from "react";

interface ShortcutConfig {
  onTabSwitch: (tab: string) => void;
  onFocusSearch?: () => void;
  onClosePanel?: () => void;
  onPrevPage?: () => void;
  onNextPage?: () => void;
  onToggleFields?: () => void;
  onCycleMotif?: () => void;
}

const TAB_MAP: Record<string, string> = {
  "1": "explorer",
  "2": "clusters",
  "3": "dfuAnalysis",
  "4": "accuracy",
  "5": "intel",
  "6": "inventory",
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

      // 1-5: tab switching
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

      // Ctrl+M / Cmd+M: cycle motif theme
      if (e.key === "m" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        config.onCycleMotif?.();
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
