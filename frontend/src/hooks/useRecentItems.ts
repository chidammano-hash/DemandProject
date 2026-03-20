import { useState, useCallback, useEffect, useRef } from "react";

const STORAGE_KEY = "demand-studio-recent-items";
const MAX_RECENT = 5;

export interface RecentItem {
  itemNo: string;
  location?: string;
  label: string;
  timestamp: number;
}

export function useRecentItems() {
  const selfDispatch = useRef(false);
  const [items, setItems] = useState<RecentItem[]>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    selfDispatch.current = true;
    window.dispatchEvent(new CustomEvent("recent-items-changed"));
    selfDispatch.current = false;
  }, [items]);

  // Listen for changes from other components (skip self-dispatched events)
  useEffect(() => {
    function handleChange() {
      if (selfDispatch.current) return;
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) setItems(JSON.parse(stored));
      } catch { /* ignore */ }
    }
    window.addEventListener("recent-items-changed", handleChange);
    window.addEventListener("storage", handleChange);
    return () => {
      window.removeEventListener("recent-items-changed", handleChange);
      window.removeEventListener("storage", handleChange);
    };
  }, []);

  const addItem = useCallback((item: Omit<RecentItem, "timestamp">) => {
    setItems((prev) => {
      const filtered = prev.filter((p) => !(p.itemNo === item.itemNo && p.location === item.location));
      return [{ ...item, timestamp: Date.now() }, ...filtered].slice(0, MAX_RECENT);
    });
  }, []);

  const clearItems = useCallback(() => setItems([]), []);

  return { recentItems: items, addRecentItem: addItem, clearRecentItems: clearItems };
}
