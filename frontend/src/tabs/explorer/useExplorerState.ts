/**
 * Local state, derived values, and handlers for the Data Explorer tab.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useDebounce } from "@/hooks/useDebounce";
import type { DomainMeta } from "@/types";

import type { ClusterSource, SortDir } from "./types";

export interface ExplorerStateApi {
  // raw state
  offset: number;
  limit: number;
  search: string;
  sortBy: string;
  sortDir: SortDir;
  columnFilters: Record<string, string>;
  visibleColumns: Record<string, boolean>;
  showFieldPanel: boolean;
  itemFilter: string;
  locationFilter: string;
  selectedModel: string;
  selectedCluster: string;
  clusterSource: ClusterSource;
  autoSampledDomain: string;
  columnSuggestions: Record<string, string[]>;
  // setters used outside (kept narrow)
  setOffset: React.Dispatch<React.SetStateAction<number>>;
  setItemFilter: React.Dispatch<React.SetStateAction<string>>;
  setLocationFilter: React.Dispatch<React.SetStateAction<string>>;
  setSelectedModel: React.Dispatch<React.SetStateAction<string>>;
  setSelectedCluster: React.Dispatch<React.SetStateAction<string>>;
  setClusterSource: React.Dispatch<React.SetStateAction<ClusterSource>>;
  setShowFieldPanel: React.Dispatch<React.SetStateAction<boolean>>;
  setAutoSampledDomain: React.Dispatch<React.SetStateAction<string>>;
  setColumnSuggestions: React.Dispatch<
    React.SetStateAction<Record<string, string[]>>
  >;
  // debounced
  debouncedSearch: string;
  debouncedColumnFilters: Record<string, string>;
  debouncedItemFilter: string;
  debouncedLocationFilter: string;
  // handlers
  toggleSort: (column: string) => void;
  toggleColumn: (column: string, checked: boolean) => void;
  handleSearchChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  handleLimitChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  handleColumnFilterChange: (col: string, value: string) => void;
  handlePreviousPage: () => void;
  handleNextPage: () => void;
  handleSelectAllColumns: () => void;
  handleDeselectAllColumns: () => void;
  handleItemFilterInput: (value: string) => void;
  handleLocationFilterInput: (value: string) => void;
  handleModelInput: (value: string) => void;
  handleClusterInput: (value: string) => void;
  handleClusterSourceInput: (value: ClusterSource) => void;
  handleToggleFieldPanel: () => void;
  resetForMeta: (meta: DomainMeta) => void;
  setVisibleColumnsBulk: (next: Record<string, boolean>) => void;
}

export function useExplorerState(domain: string): ExplorerStateApi {
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(100);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [columnFilters, setColumnFilters] = useState<Record<string, string>>({});
  const [visibleColumns, setVisibleColumns] = useState<Record<string, boolean>>({});
  const [showFieldPanel, setShowFieldPanel] = useState(false);

  const [itemFilter, setItemFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedCluster, setSelectedCluster] = useState("");
  const [clusterSource, setClusterSource] = useState<ClusterSource>("ml");
  const [autoSampledDomain, setAutoSampledDomain] = useState("");
  const [columnSuggestions, setColumnSuggestions] = useState<
    Record<string, string[]>
  >({});

  // Sync global item/location filter into local inputs.
  const { filters: globalFilters } = useGlobalFilterContext();
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1)
      setLocationFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const isFactDomain = domain === "sales" || domain === "forecast";
  const filterDebounceMs = isFactDomain ? 500 : 300;

  const debouncedSearch = useDebounce(search, filterDebounceMs);
  const debouncedColumnFilters = useDebounce(columnFilters, filterDebounceMs);
  const debouncedItemFilter = useDebounce(itemFilter, filterDebounceMs);
  const debouncedLocationFilter = useDebounce(locationFilter, filterDebounceMs);

  // ---- handlers ---------------------------------------------------------
  const toggleSort = useCallback(
    (column: string) => {
      if (sortBy === column) {
        setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
        return;
      }
      setSortBy(column);
      setSortDir("asc");
    },
    [sortBy],
  );

  const toggleColumn = useCallback((column: string, checked: boolean) => {
    setVisibleColumns((prev) => ({ ...prev, [column]: checked }));
  }, []);

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setOffset(0);
      setSearch(e.target.value);
    },
    [],
  );

  const handleLimitChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setOffset(0);
      setLimit(Number(e.target.value));
    },
    [],
  );

  const handleColumnFilterChange = useCallback(
    (col: string, value: string) => {
      setOffset(0);
      setColumnFilters((prev) => ({ ...prev, [col]: value }));
    },
    [],
  );

  const handlePreviousPage = useCallback(() => {
    setOffset((prev) => Math.max(0, prev - limit));
  }, [limit]);

  const handleNextPage = useCallback(() => {
    setOffset((prev) => prev + limit);
  }, [limit]);

  const setVisibleColumnsBulk = useCallback(
    (next: Record<string, boolean>) => {
      setVisibleColumns(next);
    },
    [],
  );

  const handleItemFilterInput = useCallback((value: string) => {
    setOffset(0);
    setItemFilter(value);
  }, []);

  const handleLocationFilterInput = useCallback((value: string) => {
    setOffset(0);
    setLocationFilter(value);
  }, []);

  const handleModelInput = useCallback((value: string) => {
    setOffset(0);
    setSelectedModel(value);
  }, []);

  const handleClusterInput = useCallback((value: string) => {
    setOffset(0);
    setSelectedCluster(value);
  }, []);

  const handleClusterSourceInput = useCallback((value: ClusterSource) => {
    setClusterSource(value);
    setSelectedCluster("");
    setOffset(0);
  }, []);

  const handleToggleFieldPanel = useCallback(() => {
    setShowFieldPanel((v) => !v);
  }, []);

  // resetForMeta: re-initialise local state when domain meta changes.
  const resetForMeta = useCallback((meta: DomainMeta) => {
    setOffset(0);
    setSearch("");
    setColumnFilters({});
    setSortBy(meta.default_sort);
    setSortDir("asc");
    setItemFilter("");
    setLocationFilter("");
    setSelectedModel("");
    setAutoSampledDomain("");
    setColumnSuggestions({});
    setVisibleColumns(
      Object.fromEntries(meta.columns.map((c) => [c, true])),
    );
  }, []);

  const handleSelectAllColumns = useCallback(() => {
    setVisibleColumns((prev) => {
      const all: Record<string, boolean> = {};
      Object.keys(prev).forEach((c) => {
        all[c] = true;
      });
      return all;
    });
  }, []);

  const handleDeselectAllColumns = useCallback(() => {
    setVisibleColumns((prev) => {
      const none: Record<string, boolean> = {};
      Object.keys(prev).forEach((c) => {
        none[c] = false;
      });
      return none;
    });
  }, []);

  return {
    offset,
    limit,
    search,
    sortBy,
    sortDir,
    columnFilters,
    visibleColumns,
    showFieldPanel,
    itemFilter,
    locationFilter,
    selectedModel,
    selectedCluster,
    clusterSource,
    autoSampledDomain,
    columnSuggestions,
    setOffset,
    setItemFilter,
    setLocationFilter,
    setSelectedModel,
    setSelectedCluster,
    setClusterSource,
    setShowFieldPanel,
    setAutoSampledDomain,
    setColumnSuggestions,
    debouncedSearch,
    debouncedColumnFilters,
    debouncedItemFilter,
    debouncedLocationFilter,
    toggleSort,
    toggleColumn,
    handleSearchChange,
    handleLimitChange,
    handleColumnFilterChange,
    handlePreviousPage,
    handleNextPage,
    handleSelectAllColumns,
    handleDeselectAllColumns,
    handleItemFilterInput,
    handleLocationFilterInput,
    handleModelInput,
    handleClusterInput,
    handleClusterSourceInput,
    handleToggleFieldPanel,
    resetForMeta,
    setVisibleColumnsBulk,
  };
}

