import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  queryKeys,
  STALE,
  fetchInventoryPosition,
  fetchInventoryKpis,
  fetchInventoryTrend,
  fetchInventoryItemDetail,
} from "@/api/queries";
import type { InventoryPosition, InventoryKpis, InventoryTrendPoint } from "@/types";
import { useDebounce } from "@/hooks/useDebounce";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

import { KpiSection } from "./inventory/KpiSection";
import { PositionTablePanel } from "./inventory/PositionTablePanel";
import { DemandVariabilityPanel } from "./inventory/DemandVariabilityPanel";
import { LeadTimeProfilePanel } from "./inventory/LeadTimeProfilePanel";

const PAGE_SIZE = 50;

type SortCol =
  | "item_id"
  | "loc"
  | "snapshot_date"
  | "qty_on_hand"
  | "qty_on_hand_on_order"
  | "qty_on_order"
  | "lead_time_days"
  | "mtd_sales";

export function InventoryTab() {
  // ---- Local state ----------------------------------------------------------
  const [itemFilter, setItemFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const [months, setMonths] = useState(12);
  const [offset, setOffset] = useState(0);
  const [sortBy, setSortBy] = useState<SortCol>("snapshot_date");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [selectedRow, setSelectedRow] = useState<{
    item: string;
    location: string;
  } | null>(null);

  // ---- Global filter sync ---------------------------------------------------
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

  const debouncedItem = useDebounce(itemFilter, 400);
  const debouncedLocation = useDebounce(locationFilter, 400);

  // ---- Derived params -------------------------------------------------------
  const kpiParams = useMemo(
    () => ({ item: debouncedItem, location: debouncedLocation, months }),
    [debouncedItem, debouncedLocation, months],
  );
  const trendParams = useMemo(
    () => ({ item: debouncedItem, location: debouncedLocation, months }),
    [debouncedItem, debouncedLocation, months],
  );
  const positionParams = useMemo(
    () => ({
      item: debouncedItem,
      location: debouncedLocation,
      limit: PAGE_SIZE,
      offset,
      sort_by: sortBy,
      sort_dir: sortDir,
    }),
    [debouncedItem, debouncedLocation, offset, sortBy, sortDir],
  );

  // ---- Data fetching --------------------------------------------------------
  const { data: kpiData, isLoading: loadingKpis } = useQuery({
    queryKey: queryKeys.inventoryKpis(kpiParams),
    queryFn: () => fetchInventoryKpis(kpiParams),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: trendPayload, isLoading: loadingTrend } = useQuery({
    queryKey: queryKeys.inventoryTrend(trendParams),
    queryFn: () => fetchInventoryTrend(trendParams),
    staleTime: STALE.FIVE_MIN,
  });

  const trendData: InventoryTrendPoint[] = trendPayload?.trend ?? [];
  const trendParams2 = trendPayload?.params;

  const { data: positionPayload, isLoading: loadingPosition } = useQuery({
    queryKey: queryKeys.inventoryPosition(positionParams),
    queryFn: () => fetchInventoryPosition(positionParams),
    staleTime: STALE.TWO_MIN,
  });

  const positions: InventoryPosition[] = positionPayload?.positions ?? [];
  const totalPositions = positionPayload?.total ?? 0;

  const { data: detailPayload, isLoading: loadingDetail } = useQuery({
    queryKey: queryKeys.inventoryItemDetail({
      item: selectedRow?.item ?? "",
      location: selectedRow?.location ?? "",
    }),
    queryFn: () =>
      fetchInventoryItemDetail({
        item: selectedRow!.item,
        location: selectedRow!.location,
        months,
      }),
    staleTime: STALE.TWO_MIN,
    enabled: selectedRow !== null,
  });

  // ---- Handlers -------------------------------------------------------------
  const handlePrevPage = useCallback(() => {
    setOffset((prev) => Math.max(0, prev - PAGE_SIZE));
  }, []);

  const handleNextPage = useCallback(() => {
    setOffset((prev) =>
      prev + PAGE_SIZE < totalPositions ? prev + PAGE_SIZE : prev,
    );
  }, [totalPositions]);

  const handleSort = useCallback(
    (col: SortCol) => {
      if (sortBy === col) {
        setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortBy(col);
        setSortDir("desc");
      }
      setOffset(0);
    },
    [sortBy],
  );

  const handleItemChange = useCallback(
    (value: string) => {
      setItemFilter(value);
      setOffset(0);
      setSelectedRow(null);
    },
    [],
  );

  const handleLocationChange = useCallback(
    (value: string) => {
      setLocationFilter(value);
      setOffset(0);
      setSelectedRow(null);
    },
    [],
  );

  const handleMonthsChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setMonths(Number(e.target.value));
      setOffset(0);
    },
    [],
  );

  const handleRowClick = useCallback(
    (row: InventoryPosition) => {
      if (
        selectedRow?.item === row.item_id &&
        selectedRow?.location === row.loc
      ) {
        setSelectedRow(null);
      } else {
        setSelectedRow({ item: row.item_id, location: row.loc });
      }
    },
    [selectedRow],
  );

  // ---- Render ---------------------------------------------------------------
  return (
    <section className="mt-4 space-y-4">
      <KpiSection
        kpiData={kpiData as InventoryKpis | undefined}
        isLoading={loadingKpis}
      />

      <PositionTablePanel
        positions={positions}
        totalPositions={totalPositions}
        isLoadingPosition={loadingPosition}
        months={months}
        onMonthsChange={handleMonthsChange}
        offset={offset}
        onPrevPage={handlePrevPage}
        onNextPage={handleNextPage}
        sortBy={sortBy}
        sortDir={sortDir}
        onSort={handleSort}
        selectedRow={selectedRow}
        onRowClick={handleRowClick}
        detailSnapshots={detailPayload?.snapshots}
        isLoadingDetail={loadingDetail}
      />

      <DemandVariabilityPanel />

      <LeadTimeProfilePanel />
    </section>
  );
}
