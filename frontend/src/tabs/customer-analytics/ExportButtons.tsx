import { useCallback } from "react";

interface Props {
  panelId: string;
  getData: () => Record<string, unknown>[];
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function toCsv(rows: Record<string, unknown>[]): string {
  if (rows.length === 0) return "";
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.join(","),
    ...rows.map((r) =>
      headers.map((h) => {
        const v = r[h];
        const s = v == null ? "" : String(v);
        return s.includes(",") || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s;
      }).join(",")
    ),
  ];
  return lines.join("\n");
}

export function ExportButtons({ panelId, getData }: Props) {
  const handlePng = useCallback(() => {
    const container = document.querySelector(`[data-panel-id="${panelId}"]`);
    if (!container) {
      // Try to find an ECharts instance
      const echartEl = document.querySelector(`[aria-label*="${panelId}"] canvas`);
      if (echartEl) {
        const canvas = echartEl as HTMLCanvasElement;
        canvas.toBlob((blob) => {
          if (blob) downloadBlob(blob, `${panelId}.png`);
        });
      }
      return;
    }
  }, [panelId]);

  const handleCsv = useCallback(() => {
    const rows = getData();
    if (!rows.length) return;
    const csv = toCsv(rows);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    downloadBlob(blob, `${panelId}.csv`);
  }, [panelId, getData]);

  return (
    <div className="flex gap-1">
      <button
        onClick={handlePng}
        className="p-1 text-muted-foreground hover:text-foreground rounded"
        title="Export PNG"
        aria-label={`Export ${panelId} as PNG`}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      </button>
      <button
        onClick={handleCsv}
        className="p-1 text-muted-foreground hover:text-foreground rounded"
        title="Export CSV"
        aria-label={`Export ${panelId} as CSV`}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="12" y1="18" x2="12" y2="12" />
          <polyline points="9 15 12 18 15 15" />
        </svg>
      </button>
    </div>
  );
}
