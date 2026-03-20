import Papa from "papaparse";

export function downloadCsv(
  data: Record<string, unknown>[],
  filename: string,
  columns?: string[],
) {
  if (data.length === 0) return;

  const fields = columns ?? Object.keys(data[0]);
  const csv = Papa.unparse(data, { columns: fields });
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
