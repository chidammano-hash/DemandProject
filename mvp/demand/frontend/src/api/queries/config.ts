/** Config management API queries — Settings tab */
import { fetchJson } from "./core";

export interface ConfigCategory {
  key: string;
  label: string;
  description: string;
}

export interface ConfigListItem {
  name: string;
  label: string;
  category: string;
  description: string;
  exists: boolean;
}

export interface ConfigListPayload {
  categories: ConfigCategory[];
  configs: ConfigListItem[];
}

export interface ConfigField {
  path: string;
  value: unknown;
  label: string;
  description: string;
  type: "number" | "integer" | "text" | "boolean" | "select" | "array" | "object";
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: string[];
}

export interface ConfigDetailPayload {
  name: string;
  label: string;
  category: string;
  description: string;
  fields: ConfigField[];
  raw: Record<string, unknown>;
}

export interface ConfigUpdateResult {
  name: string;
  changed: string[];
  message: string;
}

export const configKeys = {
  list: () => ["config-list"] as const,
  detail: (name: string) => ["config-detail", name] as const,
};

export async function fetchConfigList(): Promise<ConfigListPayload> {
  return fetchJson("/config");
}

export async function fetchConfigDetail(name: string): Promise<ConfigDetailPayload> {
  return fetchJson(`/config/${name}`);
}

export async function updateConfig(
  name: string,
  values: Record<string, unknown>,
): Promise<ConfigUpdateResult> {
  const resp = await fetch(`/config/${name}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

export async function resetConfig(name: string): Promise<ConfigUpdateResult> {
  const resp = await fetch(`/config/${name}/reset`, { method: "POST" });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}
