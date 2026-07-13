function formatErrorDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .filter((entry): entry is { loc?: unknown; msg?: unknown } => Boolean(entry) && typeof entry === "object")
      .map((entry) => {
        const location = Array.isArray(entry.loc) ? entry.loc.join(".") : "request";
        const message = typeof entry.msg === "string" ? entry.msg : "Invalid value";
        return `${location}: ${message}`;
      });
    return messages.join("; ") || "Request validation failed";
  }
  return "Request failed";
}

/** Shared JSON request helper used by every domain query module. */
export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const accessToken = getAccessToken();
  let requestInit = init;
  if (accessToken) {
    const requestHeaders = new Headers(init?.headers);
    if (!requestHeaders.has("Authorization")) {
      requestHeaders.set("Authorization", `Bearer ${accessToken}`);
    }
    requestInit = { ...init, headers: requestHeaders };
  }

  let res = await fetch(url, requestInit);
  if (res.status === 401 && url !== "/auth/login" && url !== "/auth/refresh") {
    const refreshToken = getRefreshToken();
    if (refreshToken) {
      const refreshResponse = await fetch("/auth/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (refreshResponse.ok) {
        const tokens = await refreshResponse.json() as {
          access_token: string;
          refresh_token: string;
        };
        storeTokens({ accessToken: tokens.access_token, refreshToken: tokens.refresh_token });
        const retryHeaders = new Headers(init?.headers);
        retryHeaders.set("Authorization", `Bearer ${tokens.access_token}`);
        res = await fetch(url, { ...init, headers: retryHeaders });
      } else {
        clearTokens();
      }
    }
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    // Parse FastAPI `{detail}` so formatApiError can sanitize a clean message,
    // and attach the HTTP status so the global handler maps it to friendly copy
    // (404 → "That record could not be found.") instead of leaking the raw body.
    let detail: unknown = text;
    try {
      detail = JSON.parse(text);
    } catch {
      /* non-JSON body — keep the raw text */
    }
    const message = detail && typeof detail === "object" && "detail" in detail
      ? formatErrorDetail((detail as { detail: unknown }).detail)
      : text || `HTTP ${res.status}`;
    const err = new Error(message);
    Object.assign(err, { status: res.status, detail });
    throw err;
  }
  return res.json() as Promise<T>;
}
import {
  clearTokens,
  getAccessToken,
  getRefreshToken,
  storeTokens,
} from "../authSession";
