import type { Analytics, Portfolio } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const details = payload?.details?.map((item: { row?: number; field: string; message: string }) => `${item.row ? `Row ${item.row}, ` : ""}${item.field}: ${item.message}`).join("\n");
    throw new Error(details || payload?.detail?.message || payload?.message || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  portfolios: () => request<Portfolio[]>("/v1/portfolios"),
  analytics: (id: string) => request<Analytics>(`/v1/portfolios/${id}/analytics`),
  importPortfolio: (name: string, file: File) => {
    const body = new FormData();
    body.append("name", name);
    body.append("file", file);
    return request<Portfolio>("/v1/portfolios/import", { method: "POST", body });
  },
};

