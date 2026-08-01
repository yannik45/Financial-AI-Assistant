import type {
  Account,
  Analytics,
  Portfolio,
  MarketInstrument,
  PortfolioOrderCreate,
  TradingPortfolio,
  PortfolioCreate,
  Transaction,
  TransactionCreate,
  TransactionClassification,
  TransactionClassificationRequest,
  TransactionFilters,
  TransactionPage,
} from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const details = payload?.details?.map((item: { row?: number; field: string; message: string }) => `${item.row ? `Row ${item.row}, ` : ""}${item.field}: ${item.message}`).join("\n");
    throw new Error(
      details ||
        payload?.detail?.message ||
        payload?.message ||
        `Request failed (${response.status})`,
    );
  }
  return response.json() as Promise<T>;
}

export const api = {
  portfolios: () => request<Portfolio[]>("/v1/portfolios"),
  analytics: (id: string) => request<Analytics>(`/v1/portfolios/${id}/analytics`),
  accounts: () => request<Account[]>("/v1/accounts"),
  transactions: (filters: TransactionFilters) => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== "") params.set(key, String(value));
    });
    return request<TransactionPage>(`/v1/transactions?${params}`);
  },
  createTransaction: (payload: TransactionCreate) =>
    request<Transaction>("/v1/transactions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  classifyTransaction: (payload: TransactionClassificationRequest) =>
    request<TransactionClassification>("/v1/transactions/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  importPortfolio: (name: string, file: File) => {
    const body = new FormData();
    body.append("name", name);
    body.append("file", file);
    return request<Portfolio>("/v1/portfolios/import", { method: "POST", body });
  },
  marketInstruments: (query: string) =>
    request<MarketInstrument[]>(`/v1/market/instruments?query=${encodeURIComponent(query)}`),
  portfolioOverview: (id: string) =>
    request<TradingPortfolio>(`/v1/portfolios/${id}/overview`),
  createPortfolio: (payload: PortfolioCreate) =>
    request<TradingPortfolio>("/v1/portfolios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  executePortfolioOrder: (portfolioId: string, payload: PortfolioOrderCreate) =>
    request(`/v1/portfolios/${portfolioId}/orders`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};

