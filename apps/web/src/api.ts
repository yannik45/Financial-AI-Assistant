import type {
  Account,
  Analytics,
  DemoBankFeedCreate,
  DemoBankFeedResult,
  Portfolio,
  MarketInstrument,
  MarketDataStatus,
  MarketQuote,
  MarketVolatilityForecast,
  PortfolioOrderCreate,
  TradingPortfolio,
  PortfolioCreate,
  Transaction,
  TransactionCreate,
  TransactionClassification,
  TransactionClassificationRequest,
  TransactionClassificationStatus,
  TransactionFilters,
  TransactionPage,
} from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const details = payload?.details?.map((item: { row?: number; field: string; message: string }) => `${item.row ? `Row ${item.row}, ` : ""}${item.field}: ${item.message}`).join("\n");
    const message =
      details ||
      payload?.detail?.message ||
      payload?.message ||
      `Request failed (${response.status})`;
    throw new ApiError(message, response.status, payload?.detail?.code);
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
  generateDemoBankFeed: (payload: DemoBankFeedCreate) =>
    request<DemoBankFeedResult>("/v1/transactions/demo-bank-feed", {
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
  transactionClassificationStatus: () =>
    request<TransactionClassificationStatus>("/v1/transactions/classification/status"),
  reviewTransactionCategory: (transactionId: string, category: string) =>
    request<Transaction>(`/v1/transactions/${transactionId}/category`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category }),
    }),
  importPortfolio: (name: string, file: File) => {
    const body = new FormData();
    body.append("name", name);
    body.append("file", file);
    return request<Portfolio>("/v1/portfolios/import", { method: "POST", body });
  },
  marketDataStatus: () => request<MarketDataStatus>("/v1/market/status"),
  marketInstruments: (query: string, mode: "demo" | "external") =>
    request<MarketInstrument[]>(`/v1/market/instruments?query=${encodeURIComponent(query)}&mode=${mode}`),
  marketQuote: (instrumentId: string) =>
    request<MarketQuote>(`/v1/market/instruments/${instrumentId}/quote`),
  marketVolatilityForecast: (instrumentId: string) =>
    request<MarketVolatilityForecast>(
      `/v1/market/instruments/${instrumentId}/volatility-forecast`,
    ),
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

