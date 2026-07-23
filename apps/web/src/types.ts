export type Portfolio = {
  id: string;
  name: string;
  kind: "demo" | "imported";
  base_currency: string;
  created_at: string;
  position_count: number;
};

export type Allocation = { label: string; value_eur: string; weight: number };
export type Analytics = {
  portfolio_id: string;
  as_of: string;
  data_version: string;
  market_value_eur: string;
  cost_basis_eur: string;
  unrealized_pnl_eur: string;
  unrealized_pnl_percent: number;
  trailing_return_percent: number;
  annualized_volatility_percent: number;
  max_drawdown_percent: number;
  concentration_hhi: number;
  largest_position_symbol: string;
  largest_position_weight: number;
  positions: Array<{ symbol: string; market_value_eur: string; cost_basis_eur: string; pnl_eur: string; weight: number }>;
  allocations: Record<"asset_class" | "sector" | "region" | "currency", Allocation[]>;
  value_series: Array<{ date: string; value_eur: string }>;
  warnings: string[];
};

