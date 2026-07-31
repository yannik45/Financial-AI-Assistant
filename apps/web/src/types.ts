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

export type AccountType = "checking" | "savings" | "brokerage";
export type Account = {
  id: string;
  name: string;
  account_type: AccountType;
  currency: string;
  kind: "demo" | "manual" | "imported";
  created_at: string;
  transaction_count: number;
};

export type TransactionType =
  | "card_payment"
  | "transfer"
  | "direct_debit"
  | "cash_withdrawal"
  | "salary"
  | "deposit"
  | "withdrawal"
  | "security_buy"
  | "security_sell"
  | "dividend"
  | "interest"
  | "fee"
  | "tax";

export type Transaction = {
  id: string;
  account_id: string;
  booked_at: string;
  name: string;
  amount: string;
  currency: string;
  transaction_type: TransactionType;
  counterparty: string | null;
  category: string | null;
  notes: string | null;
  source: "demo" | "manual" | "imported";
  security_symbol: string | null;
  quantity: string | null;
  unit_price: string | null;
  fees: string;
  taxes: string;
  created_at: string;
  classifications: TransactionClassificationRecord[];
};

export type ClassificationRoute = "deterministic" | "expense_model" | "needs_review";
export type ClassificationMethod = "deterministic" | "ml" | "none";
export type FeedbackStatus = "accepted" | "corrected" | "manual" | "unreviewed";

export type TransactionClassification = {
  category: string | null;
  route: ClassificationRoute;
  classification_method: ClassificationMethod;
  confidence: number | null;
  needs_review: boolean;
  reason: string;
  taxonomy_version: string;
  model_version: string | null;
};

export type TransactionClassificationRecord = Omit<TransactionClassification, "category"> & {
  id: string;
  predicted_category: string | null;
  final_category: string | null;
  feedback_status: FeedbackStatus;
  created_at: string;
};

export type TransactionClassificationRequest = {
  transaction_type: TransactionType;
  description: string;
  counterparty?: string;
};

export type TransactionPage = {
  items: Transaction[];
  total: number;
  limit: number;
  offset: number;
};

export type TransactionFilters = {
  account_id?: string;
  transaction_type?: TransactionType;
  category?: string;
  date_from?: string;
  date_to?: string;
  limit: number;
  offset: number;
};

export type TransactionCreate = {
  account_id: string;
  booked_at: string;
  name: string;
  amount: string;
  currency: string;
  transaction_type: TransactionType;
  counterparty?: string;
  category?: string;
  notes?: string;
  security_symbol?: string;
  quantity?: string;
  unit_price?: string;
  fees?: string;
  taxes?: string;
};

