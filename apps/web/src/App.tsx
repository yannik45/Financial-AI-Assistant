import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "./api";
import PortfolioTradingPanel from "./PortfolioTradingPanel";
import type {
  Account,
  Allocation,
  Analytics,
  PortfolioRiskScore,
  TransactionCreate,
  TransactionFilters,
} from "./types";

const COLORS = ["#7cf4c5", "#68a7ff", "#b991ff", "#ffce6d", "#ff7887", "#54d6e8"];
const TRANSACTION_CATEGORIES = [
  "Income",
  "Housing",
  "Groceries",
  "Dining",
  "Transport",
  "Utilities",
  "Healthcare",
  "Shopping",
  "Entertainment",
  "Travel",
  "Education",
  "Insurance",
  "Savings",
  "Investments",
  "Fees",
  "Taxes",
  "Cash",
  "Other",
];
const euro = new Intl.NumberFormat("en-IE", {
  style: "currency",
  currency: "EUR",
  maximumFractionDigits: 0,
});
const pct = (value: number) => `${value.toFixed(1)}%`;
const formatMoney = (value: string, currency: string) =>
  new Intl.NumberFormat("en-IE", { style: "currency", currency }).format(Number(value));
const categoryValue = (category: string) => category.toLowerCase();
const localIsoDate = () => {
  const now = new Date();
  const offset = now.getTimezoneOffset() * 60_000;
  return new Date(now.getTime() - offset).toISOString().slice(0, 10);
};

function useDebouncedValue(value: string, delayMs: number) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const timeoutId = window.setTimeout(() => setDebouncedValue(value), delayMs);
    return () => window.clearTimeout(timeoutId);
  }, [delayMs, value]);
  return debouncedValue;
}

function MetricCard({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: string;
  note: string;
  tone?: "good" | "risk";
}) {
  return (
    <article className={`metric ${tone ?? ""}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{note}</span>
    </article>
  );
}

function AllocationChart({ title, data }: { title: string; data: Allocation[] }) {
  return (
    <section className="panel allocation">
      <div className="panel-title">
        <div>
          <span className="eyebrow">EXPOSURE</span>
          <h3>{title}</h3>
        </div>
      </div>
      <div className="allocation-body">
        <ResponsiveContainer width="52%" height={210}>
          <PieChart>
            <Pie
              data={data}
              dataKey="weight"
              nameKey="label"
              innerRadius={55}
              outerRadius={84}
              paddingAngle={3}
            >
              {data.map((entry, index) => (
                <Cell key={entry.label} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => pct(Number(value) * 100)} />
          </PieChart>
        </ResponsiveContainer>
        <div className="legend">
          {data.map((entry, index) => (
            <div key={entry.label}>
              <i style={{ background: COLORS[index % COLORS.length] }} />
              <span>{entry.label}</span>
              <b>{pct(entry.weight * 100)}</b>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Dashboard({ analytics }: { analytics: Analytics }) {
  const positive = Number(analytics.unrealized_pnl_eur) >= 0;
  return (
    <>
      <div className="metrics-grid">
        <MetricCard
          label="Portfolio value"
          value={euro.format(Number(analytics.market_value_eur))}
          note={`As of ${analytics.as_of}`}
        />
        <MetricCard
          label="Unrealized P&L"
          value={`${positive ? "+" : ""}${euro.format(Number(analytics.unrealized_pnl_eur))}`}
          note={pct(analytics.unrealized_pnl_percent)}
          tone={positive ? "good" : "risk"}
        />
        <MetricCard
          label="Annualized volatility"
          value={pct(analytics.annualized_volatility_percent)}
          note="252-day estimate"
          tone="risk"
        />
        <MetricCard
          label="Maximum drawdown"
          value={pct(analytics.max_drawdown_percent)}
          note="Trailing period"
          tone="risk"
        />
        <MetricCard
          label="Concentration HHI"
          value={analytics.concentration_hhi.toFixed(3)}
          note={`${analytics.largest_position_symbol} · ${pct(analytics.largest_position_weight * 100)}`}
        />
      </div>
      <section className="panel performance">
        <div className="panel-title">
          <div>
            <span className="eyebrow">PORTFOLIO PATH</span>
            <h3>Current holdings, reconstructed</h3>
          </div>
          <span className="return-pill">
            {analytics.trailing_return_percent >= 0 ? "+" : ""}
            {pct(analytics.trailing_return_percent)}
          </span>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={analytics.value_series}>
            <defs>
              <linearGradient id="valueFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#7cf4c5" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#7cf4c5" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#203047" vertical={false} />
            <XAxis dataKey="date" stroke="#7890aa" tickLine={false} />
            <YAxis
              stroke="#7890aa"
              tickLine={false}
              tickFormatter={(value) => `€${Math.round(value / 1000)}k`}
            />
            <Tooltip formatter={(value) => euro.format(Number(value))} />
            <Area
              type="monotone"
              dataKey="value_eur"
              stroke="#7cf4c5"
              fill="url(#valueFill)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </section>
      <div className="chart-grid">
        <AllocationChart title="Asset class allocation" data={analytics.allocations.asset_class} />
        <AllocationChart title="Regional allocation" data={analytics.allocations.region} />
      </div>
      {analytics.risk_score ? <RiskScorePanel risk={analytics.risk_score} /> : null}
      <div className="warnings">
        {analytics.warnings.map((warning) => (
          <p key={warning}>ⓘ {warning}</p>
        ))}
      </div>
    </>
  );
}

function RiskScorePanel({ risk }: { risk: PortfolioRiskScore }) {
  const levelLabel = risk.level.replace("_", " ");
  return (
    <section className={`panel risk-score-panel risk-${risk.level}`}>
      <div className="risk-score-summary">
        <div className="risk-score-value">
          <span className="eyebrow">MARKET RISK INDICATOR</span>
          <strong>{risk.score.toFixed(1)}</strong>
          <small>/ 100 · {levelLabel}</small>
        </div>
        <div className="risk-drivers">
          <span className="eyebrow">MAIN DRIVERS</span>
          {risk.main_drivers.map((driver) => (
            <p key={driver.component}>
              <b>+{driver.contribution.toFixed(1)}</b>
              <span>{driver.explanation}</span>
            </p>
          ))}
        </div>
      </div>
      <p className="risk-interpretation">{risk.interpretation}</p>
      <div className="risk-dimensions">
        {[risk.diversification, risk.liquidity_resilience].map((dimension) => (
          <article key={dimension.key} className={`quality-${dimension.level}`}>
            <span className="eyebrow">{dimension.label}</span>
            <div><strong>{dimension.score.toFixed(0)}</strong><b>{dimension.level}</b></div>
            <div className="quality-track" aria-label={`${dimension.label} ${dimension.score.toFixed(0)} out of 100`}><i style={{ width: `${dimension.score}%` }} /></div>
            <small>{dimension.summary}</small>
          </article>
        ))}
      </div>
      <div className="risk-components">
        {risk.components.map((component) => (
          <article key={component.key}>
            <div>
              <b>{component.label}</b>
              <span>{component.score.toFixed(1)} / 100 · {(component.weight * 100).toFixed(0)}% weight</span>
            </div>
            <div className="risk-track" aria-label={`${component.label} score ${component.score.toFixed(1)} out of 100`}>
              <i style={{ width: `${component.score}%` }} />
            </div>
            <small>{component.summary}</small>
          </article>
        ))}
      </div>
      <details className="risk-methodology">
        <summary>Methodology and limitations</summary>
        <p>{risk.disclaimer}</p>
        <ul>{risk.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
        <small>{risk.methodology_version} · As of {risk.as_of}</small>
      </details>
    </section>
  );
}

function AddTransactionModal({
  accounts,
  initialAccountId,
  onClose,
}: {
  accounts: Account[];
  initialAccountId: string;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<TransactionCreate>({
    account_id: initialAccountId || accounts[0]?.id || "",
    booked_at: localIsoDate(),
    name: "",
    amount: "",
    currency: "EUR",
    transaction_type: "unspecified",
    category: "",
    fees: "0",
    taxes: "0",
  });
  const [categoryEdited, setCategoryEdited] = useState(false);
  const debouncedName = useDebouncedValue(form.name, 400);
  const debouncedCounterparty = useDebouncedValue(form.counterparty ?? "", 400);
  const hasClassificationAmount = Boolean(form.amount) && Number(form.amount) !== 0;
  const classification = useQuery({
    queryKey: [
      "transaction-classification",
      debouncedName,
      debouncedCounterparty,
      form.amount,
    ],
    queryFn: () =>
      api.classifyTransaction({
        description: debouncedName,
        amount: form.amount,
        counterparty: debouncedCounterparty || undefined,
      }),
    enabled: debouncedName.trim().length >= 3 && hasClassificationAmount,
    retry: false,
    staleTime: 30_000,
  });
  useEffect(() => {
    if (
      classification.data?.category &&
      !classification.data.needs_review &&
      !categoryEdited
    ) {
      setForm((current) => ({ ...current, category: classification.data.category ?? "" }));
    }
  }, [categoryEdited, classification.data]);
  const mutation = useMutation({
    mutationFn: () =>
      api.createTransaction({ ...form, category_confirmed: categoryEdited }),
    onSuccess: () => {
      onClose();
      void Promise.all([
        queryClient.invalidateQueries({ queryKey: ["transactions"] }),
        queryClient.invalidateQueries({ queryKey: ["accounts"] }),
        queryClient.invalidateQueries({ queryKey: ["portfolio-overview"] }),
        queryClient.invalidateQueries({ queryKey: ["analytics"] }),
      ]);
    },
  });
  const update = (key: keyof TransactionCreate, value: string) =>
    setForm((current) => ({ ...current, [key]: value }));
  const updateClassificationInput = (key: keyof TransactionCreate, value: string) => {
    setCategoryEdited(false);
    update(key, value);
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    mutation.mutate();
  };

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <form className="modal transaction-modal" onSubmit={submit} onMouseDown={(e) => e.stopPropagation()}>
        <span className="eyebrow">NEW TRANSACTION</span>
        <h2>Add transaction</h2>
        <p>
          Enter a signed cash flow. Use a negative amount for spending, fees, withdrawals, and
          security purchases.
        </p>
        <div className="form-grid">
          <label>
            Account
            <select
              required
              value={form.account_id}
              onChange={(event) => update("account_id", event.target.value)}
            >
              {accounts.map((account) => (
                <option key={account.id} value={account.id}>
                  {account.name}
                </option>
              ))}
            </select>
          </label>
          <label className="span-2">
            Name or description
            <input
              required
              maxLength={160}
              value={form.name}
              onChange={(event) => updateClassificationInput("name", event.target.value)}
              placeholder="Example: Weekly groceries"
            />
          </label>
          <label>
            Amount
            <input
              required
              type="number"
              step="0.01"
              value={form.amount}
              onChange={(event) => updateClassificationInput("amount", event.target.value)}
              placeholder="-45.90"
            />
          </label>
          <label>
            Date
            <input
              required
              type="date"
              value={form.booked_at}
              onChange={(event) => update("booked_at", event.target.value)}
            />
          </label>
          <label>
            Counterparty
            <input
              maxLength={160}
              value={form.counterparty ?? ""}
              onChange={(event) =>
                updateClassificationInput("counterparty", event.target.value)
              }
              placeholder="Example merchant"
            />
          </label>
          <label>
            Category
            <select
              aria-label="Category"
              value={form.category ?? ""}
              onChange={(event) => {
                setCategoryEdited(true);
                update("category", event.target.value);
              }}
            >
              <option value="">Uncategorized</option>
              {TRANSACTION_CATEGORIES.map((category) => (
                <option key={category} value={categoryValue(category)}>
                  {category}
                </option>
              ))}
            </select>
            <span className="classification-hint" aria-live="polite">
              {classification.isFetching
                ? "Finding a category suggestion…"
                : classification.isError
                  ? "Suggestion unavailable — choose a category manually."
                  : classification.data?.category
                    ? `${categoryEdited ? "Original suggestion" : "Suggested"}: ${classification.data.category} · ${
                        classification.data.classification_method === "ml"
                          ? `${Math.round((classification.data.confidence ?? 0) * 100)}% uncalibrated model score`
                          : classification.data.classification_method === "keyword_rule"
                            ? "text baseline rule"
                            : "deterministic rule"
                      }${classification.data.needs_review ? " · review recommended" : ""}`
                    : classification.data
                      ? `No reliable suggestion · ${classification.data.reason}`
                    : "Enter a description and non-zero amount to receive a suggestion."}
            </span>
          </label>
          <label className="span-2">
            Notes
            <textarea
              maxLength={500}
              value={form.notes ?? ""}
              onChange={(event) => update("notes", event.target.value)}
              placeholder="Optional local demo note"
            />
          </label>
        </div>
        {mutation.isError && <pre className="error">{mutation.error.message}</pre>}
        <div className="modal-actions">
          <button type="button" className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button disabled={mutation.isPending || !form.account_id || !form.amount || !form.name}>
            {mutation.isPending ? "Saving…" : "Add transaction"}
          </button>
        </div>
      </form>
    </div>
  );
}

function ActivitySection({ initialAccountId }: { initialAccountId: string }) {
  const [showAdd, setShowAdd] = useState(false);
  const [filters, setFilters] = useState<TransactionFilters>({
    account_id: initialAccountId || undefined,
    limit: 10,
    offset: 0,
  });
  const accounts = useQuery({ queryKey: ["accounts"], queryFn: api.accounts });
  const transactions = useQuery({
    queryKey: ["transactions", filters],
    queryFn: () => api.transactions(filters),
  });
  const visibleAccounts = accounts.data?.filter(
    (account) => account.account_type !== "brokerage" || account.id === initialAccountId,
  );
  const accountNames = new Map(accounts.data?.map((account) => [account.id, account.name]));
  const setFilter = (key: keyof TransactionFilters, value: string) =>
    setFilters((current) => ({ ...current, [key]: value || undefined, offset: 0 }));
  const canGoNext =
    Boolean(transactions.data) &&
    filters.offset + filters.limit < (transactions.data?.total ?? 0);

  useEffect(() => {
    setFilters((current) => ({
      ...current,
      account_id: initialAccountId || undefined,
      offset: 0,
    }));
  }, [initialAccountId]);

  return (
    <section className="activity-section" id="activity">
      <header className="section-header">
        <div>
          <span className="eyebrow">ACTIVITY</span>
          <h2>Transactions</h2>
          <p>Portfolio orders and account cash movements in one ledger.</p>
        </div>
        <div className="header-actions">
          <button onClick={() => setShowAdd(true)} disabled={!accounts.data?.length}>
            Add transaction
          </button>
        </div>
      </header>
      <details className="panel filter-panel">
        <summary>Filter activity</summary>
        <div className="filter-grid">
          <label>
            Account
            <select
              aria-label="Filter by account"
              value={filters.account_id ?? ""}
              onChange={(event) => setFilter("account_id", event.target.value)}
            >
              {visibleAccounts?.map((account) => (
                <option key={account.id} value={account.id}>
                  {account.name}{account.portfolio_name ? ` · ${account.portfolio_name}` : ""} · {formatMoney(account.current_balance, account.currency)} ({account.transaction_count})
                </option>
              ))}
            </select>
          </label>
          <label>
            Cash flow
            <select
              aria-label="Filter by cash flow"
              value={filters.cash_flow ?? ""}
              onChange={(event) => setFilter("cash_flow", event.target.value)}
            >
              <option value="">All flows</option>
              <option value="inflow">Incoming</option>
              <option value="outflow">Outgoing</option>
            </select>
          </label>
          <label>
            Category
            <select
              aria-label="Filter by category"
              value={filters.category ?? ""}
              onChange={(event) => setFilter("category", event.target.value)}
            >
              <option value="">All categories</option>
              {TRANSACTION_CATEGORIES.map((category) => (
                <option key={category}>{category}</option>
              ))}
            </select>
          </label>
          <label>
            From
            <input
              aria-label="Filter from date"
              type="date"
              value={filters.date_from ?? ""}
              onChange={(event) => setFilter("date_from", event.target.value)}
            />
          </label>
          <label>
            To
            <input
              aria-label="Filter to date"
              type="date"
              value={filters.date_to ?? ""}
              onChange={(event) => setFilter("date_to", event.target.value)}
            />
          </label>
        </div>
      </details>
      {accounts.isError || transactions.isError ? (
        <div className="error">
          {(accounts.error as Error | null)?.message ??
            (transactions.error as Error | null)?.message ??
            "Could not load transactions."}
        </div>
      ) : null}
      {transactions.isLoading && <div className="loading">Loading transaction history…</div>}
      {transactions.data && (
        <section className="panel transaction-panel">
          <div className="panel-title">
            <div>
              <span className="eyebrow">LEDGER</span>
              <h3>{transactions.data.total} transactions</h3>
            </div>
            <span className="page-meta">
              {transactions.data.total
                ? `${transactions.data.offset + 1}–${Math.min(
                    transactions.data.offset + transactions.data.items.length,
                    transactions.data.total,
                  )}`
                : "0"}
            </span>
          </div>
          <div className="table-wrap">
            <table className="transaction-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Description</th>
                  <th>Account</th>
                  <th>Category</th>
                  <th>Type</th>
                  <th>Cash flow</th>
                  <th>Amount</th>
                </tr>
              </thead>
              <tbody>
                {transactions.data.items.map((transaction) => (
                  <tr key={transaction.id}>
                    <td>{transaction.booked_at}</td>
                    <td>
                      <b>{transaction.name}</b>
                      <small>{transaction.counterparty ?? transaction.security_symbol ?? "—"}</small>
                    </td>
                    <td>{accountNames.get(transaction.account_id) ?? "Unknown account"}</td>
                    <td>
                      <span className="category-pill">{transaction.category ?? "Uncategorized"}</span>
                    </td>
                    <td>{transaction.transaction_type.replaceAll("_", " ")}</td>
                    <td>{Number(transaction.amount) >= 0 ? "Incoming" : "Outgoing"}</td>
                    <td className={Number(transaction.amount) >= 0 ? "positive" : "negative"}>
                      {formatMoney(transaction.amount, transaction.currency)}
                    </td>
                  </tr>
                ))}
                {!transactions.data.items.length && (
                  <tr>
                    <td colSpan={7} className="empty-state">
                      No transactions match the selected filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <button
              className="secondary"
              disabled={filters.offset === 0}
              onClick={() =>
                setFilters((current) => ({
                  ...current,
                  offset: Math.max(0, current.offset - current.limit),
                }))
              }
            >
              Previous
            </button>
            <button
              className="secondary"
              disabled={!canGoNext}
              onClick={() =>
                setFilters((current) => ({
                  ...current,
                  offset: current.offset + current.limit,
                }))
              }
            >
              Next
            </button>
          </div>
        </section>
      )}
      {showAdd && visibleAccounts && (
        <AddTransactionModal
          accounts={visibleAccounts}
          initialAccountId={filters.account_id ?? visibleAccounts[0]?.id ?? ""}
          onClose={() => setShowAdd(false)}
        />
      )}
    </section>
  );
}

function PortfolioView() {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState(() =>
    window.localStorage.getItem("financial-ai:selected-portfolio") ?? "",
  );
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [startingCash, setStartingCash] = useState("10000");
  const [marketDataMode, setMarketDataMode] = useState<"demo" | "external">(
    "demo",
  );
  const portfolios = useQuery({ queryKey: ["portfolios"], queryFn: api.portfolios });
  const marketStatus = useQuery({
    queryKey: ["market-data-status"],
    queryFn: api.marketDataStatus,
  });
  const visiblePortfolios = portfolios.data?.filter(
    (portfolio) =>
      portfolio.kind !== "demo" ||
      [
        "Diversified Global Portfolio",
        "Technology Concentration",
        "Defensive ETF Portfolio",
      ].includes(portfolio.name),
  );
  const selectedPortfolio = portfolios.data?.find(
    (portfolio) => portfolio.id === selected,
  );
  useEffect(() => {
    if (!portfolios.data?.length) return;
    if (!visiblePortfolios?.some((portfolio) => portfolio.id === selected)) {
      const preferred =
        visiblePortfolios?.find((portfolio) => portfolio.kind !== "demo") ??
        visiblePortfolios?.[0] ??
        portfolios.data[0];
      setSelected(preferred.id);
    }
  }, [portfolios.data, selected, visiblePortfolios]);
  useEffect(() => {
    if (selected) window.localStorage.setItem("financial-ai:selected-portfolio", selected);
  }, [selected]);
  const analytics = useQuery({
    queryKey: ["analytics", selected],
    queryFn: () => api.analytics(selected),
    enabled: Boolean(selected),
  });
  const creator = useMutation({
    mutationFn: () =>
      api.createPortfolio({
        name,
        base_currency: "EUR",
        starting_cash: startingCash,
        market_data_mode: marketDataMode,
      }),
    onSuccess: async (portfolio) => {
      await queryClient.invalidateQueries({ queryKey: ["portfolios"] });
      setSelected(portfolio.id);
      setShowCreate(false);
      setName("");
      setStartingCash("10000");
      setMarketDataMode("demo");
    },
  });

  return (
    <main className="portfolio-workspace">
      <header className="workspace-header">
        <div>
          <span className="eyebrow">PORTFOLIO</span>
          <h1>{selectedPortfolio?.name ?? "Your investments"}</h1>
          <p>Positions, risk and account activity in one place.</p>
        </div>
        <div className="header-actions">
          <select
            aria-label="Select portfolio"
            value={selected}
            onChange={(event) => setSelected(event.target.value)}
          >
            {visiblePortfolios?.map((portfolio) => (
              <option key={portfolio.id} value={portfolio.id}>
                {portfolio.name} {portfolio.kind === "demo" ? "· Demo" : ""}
              </option>
            ))}
          </select>
          <button className="secondary" onClick={() => setShowCreate(true)}>
            New portfolio
          </button>
        </div>
      </header>
      <details className="data-context">
        <summary>
          {selectedPortfolio?.market_data_mode === "external"
            ? "External market data · Paper portfolio"
            : "Demo market data · Paper portfolio"}
        </summary>
        <p>
          {selectedPortfolio?.market_data_mode === "external"
            ? "Quotes and history are retrieved from Alpaca. Orders remain simulated and no brokerage account is connected."
            : "Prices are deterministic synthetic observations for reproducible product exploration."}
        </p>
      </details>
      {portfolios.isError && (
        <div className="error">Could not reach the API. Start the FastAPI service on port 8000.</div>
      )}
      {analytics.isLoading && <div className="loading">Calculating portfolio analytics…</div>}
      {analytics.isError && <div className="error">{analytics.error.message}</div>}
      {analytics.data && <Dashboard analytics={analytics.data} />}
      {selected && <PortfolioTradingPanel portfolioId={selected} />}
      <ActivitySection initialAccountId={selectedPortfolio?.account_id ?? ""} />
      {showCreate && (
        <div className="modal-backdrop" onMouseDown={() => setShowCreate(false)}>
          <form
            className="modal"
            onSubmit={(event) => {
              event.preventDefault();
              creator.mutate();
            }}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <span className="eyebrow">NEW PAPER PORTFOLIO</span>
            <h2>Choose the market-data source</h2>
            <label>
              Portfolio name
              <input
                required
                maxLength={120}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="My market portfolio"
              />
            </label>
            <label>
              Starting cash (EUR)
              <input
                required
                min="1"
                step="0.01"
                type="number"
                value={startingCash}
                onChange={(event) => setStartingCash(event.target.value)}
              />
            </label>
            <label>
              Market data
              <select
                value={marketDataMode}
                onChange={(event) =>
                  setMarketDataMode(event.target.value as "demo" | "external")
                }
              >
                <option value="demo">Deterministic demo data</option>
                <option
                  value="external"
                  disabled={!marketStatus.data?.external_available}
                >
                  Alpaca
                  {marketStatus.data?.external_available
                    ? ""
                    : " · backend credentials required"}
                </option>
              </select>
            </label>
            <p>
              {marketDataMode === "external"
                ? "Search and adjusted daily bars come from Alpaca. Orders remain simulated."
                : "No API key is required and all prices remain reproducible."}
            </p>
            {creator.isError && <div className="error">{creator.error.message}</div>}
            <div className="modal-actions">
              <button
                type="button"
                className="secondary"
                onClick={() => setShowCreate(false)}
              >
                Cancel
              </button>
              <button disabled={!name || creator.isPending}>
                {creator.isPending ? "Creating…" : "Create portfolio"}
              </button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

export default function App() {
  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="brand">
          <div className="brand-mark">F</div>
          <strong>Financial AI</strong>
        </div>
        <a href="#activity">Activity</a>
      </div>
      <PortfolioView />
    </div>
  );
}
