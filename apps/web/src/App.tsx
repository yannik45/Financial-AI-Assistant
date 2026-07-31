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
import type {
  Account,
  Allocation,
  Analytics,
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
      <section className="panel">
        <div className="panel-title">
          <div>
            <span className="eyebrow">HOLDINGS</span>
            <h3>Position contribution</h3>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Market value</th>
                <th>Cost basis</th>
                <th>P&amp;L</th>
                <th>Weight</th>
              </tr>
            </thead>
            <tbody>
              {analytics.positions.map((position) => (
                <tr key={position.symbol}>
                  <td>
                    <b>{position.symbol}</b>
                  </td>
                  <td>{euro.format(Number(position.market_value_eur))}</td>
                  <td>{euro.format(Number(position.cost_basis_eur))}</td>
                  <td className={Number(position.pnl_eur) >= 0 ? "positive" : "negative"}>
                    {euro.format(Number(position.pnl_eur))}
                  </td>
                  <td>{pct(position.weight * 100)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <div className="warnings">
        {analytics.warnings.map((warning) => (
          <p key={warning}>ⓘ {warning}</p>
        ))}
      </div>
    </>
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
    booked_at: new Date().toISOString().slice(0, 10),
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
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["transactions"] }),
        queryClient.invalidateQueries({ queryKey: ["accounts"] }),
      ]);
      onClose();
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

function TransactionsView() {
  const [showAdd, setShowAdd] = useState(false);
  const [filters, setFilters] = useState<TransactionFilters>({ limit: 10, offset: 0 });
  const accounts = useQuery({ queryKey: ["accounts"], queryFn: api.accounts });
  const transactions = useQuery({
    queryKey: ["transactions", filters],
    queryFn: () => api.transactions(filters),
  });
  const accountNames = new Map(accounts.data?.map((account) => [account.id, account.name]));
  const setFilter = (key: keyof TransactionFilters, value: string) =>
    setFilters((current) => ({ ...current, [key]: value || undefined, offset: 0 }));
  const canGoNext =
    Boolean(transactions.data) &&
    filters.offset + filters.limit < (transactions.data?.total ?? 0);

  return (
    <main>
      <header>
        <div>
          <span className="eyebrow">TRANSACTION LEDGER</span>
          <h1>Accounts and transaction history</h1>
          <p>Synthetic local data prepared for transparent categorization and risk models.</p>
        </div>
        <div className="header-actions">
          <button onClick={() => setShowAdd(true)} disabled={!accounts.data?.length}>
            Add transaction
          </button>
        </div>
      </header>
      <div className="demo-banner">
        <b>SYNTHETIC DEMO TRANSACTIONS</b>
        <span>Educational use only · No real customer data</span>
      </div>
      <section className="panel filter-panel">
        <div className="filter-grid">
          <label>
            Account
            <select
              aria-label="Filter by account"
              value={filters.account_id ?? ""}
              onChange={(event) => setFilter("account_id", event.target.value)}
            >
              <option value="">All accounts</option>
              {accounts.data?.map((account) => (
                <option key={account.id} value={account.id}>
                  {account.name} ({account.transaction_count})
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
      </section>
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
                    <td>{Number(transaction.amount) >= 0 ? "Incoming" : "Outgoing"}</td>
                    <td className={Number(transaction.amount) >= 0 ? "positive" : "negative"}>
                      {formatMoney(transaction.amount, transaction.currency)}
                    </td>
                  </tr>
                ))}
                {!transactions.data.items.length && (
                  <tr>
                    <td colSpan={6} className="empty-state">
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
      {showAdd && accounts.data && (
        <AddTransactionModal
          accounts={accounts.data}
          initialAccountId={filters.account_id ?? ""}
          onClose={() => setShowAdd(false)}
        />
      )}
    </main>
  );
}

function PortfolioView() {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState("");
  const [showImport, setShowImport] = useState(false);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const portfolios = useQuery({ queryKey: ["portfolios"], queryFn: api.portfolios });
  useEffect(() => {
    if (!selected && portfolios.data?.length) setSelected(portfolios.data[0].id);
  }, [portfolios.data, selected]);
  const analytics = useQuery({
    queryKey: ["analytics", selected],
    queryFn: () => api.analytics(selected),
    enabled: Boolean(selected),
  });
  const importer = useMutation({
    mutationFn: () => api.importPortfolio(name, file!),
    onSuccess: async (portfolio) => {
      await queryClient.invalidateQueries({ queryKey: ["portfolios"] });
      setSelected(portfolio.id);
      setShowImport(false);
      setName("");
      setFile(null);
    },
  });

  return (
    <main>
      <header>
        <div>
          <span className="eyebrow">PORTFOLIO INTELLIGENCE</span>
          <h1>Risk, concentration and performance</h1>
          <p>Deterministic analytics for transparent financial exploration.</p>
        </div>
        <div className="header-actions">
          <select
            aria-label="Select portfolio"
            value={selected}
            onChange={(event) => setSelected(event.target.value)}
          >
            {portfolios.data?.map((portfolio) => (
              <option key={portfolio.id} value={portfolio.id}>
                {portfolio.name} {portfolio.kind === "demo" ? "· Demo" : ""}
              </option>
            ))}
          </select>
          <button onClick={() => setShowImport(true)}>Import CSV</button>
        </div>
      </header>
      <div className="demo-banner">
        <b>SYNTHETIC DEMO MARKET DATA</b>
        <span>Educational use only · Not financial advice</span>
      </div>
      {portfolios.isError && (
        <div className="error">Could not reach the API. Start the FastAPI service on port 8000.</div>
      )}
      {analytics.isLoading && <div className="loading">Calculating portfolio analytics…</div>}
      {analytics.isError && <div className="error">{analytics.error.message}</div>}
      {analytics.data && <Dashboard analytics={analytics.data} />}
      {showImport && (
        <div className="modal-backdrop" onMouseDown={() => setShowImport(false)}>
          <form
            className="modal"
            onSubmit={(event) => {
              event.preventDefault();
              importer.mutate();
            }}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <span className="eyebrow">NEW PORTFOLIO</span>
            <h2>Import positions</h2>
            <p>
              Use the exact market catalog metadata. The complete file is rejected when any row is
              invalid.
            </p>
            <label>
              Portfolio name
              <input
                required
                maxLength={120}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="My demo portfolio"
              />
            </label>
            <label>
              UTF-8 CSV
              <input
                required
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            {importer.isError && <pre className="error">{importer.error.message}</pre>}
            <div className="modal-actions">
              <a href="/portfolio_template.csv" download>
                Download template
              </a>
              <button type="button" className="secondary" onClick={() => setShowImport(false)}>
                Cancel
              </button>
              <button disabled={!file || !name || importer.isPending}>
                {importer.isPending ? "Importing…" : "Import portfolio"}
              </button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

export default function App() {
  const [view, setView] = useState<"portfolio" | "transactions">("portfolio");
  return (
    <div className="app-shell">
      <aside>
        <div className="brand">
          <div className="brand-mark">F</div>
          <div>
            <strong>FINANCIAL AI</strong>
            <span>ASSISTANT</span>
          </div>
        </div>
        <nav>
          <button
            className={view === "portfolio" ? "active" : ""}
            onClick={() => setView("portfolio")}
          >
            ⌁ Portfolio overview
          </button>
          <button
            className={view === "transactions" ? "active" : ""}
            onClick={() => setView("transactions")}
          >
            ◇ Transactions
          </button>
          <button className="disabled" disabled>
            ◈ AI assistant <small>Later phase</small>
          </button>
        </nav>
        <div className="sidebar-note">
          <span>LOCAL MODE</span>
          <p>No live market data or cloud services are active.</p>
        </div>
      </aside>
      {view === "portfolio" ? <PortfolioView /> : <TransactionsView />}
    </div>
  );
}
