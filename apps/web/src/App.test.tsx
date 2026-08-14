import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import App from "./App";

const accounts = [
  {
    id: "checking-id",
    name: "Main Checking",
    account_type: "checking",
    currency: "EUR",
    kind: "demo",
    opening_balance: "0.00",
    current_balance: "5769.17",
    portfolio_id: null,
    portfolio_name: null,
    created_at: "2026-01-01T00:00:00",
    transaction_count: 1,
  },
];
const transactions = {
  items: [
    {
      id: "transaction-id",
      account_id: "checking-id",
      booked_at: "2026-03-10",
      name: "Supermarket",
      amount: "-92.18",
      currency: "EUR",
      transaction_type: "card_payment",
      counterparty: "Fresh Market",
      category: "Groceries",
      notes: null,
      source: "demo",
      security_symbol: null,
      quantity: null,
      unit_price: null,
      fees: "0.00",
      taxes: "0.00",
      created_at: "2026-03-10T00:00:00",
    },
  ],
  total: 1,
  limit: 10,
  offset: 0,
};

const jsonResponse = (payload: unknown) =>
  Promise.resolve({ ok: true, json: () => Promise.resolve(payload) } as Response);

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: string | URL | Request) => {
      const url = String(input);
      if (url.includes("/v1/market/status")) return jsonResponse({ demo_available: true, external_available: false, external_provider: "alpaca" });
      if (url.includes("/v1/accounts")) return jsonResponse(accounts);
      if (url.includes("/v1/transactions/classify"))
        return jsonResponse({
          category: "groceries",
          route: "expense_model",
          classification_method: "ml",
          confidence: 0.81,
          needs_review: false,
          reason: "Expense category predicted by the versioned model artifact.",
          taxonomy_version: "transaction-categories-v1",
          model_version: "test-model-v1",
        });
      if (url.includes("/v1/transactions")) return jsonResponse(transactions);
      return jsonResponse([]);
    }),
  );
});

function renderApp() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <App />
    </QueryClientProvider>,
  );
}

test("renders the portfolio view and import action", () => {
  renderApp();
  expect(screen.getByText("FINANCIAL AI")).toBeInTheDocument();
  expect(screen.queryByText("NORTHSTAR")).not.toBeInTheDocument();
  expect(screen.getByText("Risk, concentration and performance")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Import CSV" })).toBeInTheDocument();
  expect(screen.getByText("SYNTHETIC DEMO MARKET DATA")).toBeInTheDocument();
});

test("keeps external portfolios unavailable without a server-side API key", async () => {
  renderApp();
  fireEvent.click(screen.getByRole("button", { name: "New portfolio" }));

  const selector = await screen.findByLabelText("Market data");
  expect(within(selector).getByRole("option", { name: /Alpaca/ })).toBeDisabled();
  expect(screen.getByText("No API key is required and all prices remain reproducible.")).toBeInTheDocument();
});

test("submits a portfolio order that updates the shared brokerage ledger", async () => {
  let overviewRequests = 0;
  const instrument = {
    id: "instrument-id",
    provider: "demo",
    symbol: "WORLD-ETF",
    name: "Global Equity Demo ETF",
    exchange: "DEMO",
    currency: "EUR",
    asset_class: "Equity ETF",
    region: "Global",
    is_active: true,
    updated_at: "2026-08-01T10:00:00",
  };
  const summary = {
    id: "portfolio-id",
    name: "My Portfolio",
    base_currency: "EUR",
    kind: "manual",
    market_data_mode: "demo",
    created_at: "2026-08-01T10:00:00",
    position_count: 0,
    account_id: "brokerage-id",
  };
  const overview = {
    id: "portfolio-id",
    name: "My Portfolio",
    base_currency: "EUR",
    market_data_mode: "demo",
    opening_cash: "10000.00",
    created_at: "2026-08-01T10:00:00",
    trade_count: 0,
    cash_balance: "10000.00",
    holdings_value: "0.00",
    total_equity: "10000.00",
    total_pnl: "0.00",
    realized_pnl: "0.00",
    holdings: [
      {
        instrument,
        quantity: "4.00",
        average_cost: "100.00",
        latest_price: "110.00",
        market_value: "440.00",
        unrealized_pnl: "40.00",
        weight: 1,
        price_observed_on: "2026-08-01",
        price_source: "demo",
        quote_is_stale: false,
      },
    ],
    trades: [
      {
        id: "existing-trade",
        client_order_id: "existing-order",
        side: "buy",
        quantity: "4.00",
        unit_price: "110.00",
        instrument_currency: "EUR",
        settlement_amount: "440.00",
        fees: "0.00",
        currency: "EUR",
        booked_at: "2026-08-01",
        price_observed_on: "2026-06-30",
        price_source: "demo",
        executed_at: "2026-08-01T10:00:00Z",
        instrument,
      },
    ],
    warnings: ["Simulation only: no real order is placed."],
  };
  const analytics = {
    portfolio_id: "portfolio-id",
    as_of: "2026-06-30",
    data_version: "test-v1",
    market_value_eur: "440.00",
    cost_basis_eur: "400.00",
    unrealized_pnl_eur: "40.00",
    unrealized_pnl_percent: 10,
    trailing_return_percent: 5,
    annualized_volatility_percent: 12,
    max_drawdown_percent: -9,
    concentration_hhi: 1,
    largest_position_symbol: "WORLD-ETF",
    largest_position_weight: 1,
    positions: [],
    allocations: { asset_class: [], sector: [], region: [], currency: [] },
    value_series: [],
    warnings: [],
    risk_score: {
      score: 58.4,
      level: "moderate",
      methodology_version: "portfolio-risk-score-v2",
      as_of: "2026-06-30",
      components: [
        {
          key: "volatility",
          label: "Historical portfolio volatility",
          score: 48,
          weight: 0.55,
          contribution: 26.4,
          raw_value: 12,
          raw_unit: "% annualized",
          summary: "Annualized historical portfolio volatility is 12.0%.",
          details: {},
        },
      ],
      main_drivers: [
        {
          component: "volatility",
          contribution: 26.4,
          explanation: "Annualized historical portfolio volatility is 12.0%.",
        },
      ],
      diversification: { key: "diversification", label: "Diversification quality", score: 75, level: "strong", summary: "Broad-market exposure receives a limited look-through credit.", details: {} },
      liquidity_resilience: { key: "liquidity_resilience", label: "Liquidity resilience", score: 20, level: "weak", summary: "Brokerage cash is 0.0% of total equity.", details: {} },
      interpretation: "The portfolio has moderate measured market risk and strong diversification.",
      disclaimer: "Heuristic market-risk indicator; not investment advice.",
      limitations: ["Historical observations cannot predict future losses."],
    },
  };
  vi.mocked(fetch).mockImplementation((input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/v1/market/instruments/instrument-id/quote")) {
      return jsonResponse({
        instrument,
        observed_on: "2026-06-30",
        close: "110.00",
        adjusted_close: "110.00",
        volume: "1000",
        source: "demo",
        retrieved_at: "2026-08-01T10:00:00",
        is_stale: false,
      });
    }
    if (url.includes("/v1/market/instruments?")) return jsonResponse([instrument]);
    if (url.endsWith("/v1/portfolios/portfolio-id/orders") && init?.method === "POST") {
      return jsonResponse({ id: "trade-id" });
    }
    if (url.endsWith("/v1/portfolios/portfolio-id/overview")) {
      overviewRequests += 1;
      if (overviewRequests > 1) return new Promise<Response>(() => undefined);
      return jsonResponse(overview);
    }
    if (url.endsWith("/v1/portfolios")) return jsonResponse([summary]);
    if (url.endsWith("/v1/portfolios/portfolio-id/analytics")) return jsonResponse(analytics);
    return jsonResponse([]);
  });

  renderApp();
  expect(await screen.findByText("58.4")).toBeInTheDocument();
  expect(screen.getByText("Diversification quality")).toBeInTheDocument();
  expect(screen.getByText("Historical portfolio volatility")).toBeInTheDocument();
  expect(await screen.findByText("2026-08-01")).toBeInTheDocument();
  expect(screen.getByText("Price observed 2026-06-30")).toBeInTheDocument();
  expect(screen.getByLabelText("Demo instrument")).toBeInTheDocument();
  expect(
    screen.getByText("All selectable instruments use deterministic synthetic prices."),
  ).toBeInTheDocument();
  fireEvent.click(await screen.findByRole("button", { name: "Buy more" }));
  expect(screen.getByLabelText("Side")).toHaveValue("buy");
  expect(screen.getByLabelText("Quantity")).toHaveValue(1);
  await waitFor(() =>
    expect(screen.getByLabelText("Selected instrument price")).toHaveTextContent("€110.00"),
  );
  expect(screen.getByLabelText("Quantity")).toHaveAttribute("step", "1");
  expect(screen.getByLabelText("Quantity")).toHaveAttribute("min", "1");
  fireEvent.click(screen.getByRole("button", { name: "Buy position" }));

  await waitFor(() => {
    const call = vi.mocked(fetch).mock.calls.find(([input, init]) =>
      String(input).includes("/v1/portfolios/portfolio-id/orders") && init?.method === "POST",
    );
    expect(call).toBeDefined();
    const payload = JSON.parse(String(call?.[1]?.body));
    expect(payload).toMatchObject({ instrument_id: "instrument-id", side: "buy", quantity: "1" });
    expect(payload).not.toHaveProperty("unit_price");
    expect(payload.client_order_id).toBeTruthy();
  });
  await waitFor(() => {
    expect(screen.queryByRole("button", { name: "Executing…" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Buy position" })).not.toBeInTheDocument();
  });
});

test("opens the transaction history and add transaction form", async () => {
  renderApp();
  fireEvent.click(screen.getByRole("button", { name: /Transactions/ }));

  expect(await screen.findByText("Accounts and transaction history")).toBeInTheDocument();
  expect(await screen.findByText("Supermarket")).toBeInTheDocument();
  expect(screen.getAllByText("Groceries")).toHaveLength(2);
  expect(screen.getByText("-€92.18")).toBeInTheDocument();

  const addButton = screen.getByRole("button", { name: "Add transaction" });
  await waitFor(() => expect(addButton).toBeEnabled());
  fireEvent.click(addButton);
  const modal = screen.getByRole("heading", { name: "Add transaction" }).closest("form")!;
  expect(within(modal).queryByLabelText("Transaction type")).not.toBeInTheDocument();
  expect(within(modal).getByLabelText("Category")).toHaveValue("");
  expect(within(modal).getByRole("option", { name: "Transport" })).toBeInTheDocument();
});

test("suggests a category and preserves a user correction when saving", async () => {
  renderApp();
  fireEvent.click(screen.getByRole("button", { name: /Transactions/ }));
  const addButton = await screen.findByRole("button", { name: "Add transaction" });
  await waitFor(() => expect(addButton).toBeEnabled());
  fireEvent.click(addButton);

  const modal = screen.getByRole("heading", { name: "Add transaction" }).closest("form")!;
  fireEvent.change(within(modal).getByLabelText("Name or description"), {
    target: { value: "Weekly supermarket purchase" },
  });
  fireEvent.change(within(modal).getByLabelText("Amount"), {
    target: { value: "-20.00" },
  });
  await waitFor(
    () => expect(within(modal).getByLabelText("Category")).toHaveValue("groceries"),
    { timeout: 2_000 },
  );
  expect(within(modal).getByText(/Suggested: groceries/)).toBeInTheDocument();
  const classifyCall = vi.mocked(fetch).mock.calls.find(([input]) =>
    String(input).endsWith("/v1/transactions/classify"),
  );
  const classificationPayload = JSON.parse(String(classifyCall?.[1]?.body));
  expect(classificationPayload).toMatchObject({
    description: "Weekly supermarket purchase",
    amount: "-20.00",
  });
  expect(classificationPayload).not.toHaveProperty("transaction_type");

  fireEvent.change(within(modal).getByLabelText("Category"), {
    target: { value: "dining" },
  });
  expect(within(modal).getByLabelText("Category")).toHaveValue("dining");
  expect(within(modal).getByText(/Original suggestion: groceries/)).toBeInTheDocument();

  fireEvent.click(within(modal).getByRole("button", { name: "Add transaction" }));

  await waitFor(() => {
    const createCall = vi.mocked(fetch).mock.calls.find(([input, init]) => {
      const url = String(input);
      return url.endsWith("/v1/transactions") && init?.method === "POST";
    });
    expect(createCall).toBeDefined();
    expect(JSON.parse(String(createCall?.[1]?.body))).toMatchObject({
      category: "dining",
      category_confirmed: true,
    });
  });
});

test("does not auto-apply a suggestion that requires review", async () => {
  vi.mocked(fetch).mockImplementation((input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("/v1/accounts")) return jsonResponse(accounts);
    if (url.includes("/v1/transactions/classify"))
      return jsonResponse({
        category: "housing",
        route: "expense_model",
        classification_method: "ml",
        confidence: 0.42,
        needs_review: true,
        reason: "Model confidence is below the review threshold.",
        taxonomy_version: "transaction-categories-v1",
        model_version: "test-model-v1",
      });
    if (url.includes("/v1/transactions")) return jsonResponse(transactions);
    return jsonResponse([]);
  });
  renderApp();
  fireEvent.click(screen.getByRole("button", { name: /Transactions/ }));
  const addButton = await screen.findByRole("button", { name: "Add transaction" });
  await waitFor(() => expect(addButton).toBeEnabled());
  fireEvent.click(addButton);
  const modal = screen.getByRole("heading", { name: "Add transaction" }).closest("form")!;
  fireEvent.change(within(modal).getByLabelText("Name or description"), {
    target: { value: "Unknown payment" },
  });
  fireEvent.change(within(modal).getByLabelText("Amount"), {
    target: { value: "-950" },
  });

  await waitFor(() =>
    expect(within(modal).getByText(/review recommended/i)).toBeInTheDocument(),
  );
  expect(within(modal).getByLabelText("Category")).toHaveValue("");
  expect(within(modal).getByText(/uncalibrated model score/i)).toBeInTheDocument();
});
