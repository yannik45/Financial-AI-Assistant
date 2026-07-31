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
