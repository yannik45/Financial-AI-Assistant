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
  expect(within(modal).getByLabelText("Category")).toHaveValue("");
  expect(within(modal).getByRole("option", { name: "Transport" })).toBeInTheDocument();
});
