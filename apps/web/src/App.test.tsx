import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, test, expect } from "vitest";
import App from "./App";

vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve([]) })));

test("renders the product heading and import action", async () => {
  render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
  expect(screen.getByText("Risk, concentration and performance")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Import CSV" })).toBeInTheDocument();
  expect(screen.getByText("SYNTHETIC DEMO MARKET DATA")).toBeInTheDocument();
});

