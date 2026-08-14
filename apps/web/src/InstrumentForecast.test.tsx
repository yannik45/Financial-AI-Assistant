import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import InstrumentForecast from "./InstrumentForecast";

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderForecast(onClose = vi.fn()) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <InstrumentForecast instrumentId="instrument-1" symbol="AAPL" onClose={onClose} />
    </QueryClientProvider>,
  );
  return onClose;
}

test("renders an instrument forecast with freshness and feed provenance", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          symbol: "AAPL",
          observed_on: "2026-08-13",
          horizon_trading_days: 20,
          predicted_annualized_volatility: 0.237,
          annualized: true,
          model_version: "market-volatility-xgboost-v1",
          source: "alpaca:iex",
          retrieved_at: "2026-08-14T08:00:00Z",
          data_status: "current",
          training_source_feed: "sip",
          feed_match: false,
        }),
    }),
  );

  const onClose = renderForecast();

  expect(await screen.findByText("6.7%")).toBeInTheDocument();
  expect(screen.getByText("23.7% annualized")).toBeInTheDocument();
  expect(screen.getByText("Expected volatility over the next 20 trading days")).toBeInTheDocument();
  expect(screen.getByText("Current market data")).toBeInTheDocument();
  expect(screen.getByText(/trained on SIP data/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(onClose).toHaveBeenCalledOnce();
});

test("does not offer retry when the local model is unavailable", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: () =>
        Promise.resolve({
          detail: {
            code: "market_forecast_model_unavailable",
            message: "Run the model build command first.",
          },
        }),
    }),
  );

  renderForecast();

  expect(await screen.findByText("Forecast model is not initialized")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
});
