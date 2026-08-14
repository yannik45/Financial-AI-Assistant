import { afterEach, expect, test, vi } from "vitest";
import { ApiError, api } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

test("requests an instrument volatility forecast", async () => {
  const forecast = {
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
  };
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve(forecast),
  });
  vi.stubGlobal("fetch", fetchMock);

  await expect(api.marketVolatilityForecast("instrument-1")).resolves.toEqual(forecast);
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/v1/market/instruments/instrument-1/volatility-forecast",
    undefined,
  );
});

test("preserves structured API error status, code, and message", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: () =>
        Promise.resolve({
          detail: {
            code: "market_forecast_model_unavailable",
            message: "Forecast model is not initialized",
          },
        }),
    }),
  );

  const error = await api.marketVolatilityForecast("instrument-1").catch((reason) => reason);

  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({
    status: 503,
    code: "market_forecast_model_unavailable",
    message: "Forecast model is not initialized",
  });
});

test("retains field-detail formatting for validation errors", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: () =>
        Promise.resolve({
          details: [{ row: 2, field: "quantity", message: "Must be positive" }],
        }),
    }),
  );

  const error = await api.marketVolatilityForecast("instrument-1").catch((reason) => reason);

  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({
    status: 422,
    message: "Row 2, quantity: Must be positive",
  });
});
