import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { MarketInstrument, PortfolioHolding, PortfolioOrderCreate } from "./types";

const formatMoney = (value: string, currency: string) =>
  new Intl.NumberFormat("en-IE", { style: "currency", currency }).format(Number(value));

function clientOrderId() {
  return globalThis.crypto?.randomUUID?.() ?? `order-${Date.now()}-${Math.random()}`;
}

export default function PortfolioTradingPanel({ portfolioId }: { portfolioId: string }) {
  const queryClient = useQueryClient();
  const [searchText, setSearchText] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [instrument, setInstrument] = useState<MarketInstrument | null>(null);
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [quantity, setQuantity] = useState("1");

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedSearch(searchText.trim()), 400);
    return () => window.clearTimeout(timeout);
  }, [searchText]);

  const overview = useQuery({
    queryKey: ["portfolio-overview", portfolioId],
    queryFn: () => api.portfolioOverview(portfolioId),
  });
  const instruments = useQuery({
    queryKey: ["market-instruments", debouncedSearch],
    queryFn: () => api.marketInstruments(debouncedSearch),
    enabled: debouncedSearch.length >= 2,
    retry: false,
    staleTime: 60_000,
  });
  const executeOrder = useMutation({
    mutationFn: (payload: PortfolioOrderCreate) =>
      api.executePortfolioOrder(portfolioId, payload),
    onSuccess: () => {
      setInstrument(null);
      setQuantity("1");
      setSearchText("");
      void Promise.all([
        queryClient.invalidateQueries({ queryKey: ["portfolios"] }),
        queryClient.invalidateQueries({ queryKey: ["portfolio-overview", portfolioId] }),
        queryClient.invalidateQueries({ queryKey: ["analytics", portfolioId] }),
        queryClient.invalidateQueries({ queryKey: ["transactions"] }),
        queryClient.invalidateQueries({ queryKey: ["accounts"] }),
      ]);
    },
  });

  const submitOrder = (event: FormEvent) => {
    event.preventDefault();
    if (!instrument) return;
    executeOrder.mutate({
      client_order_id: clientOrderId(),
      instrument_id: instrument.id,
      side,
      quantity,
    });
  };
  const prepareHoldingOrder = (holding: PortfolioHolding, orderSide: "buy" | "sell") => {
    setInstrument(holding.instrument);
    setSide(orderSide);
    setQuantity(orderSide === "buy" ? "1" : holding.quantity);
    setSearchText(holding.instrument.symbol);
  };

  if (overview.isLoading) return <div className="loading">Loading portfolio ledger…</div>;
  if (overview.isError) return <div className="error">{overview.error.message}</div>;
  if (!overview.data) return null;
  const portfolio = overview.data;

  return (
    <>
      <div className="demo-banner paper-banner">
        <b>SIMULATED EXECUTION</b>
        <span>Orders update this portfolio's brokerage cash and transaction ledger</span>
      </div>
      <section className="metrics-grid paper-metrics">
        <Metric label="Total equity" value={portfolio.total_equity} currency={portfolio.base_currency} note="Cash plus holdings" />
        <Metric label="Available cash" value={portfolio.cash_balance} currency={portfolio.base_currency} note="Opening cash plus ledger cash flows" />
        <Metric label="Holdings" value={portfolio.holdings_value} currency={portfolio.base_currency} note={`${portfolio.holdings.length} open positions`} />
        <Metric label="Realized P&L" value={portfolio.realized_pnl} currency={portfolio.base_currency} note="Average-cost method" />
      </section>

      <section className="panel order-panel">
        <div className="panel-title"><div><span className="eyebrow">BUY OR SELL</span><h3>Trade from this portfolio</h3></div><span className="page-meta">Cash: {formatMoney(portfolio.cash_balance, portfolio.base_currency)}</span></div>
        <div className="instrument-search">
          <label>Search by symbol or name<input value={searchText} onChange={(event) => { setSearchText(event.target.value); setInstrument(null); }} placeholder="Example: WORLD or Apple" /></label>
          <div className="instrument-results">
            {instruments.isFetching ? <span>Searching…</span> : null}
            {instruments.isError ? <span className="negative">Search unavailable</span> : null}
            {instruments.data?.map((item) => (
              <button type="button" className={instrument?.id === item.id ? "instrument active" : "instrument"} key={item.id} onClick={() => setInstrument(item)}>
                <b>{item.symbol}</b><span>{item.name}</span><small>{item.exchange || item.provider} · {item.currency}</small>
              </button>
            ))}
          </div>
        </div>
        {instrument ? (
          <form className="paper-order-form" noValidate onSubmit={submitOrder}>
            <div><b>{instrument.symbol}</b><span>{instrument.name}</span></div>
            <label>Side<select value={side} onChange={(event) => setSide(event.target.value as "buy" | "sell")}><option value="buy">Buy</option><option value="sell">Sell</option></select></label>
            <label>Quantity<input required min="0.00000001" step="1" type="number" value={quantity} onChange={(event) => setQuantity(event.target.value)} /></label>
            <button disabled={executeOrder.isPending}>{executeOrder.isPending ? "Executing…" : side === "buy" ? "Buy position" : "Sell position"}</button>
            {instrument.currency !== portfolio.base_currency ? <span className="order-note">The backend converts {instrument.currency} settlement into {portfolio.base_currency} using the stored FX reference rate.</span> : null}
          </form>
        ) : null}
        {executeOrder.isError ? <div className="error">{executeOrder.error.message}</div> : null}
      </section>

      <HoldingsTable holdings={portfolio.holdings} currency={portfolio.base_currency} onOrder={prepareHoldingOrder} />
      <section className="panel">
        <div className="panel-title"><div><span className="eyebrow">PORTFOLIO LEDGER</span><h3>Security transactions</h3></div><span className="page-meta">{portfolio.trade_count} trades</span></div>
        <div className="table-wrap"><table><thead><tr><th>Date</th><th>Instrument</th><th>Side</th><th>Quantity</th><th>Price</th><th>Cash flow</th></tr></thead><tbody>
          {[...portfolio.trades].reverse().map((trade) => <tr key={trade.id}><td>{trade.booked_at}</td><td>{trade.instrument.symbol}</td><td>{trade.side.toUpperCase()}</td><td>{Number(trade.quantity).toLocaleString()}</td><td>{formatMoney(trade.unit_price, trade.instrument_currency)}<small>Price observed {trade.price_observed_on}</small></td><td className={trade.side === "buy" ? "negative" : "positive"}>{trade.side === "buy" ? "−" : "+"}{formatMoney(trade.settlement_amount, trade.currency)}</td></tr>)}
          {!portfolio.trades.length ? <tr><td colSpan={6} className="empty-state">No trades yet.</td></tr> : null}
        </tbody></table></div>
      </section>
      <ul className="warnings">{portfolio.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
    </>
  );
}

function Metric({ label, value, currency, note }: { label: string; value: string; currency: string; note: string }) {
  return <div className="metric"><p>{label}</p><strong>{formatMoney(value, currency)}</strong><span>{note}</span></div>;
}

function HoldingsTable({ holdings, currency, onOrder }: { holdings: PortfolioHolding[]; currency: string; onOrder: (holding: PortfolioHolding, side: "buy" | "sell") => void }) {
  return <section className="panel"><div className="panel-title"><div><span className="eyebrow">HOLDINGS</span><h3>Current positions</h3></div></div><div className="table-wrap"><table><thead><tr><th>Instrument</th><th>Quantity</th><th>Average cost</th><th>Latest close</th><th>Value</th><th>Unrealized P&amp;L</th><th></th></tr></thead><tbody>
    {holdings.map((holding) => <tr key={holding.instrument.id}><td><b>{holding.instrument.symbol}</b><small>{holding.instrument.name}</small></td><td>{Number(holding.quantity).toLocaleString()}</td><td>{formatMoney(holding.average_cost, currency)}</td><td>{formatMoney(holding.latest_price, currency)}<small>{holding.price_observed_on} · {holding.price_source}</small></td><td>{formatMoney(holding.market_value, currency)}</td><td className={Number(holding.unrealized_pnl) >= 0 ? "positive" : "negative"}>{formatMoney(holding.unrealized_pnl, currency)}</td><td><div className="position-actions"><button className="secondary compact" type="button" onClick={() => onOrder(holding, "buy")}>Buy more</button><button className="secondary compact" type="button" onClick={() => onOrder(holding, "sell")}>Sell</button></div></td></tr>)}
    {!holdings.length ? <tr><td colSpan={7} className="empty-state">No open positions yet.</td></tr> : null}
  </tbody></table></div></section>;
}
