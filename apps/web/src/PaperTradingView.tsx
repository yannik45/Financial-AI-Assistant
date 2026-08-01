import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { MarketInstrument, PaperHolding, PaperOrderCreate } from "./types";

const formatMoney = (value: string, currency: string) =>
  new Intl.NumberFormat("en-IE", { style: "currency", currency }).format(Number(value));

function clientOrderId() {
  return globalThis.crypto?.randomUUID?.() ?? `paper-${Date.now()}-${Math.random()}`;
}

export default function PaperTradingView() {
  const queryClient = useQueryClient();
  const [selectedPortfolioId, setSelectedPortfolioId] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [portfolioName, setPortfolioName] = useState("My Paper Portfolio");
  const [startingCash, setStartingCash] = useState("10000");
  const [baseCurrency, setBaseCurrency] = useState("EUR");
  const [searchText, setSearchText] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [instrument, setInstrument] = useState<MarketInstrument | null>(null);
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [quantity, setQuantity] = useState("1");

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedSearch(searchText.trim()), 400);
    return () => window.clearTimeout(timeout);
  }, [searchText]);

  const portfolios = useQuery({ queryKey: ["paper-portfolios"], queryFn: api.paperPortfolios });
  useEffect(() => {
    if (!selectedPortfolioId && portfolios.data?.length) {
      setSelectedPortfolioId(portfolios.data[0].id);
    }
  }, [portfolios.data, selectedPortfolioId]);
  const portfolio = useQuery({
    queryKey: ["paper-portfolio", selectedPortfolioId],
    queryFn: () => api.paperPortfolio(selectedPortfolioId),
    enabled: Boolean(selectedPortfolioId),
  });
  const instruments = useQuery({
    queryKey: ["market-instruments", debouncedSearch],
    queryFn: () => api.marketInstruments(debouncedSearch),
    enabled: debouncedSearch.length >= 2,
    retry: false,
    staleTime: 60_000,
  });
  const createPortfolio = useMutation({
    mutationFn: () =>
      api.createPaperPortfolio({
        name: portfolioName,
        starting_cash: startingCash,
        base_currency: baseCurrency,
      }),
    onSuccess: async (created) => {
      await queryClient.invalidateQueries({ queryKey: ["paper-portfolios"] });
      setSelectedPortfolioId(created.id);
      setShowCreate(false);
    },
  });
  const executeOrder = useMutation({
    mutationFn: (payload: PaperOrderCreate) =>
      api.executePaperOrder(selectedPortfolioId, payload),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["paper-portfolios"] }),
        queryClient.invalidateQueries({ queryKey: ["paper-portfolio", selectedPortfolioId] }),
      ]);
      setInstrument(null);
      setQuantity("1");
      setSearchText("");
    },
  });

  const submitOrder = (event: FormEvent) => {
    event.preventDefault();
    if (!instrument || !selectedPortfolioId) return;
    executeOrder.mutate({
      client_order_id: clientOrderId(),
      instrument_id: instrument.id,
      side,
      quantity,
    });
  };
  const prepareSale = (holding: PaperHolding) => {
    setInstrument(holding.instrument);
    setSide("sell");
    setQuantity(holding.quantity);
    setSearchText(holding.instrument.symbol);
  };

  return (
    <main>
      <header>
        <div>
          <span className="eyebrow">PAPER PORTFOLIO</span>
          <h1>Explore markets without placing real orders</h1>
          <p>Server-priced simulated trades with cash, holdings, and P&amp;L derived from the ledger.</p>
        </div>
        <div className="header-actions">
          {portfolios.data?.length ? (
            <select aria-label="Select paper portfolio" value={selectedPortfolioId} onChange={(event) => setSelectedPortfolioId(event.target.value)}>
              {portfolios.data.map((item) => <option value={item.id} key={item.id}>{item.name} · {item.base_currency}</option>)}
            </select>
          ) : null}
          <button onClick={() => setShowCreate(true)}>New paper portfolio</button>
        </div>
      </header>
      <div className="demo-banner paper-banner">
        <b>PAPER TRADING ONLY</b>
        <span>No real order is placed · Latest cached daily close · No investment advice</span>
      </div>

      {portfolios.isLoading && <div className="loading">Loading paper portfolios…</div>}
      {portfolios.isError && <div className="error">{portfolios.error.message}</div>}
      {!portfolios.isLoading && !portfolios.data?.length ? (
        <section className="panel empty-paper-state">
          <span className="eyebrow">START HERE</span>
          <h2>Create a paper portfolio</h2>
          <p>Choose a starting cash balance, then discover instruments and simulate trades.</p>
          <button onClick={() => setShowCreate(true)}>Create portfolio</button>
        </section>
      ) : null}

      {portfolio.isLoading && <div className="loading">Replaying paper trades…</div>}
      {portfolio.isError && <div className="error">{portfolio.error.message}</div>}
      {portfolio.data ? (
        <>
          <section className="metrics-grid paper-metrics">
            <Metric label="Total equity" value={portfolio.data.total_equity} currency={portfolio.data.base_currency} note="Cash plus holdings" tone="good" />
            <Metric label="Available cash" value={portfolio.data.cash_balance} currency={portfolio.data.base_currency} note="Starting balance less trades" />
            <Metric label="Holdings" value={portfolio.data.holdings_value} currency={portfolio.data.base_currency} note={`${portfolio.data.holdings.length} open positions`} />
            <Metric label="Total P&L" value={portfolio.data.total_pnl} currency={portfolio.data.base_currency} note="Realized and unrealized" tone={Number(portfolio.data.total_pnl) >= 0 ? "good" : "risk"} />
            <Metric label="Realized P&L" value={portfolio.data.realized_pnl} currency={portfolio.data.base_currency} note="Average-cost method" />
          </section>

          <section className="panel order-panel">
            <div className="panel-title"><div><span className="eyebrow">SIMULATED ORDER</span><h3>Find an instrument</h3></div><span className="page-meta">Portfolio currency: {portfolio.data.base_currency}</span></div>
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
              <form className="paper-order-form" onSubmit={submitOrder}>
                <div><b>{instrument.symbol}</b><span>{instrument.name}</span></div>
                <label>Side<select value={side} onChange={(event) => setSide(event.target.value as "buy" | "sell")}><option value="buy">Buy</option><option value="sell">Sell</option></select></label>
                <label>Quantity<input required min="0.00000001" step="0.00000001" type="number" value={quantity} onChange={(event) => setQuantity(event.target.value)} /></label>
                <button disabled={executeOrder.isPending || instrument.currency !== portfolio.data.base_currency}>{executeOrder.isPending ? "Executing…" : `Simulate ${side}`}</button>
                {instrument.currency !== portfolio.data.base_currency ? <span className="order-note negative">Select an instrument in {portfolio.data.base_currency} or use a matching portfolio.</span> : <span className="order-note">The backend selects the latest cached daily close.</span>}
              </form>
            ) : null}
            {executeOrder.isError ? <div className="error">{executeOrder.error.message}</div> : null}
          </section>

          <HoldingsTable holdings={portfolio.data.holdings} currency={portfolio.data.base_currency} onSell={prepareSale} />
          <section className="panel">
            <div className="panel-title"><div><span className="eyebrow">LEDGER</span><h3>Immutable paper trades</h3></div><span className="page-meta">{portfolio.data.trade_count} trades</span></div>
            <div className="table-wrap"><table><thead><tr><th>Date</th><th>Instrument</th><th>Side</th><th>Quantity</th><th>Execution price</th><th>Source</th></tr></thead><tbody>
              {[...portfolio.data.trades].reverse().map((trade) => <tr key={trade.id}><td>{trade.price_observed_on}</td><td>{trade.instrument.symbol}</td><td>{trade.side.toUpperCase()}</td><td>{Number(trade.quantity).toLocaleString()}</td><td>{formatMoney(trade.unit_price, trade.currency)}</td><td>{trade.price_source}</td></tr>)}
              {!portfolio.data.trades.length ? <tr><td colSpan={6} className="empty-state">No paper trades yet.</td></tr> : null}
            </tbody></table></div>
          </section>
          <ul className="warnings">{portfolio.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </>
      ) : null}

      {showCreate ? (
        <div className="modal-backdrop" onMouseDown={() => setShowCreate(false)}>
          <form className="modal" onMouseDown={(event) => event.stopPropagation()} onSubmit={(event) => { event.preventDefault(); createPortfolio.mutate(); }}>
            <span className="eyebrow">NEW PAPER PORTFOLIO</span><h2>Choose a simulated cash balance</h2><p>This creates a local paper ledger. It does not open a brokerage account.</p>
            <label>Name<input required value={portfolioName} onChange={(event) => setPortfolioName(event.target.value)} /></label>
            <label>Base currency<select value={baseCurrency} onChange={(event) => setBaseCurrency(event.target.value)}><option>EUR</option><option>USD</option><option>GBP</option><option>JPY</option></select></label>
            <label>Starting cash<input required min="0.01" step="0.01" type="number" value={startingCash} onChange={(event) => setStartingCash(event.target.value)} /></label>
            {createPortfolio.isError ? <div className="error">{createPortfolio.error.message}</div> : null}
            <div className="modal-actions"><button type="button" className="secondary" onClick={() => setShowCreate(false)}>Cancel</button><button disabled={createPortfolio.isPending}>{createPortfolio.isPending ? "Creating…" : "Create portfolio"}</button></div>
          </form>
        </div>
      ) : null}
    </main>
  );
}

function Metric({ label, value, currency, note, tone = "" }: { label: string; value: string; currency: string; note: string; tone?: string }) {
  return <div className={`metric ${tone}`}><p>{label}</p><strong>{formatMoney(value, currency)}</strong><span>{note}</span></div>;
}

function HoldingsTable({ holdings, currency, onSell }: { holdings: PaperHolding[]; currency: string; onSell: (holding: PaperHolding) => void }) {
  return <section className="panel"><div className="panel-title"><div><span className="eyebrow">HOLDINGS</span><h3>Open paper positions</h3></div></div><div className="table-wrap"><table><thead><tr><th>Instrument</th><th>Quantity</th><th>Average cost</th><th>Latest close</th><th>Value</th><th>Unrealized P&amp;L</th><th></th></tr></thead><tbody>
    {holdings.map((holding) => <tr key={holding.instrument.id}><td><b>{holding.instrument.symbol}</b><small>{holding.instrument.name}</small></td><td>{Number(holding.quantity).toLocaleString()}</td><td>{formatMoney(holding.average_cost, currency)}</td><td>{formatMoney(holding.latest_price, currency)}<small>{holding.price_observed_on} · {holding.price_source}{holding.quote_is_stale ? " · stale" : ""}</small></td><td>{formatMoney(holding.market_value, currency)}</td><td className={Number(holding.unrealized_pnl) >= 0 ? "positive" : "negative"}>{formatMoney(holding.unrealized_pnl, currency)}</td><td><button className="secondary compact" onClick={() => onSell(holding)}>Sell</button></td></tr>)}
    {!holdings.length ? <tr><td colSpan={7} className="empty-state">No open paper positions yet.</td></tr> : null}
  </tbody></table></div></section>;
}
