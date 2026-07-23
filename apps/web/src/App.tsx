import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "./api";
import type { Allocation, Analytics } from "./types";

const COLORS = ["#7cf4c5", "#68a7ff", "#b991ff", "#ffce6d", "#ff7887", "#54d6e8"];
const euro = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
const pct = (value: number) => `${value.toFixed(1)}%`;

function MetricCard({ label, value, note, tone }: { label: string; value: string; note: string; tone?: "good" | "risk" }) {
  return <article className={`metric ${tone ?? ""}`}><p>{label}</p><strong>{value}</strong><span>{note}</span></article>;
}

function AllocationChart({ title, data }: { title: string; data: Allocation[] }) {
  return <section className="panel allocation"><div className="panel-title"><div><span className="eyebrow">EXPOSURE</span><h3>{title}</h3></div></div><div className="allocation-body"><ResponsiveContainer width="52%" height={210}><PieChart><Pie data={data} dataKey="weight" nameKey="label" innerRadius={55} outerRadius={84} paddingAngle={3}>{data.map((entry, index) => <Cell key={entry.label} fill={COLORS[index % COLORS.length]} />)}</Pie><Tooltip formatter={(value) => pct(Number(value) * 100)} /></PieChart></ResponsiveContainer><div className="legend">{data.map((entry, index) => <div key={entry.label}><i style={{ background: COLORS[index % COLORS.length] }} /><span>{entry.label}</span><b>{pct(entry.weight * 100)}</b></div>)}</div></div></section>;
}

function Dashboard({ analytics }: { analytics: Analytics }) {
  const positive = Number(analytics.unrealized_pnl_eur) >= 0;
  return <>
    <div className="metrics-grid">
      <MetricCard label="Portfolio value" value={euro.format(Number(analytics.market_value_eur))} note={`As of ${analytics.as_of}`} />
      <MetricCard label="Unrealized P&L" value={`${positive ? "+" : ""}${euro.format(Number(analytics.unrealized_pnl_eur))}`} note={pct(analytics.unrealized_pnl_percent)} tone={positive ? "good" : "risk"} />
      <MetricCard label="Annualized volatility" value={pct(analytics.annualized_volatility_percent)} note="252-day estimate" tone="risk" />
      <MetricCard label="Maximum drawdown" value={pct(analytics.max_drawdown_percent)} note="Trailing period" tone="risk" />
      <MetricCard label="Concentration HHI" value={analytics.concentration_hhi.toFixed(3)} note={`${analytics.largest_position_symbol} · ${pct(analytics.largest_position_weight * 100)}`} />
    </div>
    <section className="panel performance"><div className="panel-title"><div><span className="eyebrow">PORTFOLIO PATH</span><h3>Current holdings, reconstructed</h3></div><span className="return-pill">{analytics.trailing_return_percent >= 0 ? "+" : ""}{pct(analytics.trailing_return_percent)}</span></div><ResponsiveContainer width="100%" height={300}><AreaChart data={analytics.value_series}><defs><linearGradient id="valueFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#7cf4c5" stopOpacity={0.35}/><stop offset="100%" stopColor="#7cf4c5" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#203047" vertical={false}/><XAxis dataKey="date" stroke="#7890aa" tickLine={false}/><YAxis stroke="#7890aa" tickLine={false} tickFormatter={(v) => `€${Math.round(v / 1000)}k`}/><Tooltip formatter={(v) => euro.format(Number(v))}/><Area type="monotone" dataKey="value_eur" stroke="#7cf4c5" fill="url(#valueFill)" strokeWidth={2}/></AreaChart></ResponsiveContainer></section>
    <div className="chart-grid"><AllocationChart title="Asset class allocation" data={analytics.allocations.asset_class}/><AllocationChart title="Regional allocation" data={analytics.allocations.region}/></div>
    <section className="panel"><div className="panel-title"><div><span className="eyebrow">HOLDINGS</span><h3>Position contribution</h3></div></div><div className="table-wrap"><table><thead><tr><th>Symbol</th><th>Market value</th><th>Cost basis</th><th>P&L</th><th>Weight</th></tr></thead><tbody>{analytics.positions.map((position) => <tr key={position.symbol}><td><b>{position.symbol}</b></td><td>{euro.format(Number(position.market_value_eur))}</td><td>{euro.format(Number(position.cost_basis_eur))}</td><td className={Number(position.pnl_eur) >= 0 ? "positive" : "negative"}>{euro.format(Number(position.pnl_eur))}</td><td>{pct(position.weight * 100)}</td></tr>)}</tbody></table></div></section>
    <div className="warnings">{analytics.warnings.map((warning) => <p key={warning}>ⓘ {warning}</p>)}</div>
  </>;
}

export default function App() {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState("");
  const [showImport, setShowImport] = useState(false);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const portfolios = useQuery({ queryKey: ["portfolios"], queryFn: api.portfolios });
  useEffect(() => { if (!selected && portfolios.data?.length) setSelected(portfolios.data[0].id); }, [portfolios.data, selected]);
  const analytics = useQuery({ queryKey: ["analytics", selected], queryFn: () => api.analytics(selected), enabled: Boolean(selected) });
  const importer = useMutation({ mutationFn: () => api.importPortfolio(name, file!), onSuccess: async (portfolio) => { await queryClient.invalidateQueries({ queryKey: ["portfolios"] }); setSelected(portfolio.id); setShowImport(false); setName(""); setFile(null); } });

  return <div className="app-shell"><aside><div className="brand"><div className="brand-mark">N</div><div><strong>NORTHSTAR</strong><span>FINANCIAL INTELLIGENCE</span></div></div><nav><a className="active">⌁ Portfolio overview</a><a className="disabled">◈ AI assistant <small>Phase 2</small></a><a className="disabled">◇ Transactions <small>Phase 3</small></a></nav><div className="sidebar-note"><span>LOCAL MODE</span><p>No live market data or cloud services are active.</p></div></aside>
    <main><header><div><span className="eyebrow">PORTFOLIO INTELLIGENCE</span><h1>Risk, concentration and performance</h1><p>Deterministic analytics for transparent financial exploration.</p></div><div className="header-actions"><select aria-label="Select portfolio" value={selected} onChange={(e) => setSelected(e.target.value)}>{portfolios.data?.map((p) => <option key={p.id} value={p.id}>{p.name} {p.kind === "demo" ? "· Demo" : ""}</option>)}</select><button onClick={() => setShowImport(true)}>Import CSV</button></div></header>
    <div className="demo-banner"><b>SYNTHETIC DEMO MARKET DATA</b><span>Educational use only · Not financial advice</span></div>
    {portfolios.isError && <div className="error">Could not reach the API. Start the FastAPI service on port 8000.</div>}
    {analytics.isLoading && <div className="loading">Calculating portfolio analytics…</div>}
    {analytics.isError && <div className="error">{(analytics.error as Error).message}</div>}
    {analytics.data && <Dashboard analytics={analytics.data}/>}</main>
    {showImport && <div className="modal-backdrop" onMouseDown={() => setShowImport(false)}><form className="modal" onSubmit={(e) => { e.preventDefault(); importer.mutate(); }} onMouseDown={(e) => e.stopPropagation()}><span className="eyebrow">NEW PORTFOLIO</span><h2>Import positions</h2><p>Use the exact market catalog metadata. The complete file is rejected when any row is invalid.</p><label>Portfolio name<input required maxLength={120} value={name} onChange={(e) => setName(e.target.value)} placeholder="My demo portfolio"/></label><label>UTF-8 CSV<input required type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)}/></label>{importer.isError && <pre className="error">{(importer.error as Error).message}</pre>}<div className="modal-actions"><a href="/portfolio_template.csv" download>Download template</a><button type="button" className="secondary" onClick={() => setShowImport(false)}>Cancel</button><button disabled={!file || !name || importer.isPending}>{importer.isPending ? "Importing…" : "Import portfolio"}</button></div></form></div>}
  </div>;
}

