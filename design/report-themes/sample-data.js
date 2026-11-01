// Deterministic sample data shared by every theme mockup (seeded PRNG).
(function () {
  let s = 42;
  const rnd = () => ((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
  const dates = [], equity = [], benchmark = [], drawdown = [];
  let e = 100000, b = 100000, peak = e;
  const d0 = new Date(Date.UTC(2022, 0, 3));
  for (let i = 0, d = new Date(d0); equity.length < 756; d.setUTCDate(d.getUTCDate() + 1)) {
    const wd = d.getUTCDay(); if (wd === 0 || wd === 6) continue;
    const m = gauss();
    b *= 1 + 0.0003 + 0.011 * m;
    e *= 1 + 0.0007 + 0.008 * m + 0.006 * gauss();
    peak = Math.max(peak, e);
    dates.push(d.toISOString().slice(0, 10));
    equity.push(+e.toFixed(2)); benchmark.push(+b.toFixed(2));
    drawdown.push(+((e / peak - 1) * 100).toFixed(2)); i++;
  }
  const monthly = {}; // {year: [12 monthly % returns or null]}
  let prev = equity[0], prevKey = dates[0].slice(0, 7);
  dates.forEach((dt, i) => {
    const key = dt.slice(0, 7);
    const last = i === dates.length - 1;
    if (key !== prevKey || last) {
      const endVal = last ? equity[i] : equity[i - 1];
      const [y, mo] = prevKey.split("-").map(Number);
      (monthly[y] ||= Array(12).fill(null))[mo - 1] = +((endVal / prev - 1) * 100).toFixed(2);
      prev = endVal; prevKey = key;
    }
  });
  const tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "AAPL", "MSFT", "NVDA", "AMZN", "ICICIBANK", "META", "TSLA", "LT"];
  const trades = [];
  for (let i = 0; i < 64; i++) {
    const r = +(gauss() * 7 + 1.4).toFixed(2);
    const entry = dates[Math.floor(rnd() * 700)];
    trades.push({ ticker: tickers[Math.floor(rnd() * tickers.length)], entry, hold: 3 + Math.floor(rnd() * 40),
      ret: r, pnl: Math.round(r * 120 + gauss() * 40), exit: rnd() < 0.6 ? "time" : rnd() < 0.5 ? "stop" : "target" });
  }
  const sectors = ["Tech", "Financials", "Energy", "Healthcare", "Industrials", "Consumer"];
  const sectorReturns = sectors.map((name, k) => ({ name, ret: +(gauss() * 9 + 3 - k).toFixed(1) }));
  const final = equity.at(-1), years = equity.length / 252;
  window.REPORT = {
    title: "Momentum Breakout - NIFTY 50 + S&P 100",
    subtitle: "2022-01-03 to " + dates.at(-1) + " · daily bars · 5 slots · 10 bps costs",
    dates, equity, benchmark, drawdown, monthly, trades, sectorReturns,
    metrics: [
      { label: "Total return", value: ((final / 100000 - 1) * 100).toFixed(1) + "%", delta: "+" + ((final / 100000 - benchmark.at(-1) / 100000) * 100).toFixed(1) + " pp vs bench", tone: "pos" },
      { label: "CAGR", value: ((Math.pow(final / 100000, 1 / years) - 1) * 100).toFixed(1) + "%", tone: "pos" },
      { label: "Sharpe", value: "1.42", tone: "neutral" },
      { label: "Max drawdown", value: Math.min(...drawdown).toFixed(1) + "%", tone: "neg" },
      { label: "Win rate", value: Math.round(trades.filter(t => t.ret > 0).length / trades.length * 100) + "%", tone: "neutral" },
      { label: "Trades", value: String(trades.length), tone: "neutral" },
    ],
    warnings: [
      { level: "warning", text: "3 tickers had fewer than 200 bars of history and were skipped." },
      { level: "good", text: "No look-ahead detected in signal alignment." },
    ],
  };
})();
