# fastapi_backend.py — allocator with CSV/Parquet + rules, summary + chart data
import os
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Portfolio Allocator API")

# --- CORS (add your Netlify URL when deployed) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; later: ["https://YOUR-SITE.netlify.app"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Buckets (54 ETFs) ---
BUCKET_ETFS: Dict[str, List[str]] = {
    "Tech": ["QQQ","VGT","XLK"],
    "AI & Cloud & Semi": ["SMH","SOXX","SKYY","AIQ"],
    "Communication": ["XLC"],
    "Consumer Discretion": ["XLY"],
    "Consumer Staples": ["XLP"],
    "Healthcare": ["XLV","IBB"],
    "Financials": ["XLF"],
    "Industrials": ["XLI"],
    "Materials": ["XLB"],
    "Energy": ["XLE","VDE"],
    "Utilities": ["XLU"],
    "Real Estate": ["VNQ","IYR","SCHH"],
    "US Broad Market": ["VTI","ITOT","SCHB"],
    "US Mid/Small": ["VO","MDY","VB","IWM"],
    "US Value": ["VTV","IWD"],
    "US Growth": ["VUG","IWF"],
    "Dividends": ["SCHD","VYM","DGRO"],
    "Covered Call Income": ["JEPI","JEPQ","QYLD"],
    "Aggregate Bonds": ["BND","AGG","SCHZ"],
    "Treasuries Ladder": ["SHY","IEF","TLT"],
    "TIPS": ["TIP","SCHP"],
    "Corporate Bonds": ["LQD"],
    "High Yield Bonds": ["HYG","JNK"],
    "Intl Developed": ["VEA","IEFA"],
    "Emerging Markets": ["VWO","EEM"],
    "Total ex-US": ["VXUS"],
}

# ---------- Data loading ----------
CANDIDATES = [
    os.getenv("PRICES_CSV_PATH", "").strip() or None,
    "data/prices.csv", "prices.csv",
    "data/etf_prices.csv",
    "data/prices.parquet", "prices.parquet",
]

def _resolve_path() -> Optional[str]:
    for p in CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None

def _load_prices(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)

    # LONG: Date,Ticker,Adj Close/Close   |   WIDE: Date + columns=tickers
    if "Ticker" in df.columns:
        price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if price_col is None or "Date" not in df.columns: return None
        df["Date"] = pd.to_datetime(df["Date"])
        px = df.pivot(index="Date", columns="Ticker", values=price_col).sort_index()
    else:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        px = df

    px = px.replace([np.inf, -np.inf], np.nan).dropna(how="all").astype(float)
    return px

PRICES_PATH = _resolve_path()
PRICES: Optional[pd.DataFrame] = _load_prices(PRICES_PATH)

def _compute_signals(px: pd.DataFrame, mom_win=126, vol_win=20) -> pd.DataFrame:
    mom = px.pct_change(mom_win).iloc[-1]
    vol = px.pct_change().rolling(vol_win).std().iloc[-1]
    score = mom - 0.5 * vol
    out = pd.DataFrame({"momentum": mom, "vol": vol, "score": score})
    return out.dropna(how="all")

SIGNALS = _compute_signals(PRICES) if (PRICES is not None and len(PRICES) > 60) else None
MODE_READY = SIGNALS is not None and not SIGNALS.empty

# ---------- Models ----------
class AllocationRequest(BaseModel):
    bucketTargets: Dict[str, float] = Field(..., description="Percents (0–100) or decimals (0–1)")
    amount: Optional[float] = Field(None, description="Total dollars")
    mode: Optional[str] = Field(None, description="'rules' or 'simple'")
    topK: Optional[int] = Field(2, description="Top K tickers per bucket (rules mode)")

# ---------- Helpers ----------
def _normalize_targets(raw: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    clean = {k: max(0.0, float(v)) for k, v in raw.items()}
    s = sum(clean.values())
    targets = clean if s <= 1.000001 else {k: v/100.0 for k, v in clean.items()}
    s = sum(targets.values())
    if s > 1.0:
        targets = {k: v/s for k, v in targets.items()}
        s = 1.0
    cash = round(1.0 - s, 10)
    return targets, cash

def _allocate_simple(targets: Dict[str, float]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for bucket, t in targets.items():
        tickers = BUCKET_ETFS.get(bucket, [])
        if not tickers or t <= 0: continue
        w = t / len(tickers)
        for tk in tickers:
            weights[tk] = weights.get(tk, 0.0) + w
    return weights

def _allocate_rules(targets: Dict[str, float], topK: int) -> Dict[str, float]:
    if not MODE_READY or PRICES is None: return _allocate_simple(targets)
    sig = SIGNALS
    px_tickers = set(PRICES.columns)
    weights: Dict[str, float] = {}
    for bucket, t in targets.items():
        if t <= 0: continue
        cands = [tk for tk in BUCKET_ETFS.get(bucket, []) if tk in px_tickers]
        if not cands:
            # no data -> equal split of defined tickers
            return _allocate_simple({bucket: t})
        s = sig.loc[[tk for tk in cands if tk in sig.index]].copy()
        if s.empty:
            w = t / len(cands)
            for tk in cands: weights[tk] = weights.get(tk, 0.0) + w
            continue
        s = s.sort_values("score", ascending=False).head(max(1, topK))
        pos = s["score"].clip(lower=0)
        if pos.sum() <= 1e-12:
            w = t / len(s)
            for tk in s.index: weights[tk] = weights.get(tk, 0.0) + w
        else:
            total = float(pos.sum())
            for tk, sc in pos.items():
                weights[tk] = weights.get(tk, 0.0) + t * (float(sc) / total)
    return weights

def _summarize(weights: Dict[str, float], dollars: Optional[Dict[str, float]], amount: Optional[float]) -> List[str]:
    # bucket sums
    lines = []
    for bucket, tickers in BUCKET_ETFS.items():
        bw = sum(weights.get(tk, 0.0) for tk in tickers)
        if bw <= 1e-9: continue
        tick_parts = []
        for tk in tickers:
            w = weights.get(tk, 0.0)
            if w <= 1e-9: continue
            part = f"{tk} {(w*100):.1f}%"
            if dollars: part += f" (${dollars.get(tk,0):,.0f})"
            tick_parts.append(part)
        amt = f" (${bw*amount:,.0f})" if dollars and amount else ""
        lines.append(f"{bucket}: {(bw*100):.1f}%{amt} → " + ", ".join(tick_parts))
    if "CASH" in weights:
        w = weights["CASH"]
        cash_line = f"CASH {(w*100):.1f}%"
        if dollars and amount: cash_line += f" (${w*amount:,.0f})"
        lines.append(cash_line)
    return lines

# ---------- Routes ----------
@app.get("/ping")
def ping():
    return {"pong": True}

@app.get("/health")
def health():
    return {
        "csv_found": PRICES_PATH is not None,
        "path": PRICES_PATH or "(not found)",
        "mode_ready": MODE_READY,
        "rows": int(PRICES.shape[0]) if PRICES is not None else 0,
        "cols": int(PRICES.shape[1]) if PRICES is not None else 0,
        "last_date": None if PRICES is None else str(PRICES.index.max().date()),
    }

@app.get("/buckets")
def buckets():
    return {"buckets": [{"name": b, "tickers": t} for b, t in BUCKET_ETFS.items()]}

@app.post("/allocate")
def allocate(req: AllocationRequest):
    targets, cash_w = _normalize_targets(req.bucketTargets)
    use_rules = (req.mode or "").lower() == "rules" or (req.mode in (None, "") and MODE_READY)
    weights = _allocate_rules(targets, topK=req.topK or 2) if use_rules else _allocate_simple(targets)
    if cash_w > 1e-9:
        weights["CASH"] = weights.get("CASH", 0.0) + cash_w

    dollars = None
    if req.amount and req.amount > 0:
        dollars = {tk: round(float(w)*req.amount, 2) for tk, w in weights.items()}

    # chart arrays
    labels = []
    values_pct = []
    for tk, w in weights.items():
        if w <= 0: continue
        labels.append(tk)
        values_pct.append(round(w*100, 2))

    summary = _summarize(weights, dollars, req.amount)

    return {
        "portfolio_weights": {k: round(v, 6) for k, v in weights.items()},
        "portfolio_dollars": dollars,
        "chart": {"labels": labels, "values_pct": values_pct},
        "summary_lines": summary,
        "diagnostics": {
            "mode_used": "rules" if use_rules else "simple",
            "sum_weights": round(sum(weights.values()), 6),
            "auto_cash_weight": round(cash_w, 6),
        },
    }
