from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data/external")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("=", "_").replace("^", "")
    return CACHE_DIR / f"yahoo_{safe}_1d.csv"

def _to_num(s: pd.Series) -> pd.Series:
    # olası virgül/boşluk vs. temizle, sonra numeric'e zorla
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")

def get_daily_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp, force: bool = False) -> pd.DataFrame:
    path = _cache_path(symbol)
    start_d = pd.to_datetime(start).normalize()
    end_d = pd.to_datetime(end).normalize()

    if (not force) and path.exists():
        cached = pd.read_csv(path, parse_dates=["Date"])
        cached["Date"] = pd.to_datetime(cached["Date"]).dt.tz_localize(None)
        cached["Close"] = _to_num(cached["Close"])   # ✅ kritik
        cached = cached.dropna(subset=["Close"])
        return cached[(cached["Date"] >= start_d) & (cached["Date"] < end_d)].copy()

    df = yf.download(
        symbol,
        start=start_d.date(),
        end=(end_d + pd.Timedelta(days=1)).date(),
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        raise RuntimeError(f"Boş veri döndü: {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    out = df[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out["Close"] = _to_num(out["Close"])             # ✅ kritik
    out = out.dropna(subset=["Close"])

    out.to_csv(path, index=False)
    return out[(out["Date"] >= start_d) & (out["Date"] < end_d)].copy()
