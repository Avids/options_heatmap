import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime
from math import log, sqrt, exp, pi

st.set_page_config(page_title="Options Net Heatmap (with GEX)", layout="wide")

# =============================
# CONSTANTS & HELPERS
# =============================
CONTRACT_SIZE = 100  # standard equity options

def format_oi_value(val):
    try:
        v = int(round(val))
    except Exception:
        return ""
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,}"

def fetch_ticker(ticker_symbol: str):
    return yf.Ticker(ticker_symbol)

@st.cache_data(show_spinner=False)
def get_price(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    hist = t.history(period="1d")
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])

def get_option_chain_for_expiry(ticker, expiry):
    try:
        chain = ticker.option_chain(expiry)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return None, None

def bs_gamma(S, K, T, sigma):
    """Black-Scholes gamma per share (no dividends), using pdf of normal."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    pdf_d1 = exp(-0.5 * d1 * d1) / sqrt(2 * pi)
    gamma = pdf_d1 / (S * sigma * sqrtT)
    return float(gamma)
# ===== Firday ======
def is_friday(date):
    return date.weekday() == 4  # Monday=0
# ==================

def is_third_friday(date):
    if date.weekday() != 4:
        return False
    return 15 <= date.day <= 21

# =============================
# SIDEBAR / INPUTS
# =============================
st.title("Options Net Heatmap (Open Interest / Volume / GEX)")
st.markdown("Static heatmap (matplotlib) showing Calls − Puts for Open Interest, Volume, or Gamma Exposure (GEX).")

col_inputs, col_info = st.columns([1, 2])

with col_inputs:
    symbol = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    METRIC = st.selectbox(
        "Metric",
        options=["Open Interest", "Volume", "GEX (shares)", "Dollar GEX"]
    )
    # ======= Option type selector ====

    OPTION_TYPE = st.selectbox(
    "Option type",
    options=["Weekly", "Monthly", "Weekly + Monthly"]
    )
    STRIKE_RANGE = st.number_input(
        "Strikes above / below current price (count)",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )
    EXPIRY_COUNT = st.slider(
        "Number of expiries to include (max 12)", min_value=1, max_value=12, value=4
    )
    refresh = st.button("Update (re-fetch)")

with col_info:
    st.markdown("- Metric: choose Open Interest (openInterest), Volume (volume), GEX (shares) or Dollar GEX.")
    st.markdown("- GEX (shares): change in total delta (in shares) for a $1 move; Dollar GEX = GEX_shares × spot.")
    st.markdown("- If implied vol is missing for strikes, those strikes are omitted from GEX (or treated as 0).")
    st.markdown(f"- Report generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC")

if not symbol:
    st.info("Enter a ticker symbol to continue.")
    st.stop()

# About / Help panel (collapsible)
with st.expander("About / Help", expanded=False):
    st.markdown("""
    ## What this app shows
    - A static heatmap (strikes × expiries) of Calls − Puts for the selected metric.
    - Metric options: Open Interest, Volume, GEX (shares), Dollar GEX.

    ## How to read it
    - X axis: expiries (one column per expiry).
    - Y axis: strikes (highest shown at the top).
    - Colors: green = net call dominance; red = net put dominance; pale = near zero.
    - King nodes (largest positive/negative per expiry) are outlined and bolded.

    ## GEX calculation
    - Gamma per share (Black–Scholes) is computed using strike IV, spot price and time to expiry.
    - GEX_shares = Gamma_per_share × (Calls OI − Puts OI) × 100.
    - Dollar GEX = GEX_shares × spot.

    ## Use cases for traders
    - Identify strike concentrations that may act as short-term support/resistance.
    - Gauge potential hedging flows and gamma-related sensitivity near the spot.
    - Monitor how position concentrations roll between expiries.

    ## Caveats
    - Data comes from Yahoo via yfinance and may be incomplete.
    - GEX accuracy depends on available implied vol; missing IVs are warned and handled.
    """)



# =============================
# FETCH DATA
# =============================
with st.spinner(f"Fetching data for {symbol} ..."):
    price = get_price(symbol)
    ticker = fetch_ticker(symbol)
    try:
        all_expiries = list(ticker.options)
    except Exception:
        all_expiries = []

if price is None:
    st.error(f"Could not fetch price or options for '{symbol}'. Check the ticker and try again.")
    st.stop()

if not all_expiries:
    st.error(f"No option expiries returned for '{symbol}'.")
    st.stop()

# expiries = all_expiries[:EXPIRY_COUNT]
# =============== Expiry type

# convert expiry strings to datetime
expiry_dates = [
    pd.to_datetime(e, errors="coerce") for e in all_expiries
]
expiry_map = dict(zip(all_expiries, expiry_dates))

filtered_expiries = []

for e, d in expiry_map.items():
    if d is None or pd.isna(d):
        continue

    if OPTION_TYPE == "Monthly" and is_third_friday(d):
        filtered_expiries.append(e)

    elif OPTION_TYPE == "Weekly" and is_friday(d) and not is_third_friday(d):
        filtered_expiries.append(e)

    elif OPTION_TYPE == "Weekly + Monthly" and is_friday(d):
        filtered_expiries.append(e)

# hard cap at 12 expiries
filtered_expiries = filtered_expiries[:12]

# respect slider but never exceed 12
expiries = filtered_expiries[:min(EXPIRY_COUNT, 12)]

# ============


all_data = []
missing_iv_count = 0
total_iv_cells = 0

for expiry in expiries:
    calls, puts = get_option_chain_for_expiry(ticker, expiry)
    if calls is None or puts is None:
        continue

    # ensure expected columns exist; add defaults if missing
    for df in (calls, puts):
        if "strike" not in df.columns:
            df["strike"] = np.nan
        if "openInterest" not in df.columns:
            df["openInterest"] = 0
        if "volume" not in df.columns:
            df["volume"] = 0
        if "impliedVolatility" not in df.columns:
            df["impliedVolatility"] = np.nan

    calls_g = calls.groupby("strike").agg(
        call_oi=("openInterest", "sum"),
        call_vol=("volume", "sum"),
        call_iv_mean=("impliedVolatility", "mean"),
    )
    puts_g = puts.groupby("strike").agg(
        put_oi=("openInterest", "sum"),
        put_vol=("volume", "sum"),
        put_iv_mean=("impliedVolatility", "mean"),
    )

    strikes_union = sorted(set(calls_g.index).union(puts_g.index))

    rows = []
    for K in strikes_union:
        call_row = calls_g.loc[K] if K in calls_g.index else pd.Series({"call_oi": 0, "call_vol": 0, "call_iv_mean": np.nan})
        put_row = puts_g.loc[K] if K in puts_g.index else pd.Series({"put_oi": 0, "put_vol": 0, "put_iv_mean": np.nan})

        call_oi = float(call_row.get("call_oi", 0) or 0)
        put_oi = float(put_row.get("put_oi", 0) or 0)
        net_oi = call_oi - put_oi

        call_vol = float(call_row.get("call_vol", 0) or 0)
        put_vol = float(put_row.get("put_vol", 0) or 0)
        net_vol = call_vol - put_vol

        iv_num = 0.0
        iv_den = 0.0
        if not np.isnan(call_row.get("call_iv_mean", np.nan)):
            iv_num += (call_row["call_iv_mean"] * call_oi)
            iv_den += call_oi
        if not np.isnan(put_row.get("put_iv_mean", np.nan)):
            iv_num += (put_row["put_iv_mean"] * put_oi)
            iv_den += put_oi
        if iv_den > 0:
            strike_iv = iv_num / iv_den
        else:
            cand = []
            if not np.isnan(call_row.get("call_iv_mean", np.nan)):
                cand.append(call_row["call_iv_mean"])
            if not np.isnan(put_row.get("put_iv_mean", np.nan)):
                cand.append(put_row["put_iv_mean"])
            strike_iv = np.nan if len(cand) == 0 else float(np.nanmean(cand))

        if np.isnan(strike_iv):
            missing_iv_count += 1
        total_iv_cells += 1

        rows.append({
            "strike": float(K),
            "call_oi": call_oi,
            "put_oi": put_oi,
            "net_oi": net_oi,
            "call_vol": call_vol,
            "put_vol": put_vol,
            "net_vol": net_vol,
            "iv": strike_iv,
            "expiry": expiry,
        })

    df = pd.DataFrame(rows)
    all_data.append(df)

if not all_data:
    st.error("No options data could be processed for the requested expiries.")
    st.stop()

nodes = pd.concat(all_data, ignore_index=True)

if missing_iv_count > 0:
    st.warning(f"{missing_iv_count} strike-IV cells missing out of {total_iv_cells}. GEX will be 0 for strikes without IV.")

# Robust expiry -> T calculation
nodes["expiry_dt"] = pd.to_datetime(nodes["expiry"], errors="coerce")
nodes = nodes[nodes["expiry_dt"].notna()].copy()
try:
    if nodes["expiry_dt"].dt.tz is not None:
        nodes["expiry_dt"] = nodes["expiry_dt"].dt.tz_convert("UTC").dt.tz_localize(None)
except Exception:
    nodes["expiry_dt"] = nodes["expiry_dt"].dt.tz_convert("UTC").dt.tz_localize(None) if hasattr(nodes["expiry_dt"].dt, "tz") else nodes["expiry_dt"]

today = pd.Timestamp.utcnow()
if getattr(today, "tzinfo", None) is not None:
    today = today.tz_convert("UTC").tz_localize(None)

nodes["T"] = (nodes["expiry_dt"] - today).dt.total_seconds() / (365.0 * 24 * 3600)
nodes["T"] = nodes["T"].clip(lower=0.0)

# compute gamma and GEX
nodes["gamma_per_share"] = nodes.apply(
    lambda r: bs_gamma(price, r["strike"], max(r["T"], 1e-9), float(r["iv"]) if not np.isnan(r["iv"]) else 0.0),
    axis=1
)

nodes["gex_shares"] = nodes["gamma_per_share"] * nodes["net_oi"] * CONTRACT_SIZE
nodes["gex_dollar"] = nodes["gex_shares"] * price

# select strikes around price
unique_strikes = np.sort(nodes["strike"].unique())
if unique_strikes.size == 0:
    st.error("No strikes found in the option chains.")
    st.stop()

closest_idx = int(np.abs(unique_strikes - price).argmin())
low_idx = max(0, closest_idx - STRIKE_RANGE)
high_idx = min(len(unique_strikes) - 1, closest_idx + STRIKE_RANGE)
selected_strikes = unique_strikes[low_idx : high_idx + 1]

nodes = nodes[nodes["strike"].isin(selected_strikes)].copy()
if nodes.empty:
    st.error("No nodes after selecting strikes around price. Try expanding the strike range.")
    st.stop()

# pick metric
if METRIC == "Open Interest":
    value_col = "net_oi"
elif METRIC == "Volume":
    value_col = "net_vol"
elif METRIC == "GEX (shares)":
    value_col = "gex_shares"
elif METRIC == "Dollar GEX":
    value_col = "gex_dollar"
else:
    value_col = "net_oi"

# summary 
st.subheader(f"{symbol}  •  Price: {price:.2f}")
st.markdown(f"Metric: **{METRIC}** — Showing {heatmap.shape[0]} strike levels and {heatmap.shape[1]} expiries")



# build heatmap (descending so highest on top)
expiry_type_label = OPTION_TYPE

heatmap = nodes.pivot_table(index="strike", columns="expiry", values=value_col, aggfunc="sum").fillna(0)
heatmap = heatmap.sort_index(ascending=False)

expiry_labels = [pd.to_datetime(x).date().isoformat() for x in heatmap.columns]

def format_strike(s):
    return f"{s:.2f}".rstrip("0").rstrip(".")

strike_labels = [format_strike(s) for s in heatmap.index]


z = heatmap.values

# king nodes
king_call = {}
king_put = {}
for col in heatmap.columns:
    col_series = heatmap[col]
    if col_series.size == 0:
        continue
    king_call[col] = float(col_series.idxmax())
    king_put[col] = float(col_series.idxmin())

# plot
fig, ax = plt.subplots(figsize=(12, max(6, len(strike_labels) * 0.18)))
cmap = plt.get_cmap("RdYlGn")
max_abs = np.nanmax(np.abs(z)) if z.size else 1
norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
im = ax.imshow(z, aspect="auto", cmap=cmap, norm=norm, origin="upper")
ax.set_xticks(np.arange(len(expiry_labels)))
ax.set_xticklabels(expiry_labels, rotation=30, ha="right")
ax.set_yticks(np.arange(len(strike_labels)))
ax.set_yticklabels(strike_labels)
ax.set_xlabel("Expiry (date)")
ax.set_ylabel("Strike")
ax.set_title(
    f"{symbol} | {expiry_type_label} Options | Price: {price:.2f} | Net {METRIC} Heatmap"
)


cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label(f"Net {METRIC} (Calls − Puts)")

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        val = z[i, j]
        txt = format_oi_value(val)
        strike_val = float(heatmap.index[i])
        expiry_val = heatmap.columns[j]
        is_king = (expiry_val in king_call and strike_val == king_call[expiry_val]) or (
            expiry_val in king_put and strike_val == king_put[expiry_val]
        )

        text_color = "white" if abs(val) > (max_abs * 0.35) else "black"
        fontweight = "bold" if is_king else "normal"
        fontsize = 10 if is_king else 8

        ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=fontsize, fontweight=fontweight)

        if is_king:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

# watermark
fig.text(
    0.99, 0.01,
    "EpicOptions",
    ha="right",
    va="bottom",
    fontsize=8,
    color="gray",
    alpha=0.8
)


plt.tight_layout()

st.pyplot(fig)
