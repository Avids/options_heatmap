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
CONTRACT_SIZE = 100

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

@st.cache_data(show_spinner=False, ttl=300)
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

# Vectorized Black-Scholes gamma
def bs_gamma_vectorized(S, K, T, sigma):
    """Vectorized BS gamma calculation - much faster than apply()"""
    # Handle edge cases
    mask = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    gamma = np.zeros_like(K, dtype=float)
    
    if not mask.any():
        return gamma
    
    sqrtT = np.sqrt(T[mask])
    d1 = (np.log(S / K[mask]) + 0.5 * sigma[mask]**2 * T[mask]) / (sigma[mask] * sqrtT)
    pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2 * pi)
    gamma[mask] = pdf_d1 / (S * sigma[mask] * sqrtT)
    
    return gamma

def is_friday(date):
    return date.weekday() == 4

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
    OPTION_TYPE = st.selectbox(
        "Option type",
        options=["Weekly", "Monthly", "Weekly + Monthly"]
    )
    STRIKE_RANGE = st.number_input(
        "Strikes above / below current price (count)",
        min_value=1, max_value=200, value=20, step=1
    )
    EXPIRY_COUNT = st.slider(
        "Number of expiries to include (max 12)", min_value=1, max_value=12, value=4
    )
    refresh = st.button("Update (re-fetch)")

with col_info:
    st.markdown("- Metric: choose Open Interest, Volume, GEX (shares) or Dollar GEX.")
    st.markdown("- GEX (shares): change in total delta (in shares) for a $1 move.")
    st.markdown(f"- Report generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC")

if not symbol:
    st.info("Enter a ticker symbol to continue.")
    st.stop()

with st.expander("About / Help", expanded=False):
    st.markdown("""
    ## What this app shows
    - A static heatmap (strikes × expiries) of Calls − Puts for the selected metric.
    
    ## How to read it
    - X axis: expiries (one column per expiry).
    - Y axis: strikes (highest shown at the top).
    - Colors: green = net call dominance; red = net put dominance.
    - King nodes (largest positive/negative per expiry) are outlined and bolded.
    
    ## GEX calculation
    - Gamma per share (Black-Scholes) using strike IV, spot price and time to expiry.
    - GEX_shares = Gamma × (Calls OI − Puts OI) × 100.
    """)

# =============================
# FETCH DATA (OPTIMIZED)
# =============================
with st.spinner(f"Fetching data for {symbol} ..."):
    price, all_expiries, ticker = fetch_ticker_data(symbol)

if price is None or not all_expiries:
    st.error(f"Could not fetch data for '{symbol}'. Check the ticker.")
    st.stop()

# Filter expiries by type
expiry_dates = pd.to_datetime(all_expiries, errors="coerce")
expiry_map = dict(zip(all_expiries, expiry_dates))

filtered_expiries = []
for e, d in expiry_map.items():
    if pd.isna(d):
        continue
    if OPTION_TYPE == "Monthly" and is_third_friday(d):
        filtered_expiries.append(e)
    elif OPTION_TYPE == "Weekly" and is_friday(d) and not is_third_friday(d):
        filtered_expiries.append(e)
    elif OPTION_TYPE == "Weekly + Monthly" and is_friday(d):
        filtered_expiries.append(e)

expiries = filtered_expiries[:min(EXPIRY_COUNT, 12)]

# PARALLEL PROCESSING OF EXPIRIES
with st.spinner("Processing option chains..."):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_expiry, (ticker, exp, price)) for exp in expiries]
        all_data = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_data.append(result)

if not all_data:
    st.error("No options data could be processed.")
    st.stop()

nodes = pd.concat(all_data, ignore_index=True)

# Count missing IVs
missing_iv_count = nodes["iv"].isna().sum()
if missing_iv_count > 0:
    st.warning(f"{missing_iv_count} strikes missing IV. GEX will be 0 for these.")

# Calculate T (time to expiry) - VECTORIZED
nodes["expiry_dt"] = pd.to_datetime(nodes["expiry"], errors="coerce")
nodes = nodes[nodes["expiry_dt"].notna()].copy()

today = pd.Timestamp.utcnow().tz_localize(None)
nodes["T"] = (nodes["expiry_dt"].dt.tz_localize(None) - today).dt.total_seconds() / (365.0 * 24 * 3600)
nodes["T"] = nodes["T"].clip(lower=1e-9)

# VECTORIZED GAMMA CALCULATION
nodes["iv_filled"] = nodes["iv"].fillna(0.0)
nodes["gamma_per_share"] = bs_gamma_vectorized(
    price, 
    nodes["strike"].values, 
    nodes["T"].values, 
    nodes["iv_filled"].values
)

nodes["gex_shares"] = nodes["gamma_per_share"] * nodes["net_oi"] * CONTRACT_SIZE
nodes["gex_dollar"] = nodes["gex_shares"] * price

# Filter strikes around price
unique_strikes = np.sort(nodes["strike"].unique())
closest_idx = np.abs(unique_strikes - price).argmin()
low_idx = max(0, closest_idx - STRIKE_RANGE)
high_idx = min(len(unique_strikes) - 1, closest_idx + STRIKE_RANGE)
selected_strikes = unique_strikes[low_idx:high_idx + 1]

nodes = nodes[nodes["strike"].isin(selected_strikes)].copy()

if nodes.empty:
    st.error("No nodes after filtering strikes.")
    st.stop()

# Select metric
metric_map = {
    "Open Interest": "net_oi",
    "Volume": "net_vol",
    "GEX (shares)": "gex_shares",
    "Dollar GEX": "gex_dollar"
}
value_col = metric_map[METRIC]

# Build heatmap
heatmap = nodes.pivot_table(index="strike", columns="expiry", values=value_col, aggfunc="sum").fillna(0)
heatmap = heatmap.sort_index(ascending=False)

expiry_labels = [pd.to_datetime(x).date().isoformat() for x in heatmap.columns]
strike_labels = [f"{s:.2f}".rstrip("0").rstrip(".") for s in heatmap.index]

z = heatmap.values

# Find king nodes
king_call = {col: float(heatmap[col].idxmax()) for col in heatmap.columns}
king_put = {col: float(heatmap[col].idxmin()) for col in heatmap.columns}

# PLOT
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
ax.set_title(f"{symbol} | {OPTION_TYPE} Options | Price: {price:.2f} | Net {METRIC} Heatmap")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label(f"Net {METRIC} (Calls − Puts)")

# Add text annotations
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        val = z[i, j]
        txt = format_oi_value(val)
        strike_val = float(heatmap.index[i])
        expiry_val = heatmap.columns[j]
        is_king = (strike_val == king_call.get(expiry_val)) or (strike_val == king_put.get(expiry_val))
        
        text_color = "white" if abs(val) > (max_abs * 0.35) else "black"
        fontweight = "bold" if is_king else "normal"
        fontsize = 10 if is_king else 8
        
        ax.text(j, i, txt, ha="center", va="center", color=text_color, 
                fontsize=fontsize, fontweight=fontweight)
        
        if is_king:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, 
                                edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

fig.text(0.99, 0.01, "EpicOptions", ha="right", va="bottom", 
         fontsize=9, color="gray", alpha=0.9)

st.subheader(f"{symbol} • Price: {price:.2f}")
st.markdown(f"Metric: **{METRIC}** — Showing {heatmap.shape[0]} strikes and {heatmap.shape[1]} expiries")

plt.tight_layout()
st.pyplot(fig)
