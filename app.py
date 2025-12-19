import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Options Net-OI Heatmap", layout="wide")

# =============================
# HELPERS
# =============================
def format_oi_value(val):
    # Format numeric values for annotations
    try:
        v = int(val)
    except Exception:
        return ""
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,}"

# Don't cache objects that are not easily serializable (e.g. yf.Ticker)
def fetch_ticker(ticker_symbol: str):
    # Return a fresh yf.Ticker each call (not cached)
    return yf.Ticker(ticker_symbol)

@st.cache_data(show_spinner=False)
def get_price(ticker_symbol: str):
    # Cache the numeric price (serializable float)
    t = yf.Ticker(ticker_symbol)
    hist = t.history(period="1d")
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])

def get_option_chain_for_expiry(ticker, expiry):
    # Don't cache the Ticker or chain object; return DataFrames directly
    try:
        chain = ticker.option_chain(expiry)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return None, None

# =============================
# SIDEBAR / INPUTS
# =============================
st.title("Options Net Open Interest Heatmap")
st.markdown("Visualize binned Net Open Interest (Calls − Puts) across strikes and upcoming expiries.")

col_inputs, col_info = st.columns([1, 2])

with col_inputs:
    symbol = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    BIN_SIZE = st.number_input("Bin size (strike step)", min_value=1, max_value=100, value=5, step=1)
    STRIKE_RANGE = st.number_input("Strike bins either side of center (count)", min_value=1, max_value=100, value=20, step=1)
    EXPIRY_COUNT = st.slider("Number of expiries to include", min_value=1, max_value=12, value=4)
    refresh = st.button("Update")

with col_info:
    st.markdown("Usage tips:")
    st.markdown("- Try liquid tickers (AAPL, SPY) for richer option chains.")
    st.markdown("- Increase bin size to aggregate wider strike ranges.")
    st.markdown(f"- Report generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC")

if not symbol:
    st.info("Enter a ticker symbol to continue.")
    st.stop()

# =============================
# FETCH DATA
# =============================
# Use get_price (cached for numeric return) and fetch_ticker (non-cached)
price = None
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

expiries = all_expiries[:EXPIRY_COUNT]

all_data = []

for expiry in expiries:
    calls, puts = get_option_chain_for_expiry(ticker, expiry)
    if calls is None or puts is None:
        continue

    # Ensure numeric strike & openInterest columns exist
    if "strike" not in calls.columns or "openInterest" not in calls.columns:
        continue
    if "strike" not in puts.columns or "openInterest" not in puts.columns:
        continue

    # Bin strikes
    calls = calls.copy()
    puts = puts.copy()

    calls["bin"] = (calls["strike"] // BIN_SIZE) * BIN_SIZE
    puts["bin"] = (puts["strike"] // BIN_SIZE) * BIN_SIZE

    call_oi = calls.groupby("bin")["openInterest"].sum()
    put_oi = puts.groupby("bin")["openInterest"].sum()

    net_oi = call_oi.subtract(put_oi, fill_value=0)

    df = net_oi.reset_index()
    df.columns = ["strike_bin", "net_oi"]
    df["expiry"] = expiry

    all_data.append(df)

if not all_data:
    st.error("No options data could be processed for the requested expiries.")
    st.stop()

nodes = pd.concat(all_data, ignore_index=True)

# =============================
# LIMIT TO STRIKES AROUND PRICE
# =============================
nodes["dist"] = (nodes["strike_bin"] - price).abs()
center_row = nodes.loc[nodes["dist"].idxmin()]
center_bin = int(center_row["strike_bin"])

valid_bins = [center_bin + i * BIN_SIZE for i in range(-STRIKE_RANGE, STRIKE_RANGE + 1)]
nodes = nodes[nodes["strike_bin"].isin(valid_bins)]

if nodes.empty:
    st.error("No binned nodes remained after limiting to the strike range. Try increasing the strike range or bin size.")
    st.stop()

# =============================
# BUILD HEATMAP
# =============================
heatmap = nodes.pivot_table(index="strike_bin", columns="expiry", values="net_oi", aggfunc="sum").fillna(0)
heatmap = heatmap.sort_index(ascending=False)

# =============================
# FIND KING CALL & KING PUT NODES
# =============================
king_call = {}
king_put = {}
for col in heatmap.columns:
    column_vals = heatmap[col]
    king_call[col] = int(column_vals.idxmax())
    king_put[col] = int(column_vals.idxmin())

# =============================
# PREPARE PLOTLY HEATMAP
# =============================
z = heatmap.values
x = [str(c) for c in heatmap.columns]
y = [int(v) for v in heatmap.index]

# Build text annotations (formatted)
text = []
for i, strike in enumerate(y):
    row = []
    for j, expiry in enumerate(heatmap.columns):
        val = z[i, j]
        label = format_oi_value(val)
        row.append(label)
    text.append(row)

colorscale = [
    [0.0, "rgb(165,0,38)"],
    [0.5, "rgb(255,255,191)"],
    [1.0, "rgb(0,104,55)"]
]

fig = go.Figure(
    data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        text=text,
        hovertemplate="<b>Expiry:</b> %{x}<br><b>Strike bin:</b> %{y}<br><b>Net OI:</b> %{z}<extra></extra>",
        colorscale=colorscale,
        colorbar=dict(title="Net Open Interest (Calls − Puts)"),
        zmid=0
    )
)

# non-king labels
non_king_x = []
non_king_y = []
non_king_text = []
non_king_font_color = []
max_abs_z = np.nanmax(np.abs(z)) if z.size else 0
for i, strike in enumerate(y):
    for j, expiry in enumerate(heatmap.columns):
        strike_val = strike
        expiry_val = expiry
        val = z[i, j]
        label = format_oi_value(val)
        if strike_val == king_call[expiry_val] or strike_val == king_put[expiry_val]:
            continue
        non_king_x.append(str(expiry_val))
        non_king_y.append(strike_val)
        non_king_text.append(label)
        non_king_font_color.append("black" if abs(val) < (max_abs_z * 0.35 if max_abs_z>0 else 1) else "white")

fig.add_trace(
    go.Scatter(
        x=non_king_x,
        y=non_king_y,
        mode="text",
        text=non_king_text,
        textfont=dict(color=non_king_font_color, size=10),
        hoverinfo="skip",
        showlegend=False
    )
)

# king labels (bigger and white)
king_x = []
king_y = []
king_texts = []
for j, expiry in enumerate(heatmap.columns):
    for kind, strike in [("call", king_call[expiry]), ("put", king_put[expiry])]:
        king_x.append(str(expiry))
        king_y.append(str(strike))
        idx_row = list(y).index(strike)
        val = z[idx_row, j]
        king_texts.append(format_oi_value(val))

fig.add_trace(
    go.Scatter(
        x=king_x,
        y=king_y,
        mode="text",
        text=king_texts,
        textfont=dict(color="white", size=12, family="Arial", weight="bold"),
        hoverinfo="skip",
        showlegend=False
    )
)

fig.update_layout(
    title=f"{symbol} | Price: {price:.2f} | Net OI Heatmap (King Call & Put Nodes)",
    xaxis_title="Expiry",
    yaxis_title="Strike Bin",
    yaxis_autorange="reversed",
    height=700,
    margin=dict(l=120, r=20, t=80, b=120),
)

# =============================
# SHOW SUMMARY & PLOT
# =============================
st.subheader(f"{symbol}  •  Price: {price:.2f}")
st.markdown(f"Showing {len(heatmap.index)} strike bins and {len(heatmap.columns)} expiries")

with st.expander("King nodes (by expiry)"):
    rows = []
    for expiry in heatmap.columns:
        rows.append(f"- {expiry}: King Call = {king_call[expiry]}, King Put = {king_put[expiry]}")
    st.markdown("\n".join(rows))

st.plotly_chart(fig, use_container_width=True)

# Option to download CSV
csv = nodes[["expiry", "strike_bin", "net_oi"]].sort_values(["expiry", "strike_bin"])
csv_bytes = csv.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV of binned net OI", data=csv_bytes, file_name=f"{symbol}_net_oi_{datetime.utcnow().date()}.csv", mime="text/csv")
