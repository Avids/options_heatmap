import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Options Net-OI/Volume Heatmap", layout="wide")

# =============================
# HELPERS
# =============================
def format_oi_value(val):
    try:
        v = int(val)
    except Exception:
        return ""
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,}"

def fetch_ticker(ticker_symbol: str):
    # Do not cache the Ticker object (not serializable)
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

# =============================
# SIDEBAR / INPUTS
# =============================
st.title("Options Net Heatmap (Open Interest or Volume)")
st.markdown("Visualize Net Open Interest or Net Volume (Calls − Puts) across actual strikes and expiries.")

col_inputs, col_info = st.columns([1, 2])

with col_inputs:
    symbol = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    METRIC = st.selectbox("Metric", options=["Open Interest", "Volume"])
    STRIKE_RANGE = st.number_input(
        "Strikes above / below current price (count)",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )
    EXPIRY_COUNT = st.slider(
        "Number of expiries to include", min_value=1, max_value=24, value=4
    )
    refresh = st.button("Update")

with col_info:
    st.markdown("Usage tips:")
    st.markdown("- Choose metric: Open Interest (openInterest) or Volume (volume).")
    st.markdown("- Shows actual strike levels (no binning).")
    st.markdown("- Choose how many strikes above and below current price to show.")
    st.markdown(f"- Report generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC")

if not symbol:
    st.info("Enter a ticker symbol to continue.")
    st.stop()

metric_col = "openInterest" if METRIC == "Open Interest" else "volume"
colorbar_title = f"Net {METRIC} (Calls − Puts)"

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

expiries = all_expiries[:EXPIRY_COUNT]

all_data = []
missing_metric_warned = False

for expiry in expiries:
    calls, puts = get_option_chain_for_expiry(ticker, expiry)
    if calls is None or puts is None:
        continue

    # Ensure strike column exists
    if "strike" not in calls.columns or "strike" not in puts.columns:
        continue

    # If metric column missing, fill with zeros and warn once
    if metric_col not in calls.columns:
        calls[metric_col] = 0
        if not missing_metric_warned and METRIC == "Volume":
            st.warning("Volume column not present on some option chains; treating missing values as 0.")
            missing_metric_warned = True
    if metric_col not in puts.columns:
        puts[metric_col] = 0
        if not missing_metric_warned and METRIC == "Volume":
            st.warning("Volume column not present on some option chains; treating missing values as 0.")
            missing_metric_warned = True

    # Group by strike and sum the metric
    call_vals = calls.groupby("strike")[metric_col].sum()
    put_vals = puts.groupby("strike")[metric_col].sum()

    net = call_vals.subtract(put_vals, fill_value=0)

    df = net.reset_index()
    df.columns = ["strike", "net_metric"]
    df["expiry"] = expiry

    all_data.append(df)

if not all_data:
    st.error("No options data could be processed for the requested expiries.")
    st.stop()

nodes = pd.concat(all_data, ignore_index=True)

# =============================
# SELECT STRIKES AROUND PRICE (actual strikes)
# =============================
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

# =============================
# BUILD HEATMAP (simpler approach: sort DESC so top is highest strike)
# =============================
heatmap = nodes.pivot_table(index="strike", columns="expiry", values="net_metric", aggfunc="sum").fillna(0)

# sort strikes descending so highest strike is first row (displayed at top)
heatmap = heatmap.sort_index(ascending=False)

expiry_labels = [pd.to_datetime(x).date().isoformat() for x in heatmap.columns]
strike_labels = [str(int(s)) for s in heatmap.index]  # highest->lowest order

z = heatmap.values  # rows correspond to strike_labels in same order (highest->lowest)

n_rows, n_cols = z.shape
x_pos = list(range(n_cols))
y_pos = list(range(n_rows))

# customdata shaped (n_rows, n_cols, 2) for hover (expiry label, strike label)
customdata = np.empty((n_rows, n_cols, 2), dtype=object)
for i in range(n_rows):
    for j in range(n_cols):
        customdata[i, j, 0] = expiry_labels[j]
        customdata[i, j, 1] = strike_labels[i]

# =============================
# KING NODES (based on net_metric)
# =============================
king_call = {}
king_put = {}
for col in heatmap.columns:
    col_series = heatmap[col]
    if col_series.size == 0:
        continue
    king_call[col] = int(col_series.idxmax())
    king_put[col] = int(col_series.idxmin())

# =============================
# PLOTLY HEATMAP
# =============================
fig = go.Figure(
    data=go.Heatmap(
        z=z,
        x=x_pos,
        y=y_pos,
        customdata=customdata,
        colorscale=[
            [0.0, "rgb(165,0,38)"],
            [0.5, "rgb(255,255,191)"],
            [1.0, "rgb(0,104,55)"],
        ],
        colorbar=dict(title=colorbar_title),
        zmid=0,
        hovertemplate="<b>Expiry:</b> %{customdata[0]}<br><b>Strike:</b> %{customdata[1]}<br><b>Net:</b> %{z}<extra></extra>",
    )
)

# overlay text (non-king)
text_matrix = [[format_oi_value(z[i, j]) for j in range(n_cols)] for i in range(n_rows)]
max_abs_z = np.nanmax(np.abs(z)) if z.size else 0

non_king_x = []
non_king_y = []
non_king_text = []
non_king_color = []

for i in range(n_rows):
    for j in range(n_cols):
        strike_val = int(heatmap.index[i])
        expiry_val = heatmap.columns[j]
        if (expiry_val in king_call and strike_val == king_call[expiry_val]) or (
            expiry_val in king_put and strike_val == king_put[expiry_val]
        ):
            continue
        non_king_x.append(j)
        non_king_y.append(i)
        non_king_text.append(text_matrix[i][j])
        non_king_color.append("black" if abs(z[i, j]) < (max_abs_z * 0.35 if max_abs_z > 0 else 1) else "white")

fig.add_trace(
    go.Scatter(
        x=non_king_x,
        y=non_king_y,
        mode="text",
        text=non_king_text,
        textfont=dict(color=non_king_color, size=10),
        hoverinfo="skip",
        showlegend=False,
    )
)

# king labels (bigger)
king_x = []
king_y = []
king_texts = []
for j, expiry in enumerate(heatmap.columns):
    for kind, strike in [("call", king_call.get(expiry)), ("put", king_put.get(expiry))]:
        if strike is None:
            continue
        try:
            i = list(heatmap.index).index(strike)
        except ValueError:
            continue
        king_x.append(j)
        king_y.append(i)
        king_texts.append(format_oi_value(z[i, j]))

fig.add_trace(
    go.Scatter(
        x=king_x,
        y=king_y,
        mode="text",
        text=king_texts,
        textfont=dict(color="white", size=12, family="Arial", weight="bold"),
        hoverinfo="skip",
        showlegend=False,
    )
)

fig.update_layout(
    title=f"{symbol} | Price: {price:.2f} | Net {METRIC} Heatmap (King Call & Put Nodes)",
    xaxis=dict(
        title="Expiry (date)",
        tickmode="array",
        tickvals=x_pos,
        ticktext=expiry_labels,
        tickangle=-30,
        automargin=True,
    ),
    yaxis=dict(
        title="Strike",
        tickmode="array",
        tickvals=y_pos,
        ticktext=strike_labels,
    ),
    height=700,
    margin=dict(l=120, r=20, t=80, b=120),
)

# =============================
# SHOW SUMMARY & PLOT
# =============================
st.subheader(f"{symbol}  •  Price: {price:.2f}")
st.markdown(f"Metric: **{METRIC}** — Showing {len(strike_labels)} strike levels and {len(expiry_labels)} expiries")

with st.expander("King nodes (by expiry)"):
    rows = []
    for expiry in heatmap.columns:
        rows.append(
            f"- {pd.to_datetime(expiry).date().isoformat()}: King Call = {king_call.get(expiry)}, King Put = {king_put.get(expiry)}"
        )
    st.markdown("\n".join(rows))

st.plotly_chart(fig, use_container_width=True)

# Option to download CSV (include metric name)
csv = nodes[["expiry", "strike", "net_metric"]].sort_values(["expiry", "strike"])
csv = csv.rename(columns={"net_metric": f"net_{METRIC.replace(' ', '_').lower()}"})
csv_bytes = csv.to_csv(index=False).encode("utf-8")
st.download_button(
    f"Download CSV of net {METRIC} by strike",
    data=csv_bytes,
    file_name=f"{symbol}_net_{METRIC.replace(' ', '_').lower()}_{datetime.utcnow().date()}.csv",
    mime="text/csv",
)
