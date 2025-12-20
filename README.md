# Options Net-OI Heatmap — Streamlit App

This Streamlit app visualizes net open interest (Calls − Puts) binned by strike and expiry, highlights "king" call and put nodes, computes Gamma Exposure (GEX), and provides a static heatmap for quick visual analysis.

Features
- Enter any ticker symbol supported by yfinance
- Choose metric: Open Interest, Volume, GEX (shares), or Dollar GEX
- Shows actual strikes (no binning) around current price with configurable strike range
- Static matplotlib heatmap with annotated values and king-node highlights
- CSV export includes GEX columns and inputs used

Why this app is useful for traders

- Surface positioning: The heatmap shows where option open interest and volume concentrate across strikes and expiries. High concentrations often act as short-term support/resistance.
- Gauge hedging sensitivity: GEX (Gamma Exposure) estimates how much option-related delta would change for a $1 move, helping you anticipate dealer hedging flows.
- Compare across expiries: Quickly see how positioning and gamma are distributed across upcoming cycles.
- Export & analyze: Download raw CSV to run further analytics or integrate with portfolio risk models.

How GEX is calculated

- For each strike we compute Black–Scholes gamma per share: Gamma = phi(d1) / (S * sigma * sqrt(T)), where phi is the standard normal PDF, S is spot, sigma is implied volatility, and T is time-to-expiry in years.
- Per-strike GEX (shares) = Gamma_per_share * (Calls OI − Puts OI) * 100
- Dollar GEX = GEX_shares * spot
- The app estimates strike IV from the option chain (OI-weighted average of call/put IV when both exist). Missing IVs are handled gracefully and the app warns when IV is sparse.

Limitations & caveats

- Data source: The app uses yfinance/Yahoo data which can be incomplete; for production or trading use a paid data source.
- IV availability: GEX accuracy depends on strike-level IV; missing IV reduces accuracy.
- Simplifying assumptions: Black–Scholes gamma (no dividends) is used — fine for quick analysis but not a full risk model.

Run locally
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

2. Start the app:
   ```bash
   streamlit run app.py
   ```

3. The app will open in your browser (default http://localhost:8501). Enter a ticker and adjust parameters.

If you want, I can help customize visual styles, add more metrics, or connect to a commercial options data feed.
