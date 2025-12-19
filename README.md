```markdown
# Options Net-OI Heatmap — Streamlit App

This small Streamlit app visualizes net open interest (Calls − Puts) binned by strike and expiry, highlights "king" call and put nodes, and provides an interactive heatmap.

Features
- Enter any ticker symbol supported by yfinance
- Adjust bin size, strike range, and number of expiries to include
- Interactive Plotly heatmap with hover details and annotated values
- Caching to speed up repeated lookups

Requirements
- Python 3.10+ recommended

Included files
- `app.py` — the Streamlit application
- `requirements.txt` — Python dependencies

Run locally
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

2. Start the app:
   ```bash
   streamlit run app.py
   ```

3. The app will open in your browser (default http://localhost:8501). Enter a ticker and adjust parameters.

Deploy & share publicly
- Easiest: Deploy to Streamlit Community Cloud:
  1. Push this repository to GitHub.
  2. Go to https://share.streamlit.io, sign in with GitHub, and create a new app pointing to your repo and `app.py`.
  3. Once deployed, Streamlit Cloud gives you a public URL you can share.

- Alternatives: Render, Heroku (containerize), or any host that supports Python web services. Streamlit Cloud is recommended for fastest sharing.

Notes & privacy
- This app fetches data from Yahoo Finance through the `yfinance` package.
- Be mindful of API/website rate limits when making many requests.

If you want, I can:
- Create a GitHub repo with these files (if you give me repo details), or
- Modify styling, add caching parameters, or add CSV export and scheduled updates.
```
