import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import streamlit as st
import plotly.graph_objects as go

# === Core Functions ===

def compute_frontier(mu, cov, theta_range):
    n = len(mu)
    results = {'Return': [], 'Volatility': [], 'Weights': []}
    def objective(w, mu, cov, theta):
        return -w @ mu + theta * (w @ cov @ w)
    for theta in theta_range:
        result = minimize(
            objective, np.ones(n) / n, args=(mu, cov, theta),
            method='SLSQP', bounds=[(0, 1)] * n,
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
            options={'ftol': 1e-9})
        if result.success:
            w = result.x
            results['Return'].append(w @ mu)
            results['Volatility'].append(np.sqrt(w @ cov @ w))
            results['Weights'].append(w)
    return pd.DataFrame(results)

def build_plot(df, tickers):
    hover_texts = []
    for w, r, v in zip(df['Weights'], df['Return'], df['Volatility']):
        text = (f"<b>Expected Return:</b> {r*100:.2f}%<br>"
                f"<b>Volatility:</b> {v*100:.2f}%<br><br>"
                + "<b>Breakdown:</b><br>"
                + "<br>".join([f"{t}: {w_i*100:.2f}%" for t, w_i in zip(tickers, w)]))
        hover_texts.append(text)
    fig = go.Figure(go.Scatter(
        x=df['Volatility'] * 100,
        y=df['Return'] * 100,
        mode='markers+lines',
        text=hover_texts,
        hoverinfo='text',
        marker=dict(size=6),
        line=dict(color='purple'),
        name="Efficient Frontier"
    ))
    fig.update_layout(
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        template="plotly_white",
        hovermode="closest"
    )
    return fig

# === Streamlit UI ===

st.title("Portfolio Optimiser")

tickers_input = st.text_input("Enter tickers separated by commas:", "AAPL, MSFT, GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if tickers:
    raw = yf.download(tickers, start="1970-01-01", interval="1mo", auto_adjust=True, progress=False)['Close']
    raw = raw.dropna()
    data = raw[[t for t in tickers if t in raw.columns]].dropna()
    if data.empty:
        st.error("None of the tickers returned usable data.")
        st.stop()

    returns = data.pct_change().dropna()
    mu_h  = returns.mean()
    sig_h = returns.std()
    rho_h = returns.corr()
    tickers = list(rho_h.columns)
    n = len(tickers)

    # === Asset Inputs ===
    st.subheader("📋 Asset inputs")
    asset_data = {}

    # Add column headers
    c1_header, c2_header = st.columns(2)
    with c1_header:
        st.markdown("**Return**")
    with c2_header:
        st.markdown("**Volatility**")

    for t in tickers:
        c1, c2 = st.columns(2)
        with c1:
            mu = st.number_input(f"{t}", value=round(mu_h[t], 4), format="%.4f", key=f"mu_{t}", label_visibility="collapsed")
        with c2:
            vol = st.number_input(f"{t}", value=round(sig_h[t], 4), format="%.4f", key=f"vol_{t}", label_visibility="collapsed")
        asset_data[t] = {"mu": mu, "vol": vol}

    # === Correlation Matrix ===
    st.subheader("📐 Correlation matrix")
    corr_matrix = np.eye(n)
    display_matrix = []
    for i in range(n):
        row = []
        cols = st.columns(n)
        for j in range(n):
            if j < i:
                cols[j].markdown("‎")
                row.append("–")
            elif i == j:
                cols[j].markdown("**1.0000**")
                row.append("1.0000")
            else:
                default = round(rho_h.iloc[i, j], 4)
                val = cols[j].number_input("", value=default, format="%.4f", key=f"corr_{i}_{j}")
                corr_matrix[i, j] = corr_matrix[j, i] = val
                row.append(f"{val:.4f}")
        display_matrix.append(row)

    # === Compute Frontier ===
    mu = np.array([asset_data[t]['mu'] for t in tickers])
    vol = np.array([asset_data[t]['vol'] for t in tickers])
    cov = corr_matrix * np.outer(vol, vol)

    st.subheader("📉 Efficient frontier")
    frontier = compute_frontier(mu, cov, np.logspace(-3, 3, 100))
    st.plotly_chart(build_plot(frontier, tickers), use_container_width=True)

    if st.checkbox("Show portfolio weights table"):
        df_w = pd.DataFrame(frontier['Weights'].tolist(), columns=tickers)
        df_w.insert(0, "Volatility (%)", frontier['Volatility'] * 100)
        df_w.insert(1, "Expected Return (%)", frontier['Return'] * 100)
        st.dataframe(df_w.style.format(precision=2))

    st.download_button("Download CSV", frontier.to_csv(index=False), file_name="efficient_frontier.csv")
