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
original_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if original_tickers:
    # Initialize historical stats with defaults for all original_tickers
    mu_h = pd.Series(0.0, index=original_tickers, dtype=float)
    sig_h = pd.Series(0.01, index=original_tickers, dtype=float) # Default volatility to a small positive number
    rho_h = pd.DataFrame(np.eye(len(original_tickers)), index=original_tickers, columns=original_tickers, dtype=float)

    # Attempt to download data only if there are tickers specified
    if original_tickers:
        try:
            # yf.download returns a DataFrame. ['Close'] selects close prices.
            # If only one ticker is successful, raw_dl_prices becomes a Series.
            # If multiple tickers are successful, it's a DataFrame.
            # If no tickers are successful, yf.download might return an empty DataFrame, leading to KeyError on ['Close'].
            raw_dl_prices = yf.download(original_tickers, start="1970-01-01", interval="1mo", auto_adjust=True, progress=False)['Close']
            
            raw_dl_prices_df = None
            if isinstance(raw_dl_prices, pd.Series):
                series_name = raw_dl_prices.name
                # If only one ticker was in original_tickers and yf names the series 'Close' (common for single ticker)
                if len(original_tickers) == 1 and series_name == 'Close':
                     raw_dl_prices_df = raw_dl_prices.to_frame(name=original_tickers[0])
                else: # yf usually names the series by the ticker if one of multiple requested tickers succeeded
                     raw_dl_prices_df = raw_dl_prices.to_frame() 
            elif isinstance(raw_dl_prices, pd.DataFrame):
                raw_dl_prices_df = raw_dl_prices
            else: # Should not happen if yf.download worked, but as a fallback
                raw_dl_prices_df = pd.DataFrame()


            if not raw_dl_prices_df.empty:
                # Filter for columns that are in original_tickers and have actual data
                actual_tickers_with_data = [t for t in original_tickers if t in raw_dl_prices_df.columns and not raw_dl_prices_df[t].isnull().all()]

                if actual_tickers_with_data:
                    # Use .copy() to avoid SettingWithCopyWarning later
                    valid_data = raw_dl_prices_df[actual_tickers_with_data].dropna(how='all').copy()
                    
                    if not valid_data.empty:
                        returns = valid_data.pct_change().dropna(how='all')

                        if not returns.empty:
                            returns_df = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
                            
                            # Calculate stats only for columns with enough data points for std deviation
                            final_cols_for_stats = [col for col in returns_df.columns if returns_df[col].count() > 1]

                            if final_cols_for_stats:
                                mu_calculated = returns_df[final_cols_for_stats].mean()
                                sig_calculated = returns_df[final_cols_for_stats].std()

                                mu_h.update(mu_calculated)
                                sig_h.update(sig_calculated)
                                # Ensure sig_h doesn't have NaNs (e.g. if std was 0 or somehow became NaN) and is not zero
                                sig_h.fillna(0.01, inplace=True)
                                sig_h[sig_h == 0] = 0.01
                                
                                if len(final_cols_for_stats) >= 1:
                                    rho_calculated = returns_df[final_cols_for_stats].corr()
                                    # Update rho_h carefully, only for pairs present in rho_calculated
                                    for t1_calc in rho_calculated.index:
                                        for t2_calc in rho_calculated.columns:
                                            if t1_calc in rho_h.index and t2_calc in rho_h.columns:
                                                rho_h.loc[t1_calc, t2_calc] = rho_calculated.loc[t1_calc, t2_calc]
        except KeyError: # Specifically catch if 'Close' key is not found (e.g. all tickers failed)
            st.warning("Could not retrieve 'Close' prices for any of the specified tickers. Please check ticker symbols or enter data manually.")
        except Exception as e:
            st.warning(f"An error occurred while fetching/processing data: {e}. Please check ticker symbols or enter data manually.")

    # Use the original list of tickers for the UI
    tickers = original_tickers 
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
