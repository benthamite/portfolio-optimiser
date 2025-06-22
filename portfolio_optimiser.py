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

# Initialize session state for editable start dates
if "editable_start_dates" not in st.session_state:
    st.session_state.editable_start_dates = {}

tickers_input = st.text_input("Enter tickers separated by commas:", "AAPL, MSFT, GOOGL")
original_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Update session state with default start dates for any new tickers
for t in original_tickers:
    if t not in st.session_state.editable_start_dates:
        st.session_state.editable_start_dates[t] = "1970-01-01" # Default initial start date

if original_tickers:
    # Initialize historical stats with defaults for all original_tickers
    mu_h = pd.Series(0.0, index=original_tickers, dtype=float)
    sig_h = pd.Series(0.01, index=original_tickers, dtype=float) # Default volatility to a small positive number
    rho_h = pd.DataFrame(np.eye(len(original_tickers)), index=original_tickers, columns=original_tickers, dtype=float)
    start_dates_h = pd.Series("N/A", index=original_tickers, dtype=str)

    # Attempt to download data only if there are tickers specified
    if original_tickers:
        try:
            # Determine the earliest start date from user selections for the global fetch
            all_user_start_dates_str = [
                st.session_state.editable_start_dates.get(t, "1970-01-01") for t in original_tickers
            ]
            valid_start_dates_obj = []
            for date_str in all_user_start_dates_str:
                try:
                    valid_start_dates_obj.append(pd.to_datetime(date_str))
                except (ValueError, TypeError):
                    pass # Ignore unparseable dates for min() calculation

            if not valid_start_dates_obj: # Fallback if no valid dates
                global_fetch_start_date_str = "1970-01-01"
            else:
                global_fetch_start_date_str = min(valid_start_dates_obj).strftime('%Y-%m-%d')

            # yf.download returns a DataFrame. ['Close'] selects close prices.
            # If only one ticker is successful, raw_dl_prices becomes a Series.
            # If multiple tickers are successful, it's a DataFrame.
            # If no tickers are successful, yf.download might return an empty DataFrame, leading to KeyError on ['Close'].
            raw_dl_prices = yf.download(original_tickers, start=global_fetch_start_date_str, interval="1mo", auto_adjust=True, progress=False)['Close']
            
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
                        # Slice data for each ticker based on its user-defined start date
                        data_for_returns_calculation_list = []
                        processed_tickers_for_returns = []

                        for t_col in valid_data.columns:
                            user_start_for_ticker_str = st.session_state.editable_start_dates.get(t_col, global_fetch_start_date_str)
                            ticker_series_sliced = valid_data[t_col].loc[user_start_for_ticker_str:]
                            if not ticker_series_sliced.empty:
                                data_for_returns_calculation_list.append(ticker_series_sliced)
                                processed_tickers_for_returns.append(t_col)
                        
                        returns_df = pd.DataFrame(columns=original_tickers) # Default to empty
                        if data_for_returns_calculation_list:
                            final_prices_df = pd.concat(data_for_returns_calculation_list, axis=1, join='outer')
                            if not final_prices_df.empty: # Ensure columns are named correctly if some series were all NaN and dropped by concat
                                final_prices_df.columns = [s.name for s in data_for_returns_calculation_list if not s.empty]

                            returns = final_prices_df.pct_change().dropna(how='all')
                            if not returns.empty:
                                returns_df = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
                        
                        # Ensure returns_df has columns for all original_tickers, even if empty, for consistent stat calculation
                        missing_cols = [tc for tc in original_tickers if tc not in returns_df.columns]
                        for mc in missing_cols:
                            returns_df[mc] = np.nan

                        if not returns_df.empty:
                            # Calculate stats only for columns with enough data points for std deviation
                            final_cols_for_stats = [col for col in returns_df.columns if returns_df[col].count() > 1]

                            if final_cols_for_stats:
                                mu_calculated = returns_df[final_cols_for_stats].mean()
                                sig_calculated = returns_df[final_cols_for_stats].std()
                                # Get the first valid index (date) for each column in returns_df
                                for col in final_cols_for_stats:
                                    first_valid_idx = returns_df[col].first_valid_index()
                                    if pd.notnull(first_valid_idx):
                                        start_dates_h[col] = first_valid_idx.strftime('%Y-%m-%d')

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
    c1_header, c2_header, c3_header = st.columns(3)
    with c1_header:
        st.markdown("**Return**")
    with c2_header:
        st.markdown("**Volatility**")
    with c3_header:
        st.markdown("**Start Date**")

    for t in tickers:
        c1, c2, c3 = st.columns(3)
        with c1:
            mu = st.number_input(f"{t}", value=round(mu_h[t], 4), format="%.4f", key=f"mu_{t}", label_visibility="collapsed")
        with c2:
            vol = st.number_input(f"{t}", value=round(sig_h[t], 4), format="%.4f", key=f"vol_{t}", label_visibility="collapsed")
        with c3:
            user_defined_start_str = st.session_state.editable_start_dates.get(t, "1970-01-01")
            try:
                user_defined_date_obj = pd.to_datetime(user_defined_start_str).date()
            except (ValueError, TypeError): # Handle parsing errors or "N/A"
                user_defined_date_obj = pd.to_datetime("1970-01-01").date()

            new_selected_date_obj = st.date_input(
                label=f"Start Date for {t}", # Label for screen readers, etc.
                value=user_defined_date_obj,
                min_value=pd.to_datetime("1900-01-01").date(),
                max_value=pd.Timestamp.today().date(),
                key=f"date_input_{t}", # Unique key for the date input widget
                label_visibility="collapsed"
            )
            if new_selected_date_obj:
                st.session_state.editable_start_dates[t] = new_selected_date_obj.strftime('%Y-%m-%d')
            
            # Optionally, to inform the user of the actual data start if different (e.g., due to market holidays)
            # You could add another small text display here using start_dates_h[t] if desired.
            # For now, the editable date input is the primary interface for this column.

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
