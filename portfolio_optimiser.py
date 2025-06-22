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

# This section will be moved and updated after initial data fetch
# to use actual earliest dates.

if original_tickers:
    # Initialize historical stats with defaults for all original_tickers
    mu_h = pd.Series(0.0, index=original_tickers, dtype=float)
    sig_h = pd.Series(0.01, index=original_tickers, dtype=float) # Default volatility to a small positive number
    rho_h = pd.DataFrame(np.eye(len(original_tickers)), index=original_tickers, columns=original_tickers, dtype=float)
    start_dates_h = pd.Series("N/A", index=original_tickers, dtype=str)

    # Attempt to download data only if there are tickers specified
    if original_tickers:
        try:
            # Always fetch a broad range to determine actual earliest dates
            broad_fetch_start_date_str = "1900-01-01"
            raw_dl_prices = yf.download(original_tickers, start=broad_fetch_start_date_str, interval="1mo", auto_adjust=True, progress=False)['Close']
            
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

            # Determine actual earliest data dates from the broad fetch
            actual_earliest_data_dates_map = {}
            if not raw_dl_prices_df.empty:
                for t in original_tickers:
                    if t in raw_dl_prices_df.columns:
                        first_idx = raw_dl_prices_df[t].first_valid_index()
                        if pd.notnull(first_idx):
                            actual_earliest_data_dates_map[t] = first_idx
            
            # Initialize session state for editable_start_dates for new tickers
            # using their actual earliest available date.
            for t in original_tickers:
                if t not in st.session_state.editable_start_dates:
                    actual_start_obj = actual_earliest_data_dates_map.get(t)
                    if actual_start_obj and pd.notnull(actual_start_obj):
                        st.session_state.editable_start_dates[t] = actual_start_obj.strftime('%Y-%m-%d')
                    else:
                        # Fallback if no data found for this ticker in the broad fetch
                        st.session_state.editable_start_dates[t] = "1900-01-01" 

            # Prepare effective start dates for slicing and UI
            effective_slicing_starts_map = {}
            for t in original_tickers:
                # Prefer the value coming from the date_input widget (if present)
                widget_date_obj = st.session_state.get(f"date_input_{t}")
                if widget_date_obj:
                    user_desired_dt_obj = widget_date_obj
                    # Keep editable_start_dates in sync with widget value
                    st.session_state.editable_start_dates[t] = widget_date_obj.strftime('%Y-%m-%d')
                else:
                    user_desired_start_str = st.session_state.editable_start_dates.get(t, "1900-01-01")
                    user_desired_dt_obj = pd.to_datetime(user_desired_start_str).date()

                actual_earliest_dt_obj = actual_earliest_data_dates_map.get(t)
                min_date_for_logic = pd.to_datetime("1900-01-01").date()  # Default earliest if no data
                if actual_earliest_dt_obj and pd.notnull(actual_earliest_dt_obj):
                    min_date_for_logic = actual_earliest_dt_obj.date()

                effective_date_obj = max(user_desired_dt_obj, min_date_for_logic)
                effective_slicing_starts_map[t] = effective_date_obj.strftime('%Y-%m-%d')

            if not raw_dl_prices_df.empty:
                # Filter for columns that are in original_tickers and have actual data
                actual_tickers_with_data = [t for t in original_tickers if t in raw_dl_prices_df.columns and not raw_dl_prices_df[t].isnull().all()]

                if actual_tickers_with_data:
                    # Use .copy() to avoid SettingWithCopyWarning later
                    valid_data = raw_dl_prices_df[actual_tickers_with_data].dropna(how='all').copy()
                    
                    # Slice data for each ticker based on its *effective* start date
                    data_for_returns_calculation_list = []
                    for t_col in valid_data.columns:
                        effective_start_for_ticker_str = effective_slicing_starts_map[t_col]
                        ticker_series_sliced = valid_data[t_col].loc[effective_start_for_ticker_str:]
                        if not ticker_series_sliced.empty:
                            data_for_returns_calculation_list.append(ticker_series_sliced)
                    
                    returns_df = pd.DataFrame(columns=original_tickers)
                    if data_for_returns_calculation_list:
                        final_prices_df = pd.concat(data_for_returns_calculation_list, axis=1, join='outer')
                        final_prices_df.columns = [s.name for s in data_for_returns_calculation_list if not s.empty]
                        returns = final_prices_df.pct_change().dropna(how='all')
                        returns_df = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
                    
                    missing_cols = [tc for tc in original_tickers if tc not in returns_df.columns]
                    for mc in missing_cols:
                        returns_df[mc] = np.nan

                    mu_h = pd.Series(0.0, index=original_tickers, dtype=float)
                    sig_h = pd.Series(0.01, index=original_tickers, dtype=float)
                    start_dates_h = pd.Series("N/A", index=original_tickers, dtype=str)
                    rho_h = pd.DataFrame(np.eye(len(original_tickers)), index=original_tickers, columns=original_tickers, dtype=float)

                    if not returns_df.empty:
                        final_cols_for_stats = [col for col in returns_df.columns if returns_df[col].count() > 1]

                        if final_cols_for_stats:
                            # Annualise statistics derived from monthly returns
                            mu_subset = returns_df[final_cols_for_stats].mean() * 12
                            sig_subset = returns_df[final_cols_for_stats].std() * np.sqrt(12)

                            for col in final_cols_for_stats:
                                mu_h[col] = mu_subset.get(col, 0.0)
                                sig_h[col] = sig_subset.get(col, 0.01)
                                if pd.isnull(sig_h[col]) or sig_h[col] == 0:
                                    sig_h[col] = 0.01
                                first_valid_idx = returns_df[col].first_valid_index()
                                if pd.notnull(first_valid_idx):
                                    start_dates_h[col] = first_valid_idx.strftime('%Y-%m-%d')
                            
                            if len(final_cols_for_stats) >= 1:
                                rho_subset_calculated = returns_df[final_cols_for_stats].corr()
                                for t1 in rho_subset_calculated.index:
                                    for t2 in rho_subset_calculated.columns:
                                        if t1 in rho_h.index and t2 in rho_h.columns:
                                            rho_h.loc[t1, t2] = rho_subset_calculated.loc[t1, t2]
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
    name_header, c1_header, c2_header, c3_header = st.columns(4)
    with name_header:
        st.markdown("**Asset**")
    with c1_header:
        st.markdown("**Return**")
    with c2_header:
        st.markdown("**Volatility**")
    with c3_header:
        st.markdown("**Start date**")

    for t in tickers:
        name_col, c1, c2, c3 = st.columns(4)
        with name_col:
            st.markdown(f"**{t}**")
        with c1:
            mu = st.number_input(f"{t}", value=round(mu_h[t], 4), format="%.4f", key=f"mu_{t}", label_visibility="collapsed")
        with c2:
            vol = st.number_input(f"{t}", value=round(sig_h[t], 4), format="%.4f", key=f"vol_{t}", label_visibility="collapsed")
        with c3:
            # Get user's desired start date from session state
            user_desired_start_str = st.session_state.editable_start_dates.get(t, "1900-01-01")
            user_desired_date_obj = pd.to_datetime(user_desired_start_str).date()

            # Get actual earliest date for this ticker
            actual_earliest_dt_obj = actual_earliest_data_dates_map.get(t)
            min_value_for_widget = pd.to_datetime("1900-01-01").date() # Fallback
            if actual_earliest_dt_obj and pd.notnull(actual_earliest_dt_obj):
                min_value_for_widget = actual_earliest_dt_obj.date()
            
            # Value for widget is the later of user's desire and actual earliest
            value_for_widget = max(user_desired_date_obj, min_value_for_widget)

            new_selected_date_obj = st.date_input(
                label=f"Start Date for {t}", # Label for screen readers, etc.
                value=value_for_widget,
                min_value=min_value_for_widget,
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
    # Header row with asset names
    header_cols = st.columns(n + 1)
    header_cols[0].markdown("**Asset**")
    for idx, tkr in enumerate(tickers):
        header_cols[idx + 1].markdown(f"**{tkr}**")
    display_matrix = []
    for i in range(n):
        row = []
        cols = st.columns(n + 1)
        cols[0].markdown(f"**{tickers[i]}**")
        for j in range(n):
            if j < i:
                cols[j + 1].markdown("‎")
                row.append("–")
            elif i == j:
                cols[j + 1].markdown("**1.0000**")
                row.append("1.0000")
            else:
                default = round(rho_h.iloc[i, j], 4)
                val = cols[j + 1].number_input("", value=default, format="%.4f", key=f"corr_{i}_{j}")
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
