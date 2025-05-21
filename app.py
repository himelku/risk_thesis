# ----------------------------
# File: app.py
# ----------------------------
import streamlit as st
import datetime
import pandas as pd
import plotly.express as px

from data_ingestion import DataIngestion
from models import EWMAModel, GARCHModel, VIXModel
from risk_calculation import RiskCalculator
from output_visualization import Visualizer

st.title("Modular Risk Assessment Framework")
st.write(
    "This app allows you to assess financial risk (VaR, ES) using different models. Choose parameters on the sidebar and click **Run Assessment** to see results."
)

st.sidebar.header("Configuration")
horizon = st.sidebar.selectbox("Time Horizon (days)", options=[1, 10, 30], index=0)
tickers = st.sidebar.multiselect(
    "Select One or More Financial Index Tickers (Yahoo Finance symbol)",
    options=["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^N225", "^HSI"],
    default=["^GSPC"],
)
ticker_weights = {}
total_weight = 0.0
for ticker in tickers:
    weight = st.sidebar.slider(
        f"Weight for {ticker}", min_value=0.0, max_value=1.0, value=1.0 / len(tickers)
    )
    ticker_weights[ticker] = weight
    total_weight += weight
if abs(total_weight - 1.0) > 0.01:
    st.sidebar.error("The sum of weights must be approximately 1.0")
    st.stop()

indicator_choice = st.sidebar.selectbox(
    "Market Indicator (optional)", options=["None", "VIX"], index=0
)
indicator_ticker = "^VIX" if indicator_choice == "VIX" else None
model_options = ["EWMA", "GARCH(1,1)", "GARCH(1,2)", "GJR-GARCH", "EGARCH"]
selected_models = st.sidebar.multiselect(
    "Select Risk Model(s)", options=model_options, default=["EWMA"]
)
data_source = st.sidebar.radio(
    "Data Source", options=["Yahoo Finance", "Bloomberg"], index=0
)

if st.sidebar.button("Run Assessment"):
    data_fetcher = DataIngestion()
    try:
        main_data, indicator_data = data_fetcher.fetch_data(
            main_ticker_list=tickers,
            indicator_ticker=indicator_ticker,
            source="yahoo" if data_source == "Yahoo Finance" else "bloomberg",
        )
    except Exception as e:
        st.error(f"Data fetching failed: {e}")
        st.stop()

    if not main_data.empty:
        st.write(
            f"Fetched **{tickers}** data from {main_data.index[0].date()} to {main_data.index[-1].date()} using {data_source}."
        )
        if indicator_ticker and indicator_data is not None and not indicator_data.empty:
            st.write(f"Fetched **{indicator_choice}** data for the same period.")

    end_date = main_data.index[-1]
    start_date = end_date - datetime.timedelta(days=365 * 10)
    main_data = main_data.loc[start_date:end_date]
    if indicator_data is not None:
        indicator_data = indicator_data.loc[start_date:end_date]

    # Portfolio-weighted price and returns
    adj_close_cols = [
        col for col in main_data.columns if "Adj Close" in col or "Close" in col
    ]
    weighted_prices = pd.DataFrame(index=main_data.index)
    for ticker in tickers:
        col = next((c for c in adj_close_cols if ticker in c), None)
        if col:
            weighted_prices[ticker] = main_data[col] * ticker_weights[ticker]
    portfolio_price = weighted_prices.sum(axis=1)
    returns = portfolio_price.pct_change().dropna()
    risk_calc = RiskCalculator(confidence_level=0.95)

    st.subheader("Portfolio-Level Risk Metrics")
    st.write("Computed from the weighted combination of selected indices.")
    asset_metrics = {
        "Volatility": risk_calc.compute_realized_volatility(returns),
        "Max Drawdown": risk_calc.compute_max_drawdown(portfolio_price),
        "Sharpe Ratio": risk_calc.compute_sharpe_ratio(returns, risk_free_rate=0.0),
    }
    asset_df = pd.DataFrame(asset_metrics, index=["Metric"]).T
    asset_df.columns = ["Value"]
    asset_df["Value"] = asset_df["Value"].apply(
        lambda x: (
            f"{x * 100:.2f}%" if isinstance(x, float) and abs(x) < 10 else f"{x:.2f}"
        )
    )
    st.table(asset_df)

    models = []
    for model_name in selected_models:
        if model_name == "EWMA":
            models.append((model_name, EWMAModel(price_data=portfolio_price)))
        elif model_name == "GARCH(1,1)":
            models.append(
                (model_name, GARCHModel(price_data=portfolio_price, p=1, q=1))
            )
        elif model_name == "GARCH(1,2)":
            models.append(
                (model_name, GARCHModel(price_data=portfolio_price, p=1, q=2))
            )
        elif model_name == "GJR-GARCH":
            models.append(
                (
                    model_name,
                    GARCHModel(
                        price_data=portfolio_price, model_type="GARCH", p=1, q=1
                    ),
                )
            )
        elif model_name == "EGARCH":
            models.append(
                (
                    model_name,
                    GARCHModel(
                        price_data=portfolio_price, model_type="EGARCH", p=1, q=1
                    ),
                )
            )

    results = {}
    rolling_var_df = pd.DataFrame(index=returns.index)
    rolling_vol_df = pd.DataFrame(index=returns.index)

    for model_name, model in models:
        model_output = model.forecast(horizon=horizon)
        model_returns = getattr(model, "returns", None)
        model_prices = getattr(model, "price_series", None)
        metrics = risk_calc.compute_risk_metrics(
            model_name=model_name,
            model_output=model_output,
            returns=model_returns,
            price_series=model_prices,
            risk_free_rate=0.0,
        )
        results[model_name] = metrics

        if model_returns is not None:
            rolling_vol_df[model_name] = model_returns.rolling(window=horizon).std()
            rolling_var_df[model_name] = (
                model_returns.rolling(window=horizon).std() * -1.65
            )

    vis = Visualizer()

    st.subheader(f"Risk Metrics (VaR and ES at 95% confidence, {horizon}-day)")
    metrics_table = vis.display_metrics_table(results)
    st.write(metrics_table)

    st.subheader("Volatility Forecasts Comparison")
    vol_data = {
        model: results[model]["Volatility"]
        for model in results
        if "Volatility" in results[model]
    }
    vol_df = pd.DataFrame.from_dict(vol_data, orient="index", columns=["Volatility"])
    vol_df["Volatility"] = vol_df["Volatility"].astype(float) * 100
    fig = px.bar(
        vol_df, x=vol_df.index, y="Volatility", title="Forecasted Volatility by Model"
    )
    fig.update_layout(yaxis_title="Volatility (%)")
    st.plotly_chart(fig)

    st.subheader("Rolling Volatility")
    st.line_chart(rolling_vol_df * 100)

    st.subheader("Rolling Value at Risk (VaR)")
    st.line_chart(rolling_var_df * 100)

    st.subheader("Value at Risk Comparison")
    fig_var = vis.plot_var(results, confidence_level=0.95)
    st.plotly_chart(fig_var, use_container_width=True)

    st.subheader("Model Interpretations")
    for model_name, metrics in results.items():
        interpretation = vis.generate_interpretation(
            model_name, metrics, horizon=horizon, confidence=0.95
        )
        st.write(interpretation)

    st.subheader("Cumulative Return")
    cumulative_returns = (1 + returns).cumprod()
    st.line_chart(cumulative_returns * 100)

    st.subheader("Cumulative Drawdown")
    rolling_max = portfolio_price.cummax()
    drawdown = (portfolio_price - rolling_max) / rolling_max

    import plotly.graph_objects as go

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown * 100, mode="lines", name="Drawdown")
    )

    # Highlight periods below -20%
    threshold = -0.2  # -20%
    severe_drawdowns = drawdown[drawdown < threshold]
    if not severe_drawdowns.empty:
        fig_dd.add_trace(
            go.Scatter(
                x=severe_drawdowns.index,
                y=severe_drawdowns * 100,
                mode="markers",
                marker=dict(color="red", size=4),
                name="Below -20%",
            )
        )
    fig_dd.update_layout(
        title="Cumulative Drawdown with Highlighted -20% Threshold",
        yaxis_title="Drawdown (%)",
    )
    st.plotly_chart(fig_dd)

    # Annotated drawdown threshold
    st.markdown("### Drawdown Alert Threshold")
    if not severe_drawdowns.empty:
        st.warning(f"⚠️ {len(severe_drawdowns)} days where drawdown exceeded -20%")

    # Time-under-water / recovery period
    st.markdown("### Recovery Period (Time Under Water)")
    underwater = drawdown < 0
    recovery_periods = []
    current_duration = 0
    for val in underwater:
        if val:
            current_duration += 1
        elif current_duration > 0:
            recovery_periods.append(current_duration)
            current_duration = 0
    if current_duration > 0:
        recovery_periods.append(current_duration)

    if recovery_periods:
        max_recovery = max(recovery_periods)
        avg_recovery = sum(recovery_periods) / len(recovery_periods)
        st.success(f"📉 Longest recovery period: {max_recovery} days")
        st.info(f"📊 Average recovery period: {avg_recovery:.2f} days")
    else:
        st.info("✅ No underwater periods detected.")

    st.subheader("Export Results")
    csv_file = vis.export_results(results, filename="risk_metrics.csv")
    st.download_button(
        label="Download CSV of Results",
        data=open(csv_file, "rb").read(),
        file_name=csv_file,
        mime="text/csv",
    )

    st.download_button(
        label="Download Rolling Volatility CSV",
        data=rolling_vol_df.to_csv().encode(),
        file_name="rolling_volatility.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Rolling VaR CSV",
        data=rolling_var_df.to_csv().encode(),
        file_name="rolling_var.csv",
        mime="text/csv",
    )
