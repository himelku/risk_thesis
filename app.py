# app.py
import streamlit as st
from data_ingestion import DataIngestion
from models import EWMAModel, GARCHModel, VIXModel
from risk_calculation import RiskCalculator
from output_visualization import Visualizer
import datetime


# Set up the Streamlit app title and description
st.title("Modular Risk Assessment Framework")
st.write(
    "This app allows you to assess financial risk (VaR, ES) using different models. "
    "Choose parameters on the sidebar and click **Run Assessment** to see results."
)

# Sidebar inputs for user configuration
st.sidebar.header("Configuration")
# 1. Time horizon selection
horizon = st.sidebar.selectbox("Time Horizon (days)", options=[1, 10, 30], index=0)
# 2. Financial index ticker input (default to S&P 500)
tickers = st.sidebar.multiselect(
    "Select One or More Financial Index Tickers (Yahoo Finance symbol)",
    options=["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^N225", "^HSI"],
    default=["^GSPC"],
)


# 3. Indicator selection (None or VIX for now; could be extended)
indicator_choice = st.sidebar.selectbox(
    "Market Indicator (optional)", options=["None", "VIX"], index=0
)
indicator_ticker = "^VIX" if indicator_choice == "VIX" else None


# 4. Model selection (allow multiple models)
model_options = ["EWMA", "GARCH", "VIX"]
selected_models = st.sidebar.multiselect(
    "Select Risk Model(s)", options=model_options, default=["EWMA"]
)


# 5. Data source selection
data_source = st.sidebar.radio(
    "Data Source", options=["Yahoo Finance", "Bloomberg"], index=0
)


# Trigger the risk assessment when button is clicked
if st.sidebar.button("Run Assessment"):
    # Instantiate the data ingestion module
    data_fetcher = DataIngestion()
    # Fetch data for the main ticker (and indicator if selected)
    try:
        main_data, indicator_data = data_fetcher.fetch_data(
            main_ticker_list=tickers,
            indicator_ticker=indicator_ticker,
            source="yahoo" if data_source == "Yahoo Finance" else "bloomberg",
        )
    except Exception as e:
        st.error(f"Data fetching failed: {e}")
        st.stop()  # stop the app if data fetch failed

    # Inform the user about the data range fetched
    if not main_data.empty:
        st.write(
            f"Fetched **{tickers}** data from {main_data.index[0].date()} to {main_data.index[-1].date()} "
            f"using {data_source}."
        )
        if indicator_ticker and indicator_data is not None and not indicator_data.empty:
            st.write(f"Fetched **{indicator_choice}** data for the same period.")

    # Limit to last 10 years
    end_date = main_data.index[-1]
    start_date = end_date - datetime.timedelta(days=365 * 10)

    main_data = main_data.loc[start_date:end_date]

    if indicator_data is not None:
        indicator_data = indicator_data.loc[start_date:end_date]

    # Initialize selected models with the data
    models = []
    for model_name in selected_models:
        # Create an instance of each selected model
        if model_name == "EWMA":
            models.append(
                (
                    model_name,
                    EWMAModel(price_data=main_data, indicator_data=indicator_data),
                )
            )
        elif model_name == "GARCH":
            models.append(
                (
                    model_name,
                    GARCHModel(price_data=main_data, indicator_data=indicator_data),
                )
            )
        elif model_name == "VIX":
            models.append(
                (
                    model_name,
                    VIXModel(price_data=main_data, indicator_data=indicator_data),
                )
            )

    # Compute risk metrics for each model
    risk_calc = RiskCalculator(confidence_level=0.95)  # 95% confidence VaR/ES
    results = {}  # to store metrics for each model
    for model_name, model in models:
        # Run the model forecast for the selected horizon
        model_output = model.forecast(horizon=horizon)
        # Compute VaR and ES based on model output
        metrics = risk_calc.compute_risk_metrics(model_name, model_output)
        results[model_name] = metrics

    # Visualization and output
    vis = Visualizer()
    # Section: Historical Data Plot
    st.subheader("Historical Data")
    fig_price, fig_indicator = vis.plot_price_data(main_data, indicator_data)
    st.plotly_chart(fig_price, use_container_width=True)
    if fig_indicator:
        st.plotly_chart(fig_indicator, use_container_width=True)
    # Section: Risk Metrics Table
    st.subheader(f"Risk Metrics (VaR and ES at 95% confidence, {horizon}-day)")
    metrics_table = vis.display_metrics_table(results)
    st.write(metrics_table)  # display as a table
    # Section: VaR Comparison Bar Chart
    st.subheader("Value at Risk Comparison")
    fig_var = vis.plot_var(results, confidence_level=0.95)
    st.plotly_chart(fig_var, use_container_width=True)
    # Section: Interpretation Text
    st.subheader("Model Interpretations")
    for model_name, metrics in results.items():
        interpretation = vis.generate_interpretation(
            model_name, metrics, horizon=horizon, confidence=0.95
        )
        st.write(interpretation)
    # Section: Export Results
    st.subheader("Export Results")
    csv_file = vis.export_results(results, filename="risk_metrics.csv")
    st.download_button(
        label="Download CSV of Results",
        data=open(csv_file, "rb").read(),
        file_name=csv_file,
        mime="text/csv",
    )
