# -------------------------------
# File: output_visualization.py
# -------------------------------
import pandas as pd
import plotly.express as px


class Visualizer:
    def plot_price_data(self, price_df, indicator_df=None):
        price_df = price_df.reset_index()
        # Filter for columns that include 'Adj Close' or 'Close'
        price_cols = [
            col for col in price_df.columns if "Adj Close" in col or "Close" in col
        ]
        if not price_cols:
            raise ValueError(
                f"No 'Adj Close' or 'Close' columns found. Got: {price_df.columns.tolist()}"
            )

        fig_price = px.line(
            price_df,
            x="Date",
            y=price_cols,
            title="Historical Price for Selected Indexes",
        )
        fig_price.update_layout(yaxis_title="Price")

        fig_indicator = None
        if indicator_df is not None:
            indicator_df = indicator_df.reset_index()
            ind_col = next(
                (
                    col
                    for col in indicator_df.columns
                    if "Adj Close" in col or "Close" in col
                ),
                None,
            )
            if not ind_col:
                raise ValueError(
                    f"No valid column found in indicator data: {indicator_df.columns.tolist()}"
                )

            fig_indicator = px.line(
                indicator_df, x="Date", y=ind_col, title="Indicator History"
            )
            fig_indicator.update_layout(yaxis_title="Indicator Value")

        return fig_price, fig_indicator

    def plot_var(self, var_results, confidence_level=0.95):
        df_metrics = pd.DataFrame(var_results).T
        df_metrics.index.name = "Model"

        fig = px.bar(
            df_metrics,
            x=df_metrics.index,
            y="VaR",
            title=f"VaR by Model ({int(confidence_level*100)}% confidence)",
            labels={"VaR": f"VaR ({int(confidence_level*100)}% conf)"},
        )
        fig.update_layout(yaxis_title="Value at Risk")
        return fig

    def display_metrics_table(self, var_results):
        df_metrics = pd.DataFrame(var_results).T
        df_metrics.index.name = "Model"
        return df_metrics

    def generate_interpretation(self, model_name, metrics, horizon=1, confidence=0.95):
        VaR_value = metrics["VaR"] * 100
        ES_value = metrics["ES"] * 100
        alpha_pct = int((1 - confidence) * 100)

        return (
            f"For model **{model_name}**: the estimated **{int(confidence*100)}% {horizon}-day VaR** is "
            f"**{VaR_value:.2f}%**, meaning there's a {alpha_pct}% chance of exceeding this loss. "
            f"The **Expected Shortfall** is approximately **{ES_value:.2f}%**."
        )

    def export_results(self, var_results, filename="risk_metrics.csv"):
        df_metrics = pd.DataFrame(var_results).T
        df_metrics.index.name = "Model"
        df_metrics.to_csv(filename)
        return filename
