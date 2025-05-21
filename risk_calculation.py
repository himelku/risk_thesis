# -------------------------
# File: risk_calculation.py
# -------------------------
import numpy as np
from scipy.stats import norm


class RiskCalculator:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compute_risk_metrics(
        self,
        model_name,
        model_output,
        returns=None,
        price_series=None,
        risk_free_rate=0.0,
    ):
        """
        Compute primary risk metrics (VaR and ES) and additional metrics if return/price data is provided.
        """
        result = {}

        # Compute VaR and ES based on model output
        if isinstance(model_output, dict):
            if "volatility" in model_output:
                sigma = float(model_output["volatility"])
                z = norm.ppf(self.alpha)
                result["VaR"] = -z * sigma
                result["ES"] = sigma * (norm.pdf(z) / self.alpha)
            elif "distribution" in model_output:
                returns_dist = np.array(model_output["distribution"])
                if returns_dist.size == 0:
                    raise ValueError(
                        "Distribution is empty. Cannot compute risk metrics."
                    )
                loss_dist = -returns_dist
                VaR = np.quantile(loss_dist, 1 - self.confidence_level)
                tail_losses = loss_dist[loss_dist >= VaR]
                ES = tail_losses.mean() if tail_losses.size > 0 else VaR
                result["VaR"] = VaR
                result["ES"] = ES
            else:
                raise ValueError(
                    f"Unrecognized model output keys: {model_output.keys()}"
                )
        elif isinstance(model_output, (int, float, np.floating)):
            sigma = float(model_output)
            z = norm.ppf(self.alpha)
            result["VaR"] = -z * sigma
            result["ES"] = sigma * (norm.pdf(z) / self.alpha)
        else:
            raise ValueError("Unsupported model output type for risk calculation.")

        # Additional metrics if data is provided
        if returns is not None:
            result["Volatility"] = self.compute_realized_volatility(returns)
            result["Sharpe Ratio"] = self.compute_sharpe_ratio(returns, risk_free_rate)
        if price_series is not None:
            result["Max Drawdown"] = self.compute_max_drawdown(price_series)

        return result

    def compute_realized_volatility(self, returns):
        """Compute standard deviation of returns."""
        return np.std(returns)

    def compute_max_drawdown(self, price_series):
        """Compute the maximum drawdown from a price series."""
        cumulative = price_series / price_series.iloc[0]
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def compute_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Compute the Sharpe Ratio given returns and optional risk-free rate."""
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std()
