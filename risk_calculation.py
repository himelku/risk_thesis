# risk_calculation.py
import numpy as np
from scipy.stats import norm


class RiskCalculator:
    """
    Computes risk measures like Value at Risk (VaR) and Expected Shortfall (ES).
    Adaptable to different types of model outputs (volatility estimates, return distributions, etc.).
    """

    def __init__(self, confidence_level=0.95):
        """
        Initialize the RiskCalculator with a given confidence level.
        confidence_level: e.g., 0.95 for 95% confidence (5% VaR).
        """
        self.confidence_level = confidence_level
        # alpha is the tail probability (e.g., 0.05 for 95% confidence)
        self.alpha = 1 - confidence_level

    def compute_risk_metrics(self, model_name, model_output):
        """
        Compute VaR and ES for a given model output.
        model_name: Name of the model (string) - not strictly used in calculation, but can be used for context or logging.
        model_output: Could be a dictionary with keys like 'volatility' or 'distribution', or a numeric value (volatility).

        Returns a dict with keys 'VaR' and 'ES' representing Value at Risk and Expected Shortfall, respectively.
        """
        # Determine how to interpret model_output
        if isinstance(model_output, dict):
            # If the model output is a dictionary, check for known keys
            if "volatility" in model_output:
                # Treat 'volatility' as the standard deviation of returns
                sigma = float(model_output["volatility"])
                # Compute parametric VaR assuming normal distribution, mean = 0
                # VaR is the α-quantile of returns (as a negative number for loss), we report positive loss:contentReference[oaicite:12]{index=12}.
                z = norm.ppf(self.alpha)  # z quantile (will be negative for alpha=0.05)
                VaR = -z * sigma  # invert sign to get positive VaR (loss) value
                # Compute parametric ES (for normal, ES = sigma * (phi(z) / alpha), where phi is PDF at z):contentReference[oaicite:13]{index=13}.
                ES = sigma * (norm.pdf(z) / self.alpha)
                return {"VaR": VaR, "ES": ES}
            elif "distribution" in model_output:
                # If a distribution of returns is provided (e.g., from simulation), calculate empirical VaR/ES
                returns_dist = np.array(model_output["distribution"])
                if returns_dist.size == 0:
                    raise ValueError(
                        "Distribution is empty. Cannot compute risk metrics."
                    )
                # Compute VaR as the alpha-quantile of the distribution of losses
                # If returns_dist are actual returns, losses = -returns. We find the quantile for losses.
                loss_dist = -returns_dist  # convert returns to losses
                VaR = np.quantile(
                    loss_dist, 1 - self.confidence_level
                )  # the (1-confidence) quantile of losses
                # Compute ES as the mean loss beyond VaR (i.e., conditional loss average in tail)
                tail_losses = loss_dist[
                    loss_dist >= VaR
                ]  # losses that are beyond (>=) VaR threshold
                ES = tail_losses.mean() if tail_losses.size > 0 else VaR
                return {"VaR": VaR, "ES": ES}
            else:
                raise ValueError(
                    f"Unrecognized model output keys: {model_output.keys()}"
                )
        elif isinstance(model_output, (int, float, np.floating)):
            # If a raw number is given, assume it's a volatility (std dev) for returns
            sigma = float(model_output)
            z = norm.ppf(self.alpha)
            VaR = -z * sigma
            ES = sigma * (norm.pdf(z) / self.alpha)
            return {"VaR": VaR, "ES": ES}
        else:
            raise ValueError("Unsupported model output type for risk calculation.")

    # (Optional) Additional methods for other risk metrics could be added here (e.g., volatility, drawdown, etc.)
