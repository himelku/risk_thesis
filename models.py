# models.py
import numpy as np


# Base class for risk models
class RiskModelBase:
    """
    Base class for risk models. It prepares the price and return data and provides a template for the forecast method.
    New models should inherit from this and implement the forecast() method.
    """

    def __init__(self, price_data, indicator_data=None):
        # Store price data (pandas DataFrame expected) and optional indicator data
        self.price_data = price_data
        self.indicator_data = indicator_data
        # Extract the adjusted closing prices for analysis (if available, else use the first column)
        if hasattr(price_data, "columns"):
            if "Adj Close" in price_data.columns:
                self.price_series = price_data["Adj Close"]
            else:
                # if DataFrame but no 'Adj Close', use the last column as a proxy for price
                self.price_series = price_data.iloc[:, -1]
        else:
            # If price_data is passed as a Series
            self.price_series = price_data
        # Compute daily returns from price series for volatility modeling
        # We'll use percentage returns (log returns could be used as well for small differences)
        self.returns = self.price_series.pct_change().dropna()
        # Prepare indicator series if provided (e.g., VIX values)
        if indicator_data is not None:
            if (
                hasattr(indicator_data, "columns")
                and "Adj Close" in indicator_data.columns
            ):
                self.indicator_series = indicator_data["Adj Close"]
            else:
                self.indicator_series = indicator_data  # assume Series if not DataFrame
        else:
            self.indicator_series = None

    def forecast(self, horizon=1):
        """
        To be implemented by subclasses. Should return a result (e.g., volatility forecast, distribution, etc.)
        for the given time horizon (in days).
        """
        raise NotImplementedError("Subclasses should implement this method.")


class EWMAModel(RiskModelBase):
    """
    Exponentially Weighted Moving Average model for volatility forecasting.
    Uses a decay factor lambda (default 0.94 as in J.P. Morgan's RiskMetrics).
    Forecasts volatility based on historical returns, giving more weight to recent data.
    """

    def __init__(self, price_data, indicator_data=None, decay_factor=0.94):
        super().__init__(price_data, indicator_data)
        self.lambda_ = decay_factor  # decay factor (λ). RiskMetrics often uses 0.94:contentReference[oaicite:2]{index=2}.

    def forecast(self, horizon=1):
        """
        Compute the forecasted volatility (standard deviation of returns) for the given horizon using EWMA.
        If horizon > 1, scale the 1-day volatility by sqrt(horizon) (assuming independent days for simplicity).
        Returns a dictionary with the volatility forecast.
        """
        # Initialize variance with the first return's square (as a starting point for recursion)
        returns_array = self.returns.values  # numpy array of return values
        if len(returns_array) == 0:
            raise ValueError("Not enough data to compute EWMA volatility.")
        var_ewma = (
            returns_array[0] ** 2
        )  # start with first observation's squared return
        # Recursively apply EWMA formula for variance:
        # var_t = λ * var_{t-1} + (1-λ) * (return_{t-1})^2 for each return in sequence.
        for r in returns_array[1:]:
            var_ewma = self.lambda_ * var_ewma + (1 - self.lambda_) * (r**2)
        # var_ewma now holds the variance estimate for the most recent day
        sigma_1day = np.sqrt(var_ewma)  # 1-day ahead volatility forecast (std dev)
        # If a longer horizon is requested, assume i.i.d. and scale volatility by sqrt(horizon)
        sigma_horizon = sigma_1day * np.sqrt(horizon)
        return {"volatility": sigma_horizon}


class GARCHModel(RiskModelBase):
    """
    GARCH(1,1) model for volatility forecasting.
    In an actual implementation, we would fit a GARCH model to the return series and forecast future volatility.
    Here, we provide a structure and a placeholder for integration with a GARCH library (e.g., 'arch').
    """

    def __init__(self, price_data, indicator_data=None):
        super().__init__(price_data, indicator_data)
        # Additional parameters for GARCH can be added here (p, q orders, etc.)
        # For simplicity, we'll use a standard GARCH(1,1) assumption.

    def forecast(self, horizon=1):
        """
        Forecast volatility using a GARCH(1,1) model. If a GARCH library is available, use it to fit and forecast.
        Otherwise, use a placeholder approach (e.g., sample standard deviation) for demonstration.
        Returns a dictionary with the forecast volatility.
        """
        # Placeholder for actual GARCH fitting:
        # In practice, one would use a library like 'arch' to fit a GARCH(1,1) model to self.returns.
        # Example (if the 'arch' package is installed):
        # from arch import arch_model
        # am = arch_model(self.returns * 100, vol='Garch', p=1, q=1, mean='constant', dist='normal')
        # res = am.fit(disp='off')
        # forecast = res.forecast(horizon=horizon)
        # predicted_var = forecast.variance.values[-1, -1]  # variance for last period of horizon
        # sigma_1day = np.sqrt(predicted_var) / 100  # convert back to percentage if we scaled returns
        # sigma_horizon = sigma_1day * np.sqrt(horizon)
        # return {'volatility': sigma_horizon}
        #
        # Since we cannot perform actual GARCH fitting here, we'll use a simple fallback:
        sigma_1day = np.std(
            self.returns
        )  # sample standard deviation as a crude estimate
        sigma_horizon = sigma_1day * np.sqrt(horizon)
        return {"volatility": sigma_horizon}


class VIXModel(RiskModelBase):
    """
    VIX-based model for risk forecasting.
    Uses the VIX index (implied 30-day volatility for S&P 500) as an indicator of expected volatility.
    The model derives a volatility forecast from the latest VIX value.
    Note: VIX is an annualized 30-day volatility percentage:contentReference[oaicite:3]{index=3}, so conversion to daily volatility is required.
    """

    def __init__(self, price_data, indicator_data=None):
        super().__init__(price_data, indicator_data)
        # Ensure that VIX data is provided
        if self.indicator_series is None:
            raise ValueError(
                "VIXModel requires an indicator series (VIX) for volatility input."
            )

    def forecast(self, horizon=1):
        """
        Compute volatility forecast using the latest VIX value.
        - VIX is the expected annualized volatility (% per year) over the next 30 days for the S&P 500.
        - We convert VIX to a daily volatility (stdev of daily returns) assuming ~252 trading days in a year.
        - For multi-day horizon, we scale the daily volatility by sqrt(horizon).
        Returns a dictionary with the forecast volatility.
        """
        # Get the most recent VIX value (as a percentage, e.g., 20 for 20%)
        latest_vix = float(self.indicator_series.iloc[-1])
        # Convert VIX percentage to decimal (e.g., 20 -> 0.20) and compute daily volatility:
        # VIX is 30-day annualized vol, so first convert to annual vol in decimal, then to daily.
        annual_vol = latest_vix / 100.0  # 20 -> 0.20 (annual vol in decimal)
        daily_vol = annual_vol / np.sqrt(
            252
        )  # e.g., 0.20/sqrt(252) ≈ 0.0126 (1.26% daily stdev):contentReference[oaicite:4]{index=4}
        # If horizon is multiple days, scale by sqrt(horizon) (assuming independent daily moves for simplicity)
        sigma_horizon = daily_vol * np.sqrt(horizon)
        return {"volatility": sigma_horizon}
