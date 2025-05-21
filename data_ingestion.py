import yfinance as yf
import pandas as pd


class DataIngestion:
    def __init__(self):
        pass

    def _flatten_columns(self, df):
        """Flatten MultiIndex columns if present."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(col).strip() for col in df.columns.values]
        return df

    def fetch_data(
        self,
        main_ticker_list,
        indicator_ticker=None,
        start_date=None,
        end_date=None,
        source="yahoo",
    ):
        if source.lower() == "yahoo":
            # Join tickers if list
            tickers = (
                main_ticker_list
                if isinstance(main_ticker_list, list)
                else [main_ticker_list]
            )
            main_df = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                group_by="ticker",
            )

            if isinstance(main_df.columns, pd.MultiIndex):
                main_df.columns = [
                    " ".join(map(str, col)).strip() for col in main_df.columns.values
                ]

            # Indicator fetch
            if indicator_ticker:
                indicator_df = yf.download(
                    indicator_ticker, start=start_date, end=end_date, auto_adjust=False
                )
                if isinstance(indicator_df.columns, pd.MultiIndex):
                    indicator_df.columns = [
                        " ".join(map(str, col)).strip()
                        for col in indicator_df.columns.values
                    ]
            else:
                indicator_df = None

            if main_df.empty:
                raise ValueError(
                    f"No data found for tickers {tickers} from Yahoo Finance."
                )
            if indicator_ticker and indicator_df is not None and indicator_df.empty:
                raise ValueError(
                    f"No data found for indicator {indicator_ticker} from Yahoo Finance."
                )

            return (main_df, indicator_df) if indicator_ticker else (main_df, None)
