from functools import cached_property
import os
import csv
import numpy as np
import pandas as pd


class StockInfo:

    BASE_DIR = 'data'
    FILE_FOLDER_STOCK = 'dataset-stock'
    FILE_NAME_LASTDAY_CLOSE = 'dataset-lastday_closing_prices.csv'
    FILE_NAME_RETURNS = 'dataset-returns.csv'

    def __init__(self):
        self.file_path_lastday_close = os.path.join(self.BASE_DIR, self.FILE_FOLDER_STOCK, self.FILE_NAME_LASTDAY_CLOSE)
        self.file_path_returns = os.path.join(self.BASE_DIR, self.FILE_FOLDER_STOCK, self.FILE_NAME_RETURNS)

    @cached_property
    def df_returns(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path_returns, index_col='Date')

    @cached_property
    def tickers(self) -> list[str]:
        return self.df_returns.columns.tolist()        

    @cached_property
    def last_prices(self) -> np.ndarray:
        with open(self.file_path_lastday_close) as f:
            prices = []
            reader = csv.reader(f)
            for row in reader:
                prices.append(row)
            return np.array(prices[-1],dtype=float)

    @cached_property
    def mapping_prices(self) -> dict[str, float]:
        return {ticker: float(price) for ticker, price in zip(self.tickers, self.last_prices)}

    @cached_property
    def average_returns(self) -> list[float]:
        return list(self.df_returns.mean(axis=0))

    @cached_property
    def df_covariances(self) -> pd.DataFrame:
        return self.df_returns.cov()

    @cached_property
    def covariances(self) -> list[list[float]]:
        return self.df_covariances.values.tolist()

    def get_covariance_component(self, ticker1: str, ticker2: str) -> float:
        return self.df_covariances.loc[ticker1, ticker2]

    def get_covariance_column(self, ticker: str) -> list[float]:
        return self.df_covariances[[ticker]]