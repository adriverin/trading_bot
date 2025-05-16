# This module defines an interface and a base class for accessing crypto price datasets.
# The `CryptoPriceDatasetAdapter` class serves as an abstract base class that outlines
# the structure for any data source of crypto prices, allowing for multiple implementations
# to support different sources. It includes properties for obtaining training and validation
# datasets for specified crypto pairs, which are expected to return pandas DataFrames 
# containing date and price information. The `BaseCryptoPriceDatasetAdapter` class 
# extends this interface, providing a foundation for specific dataset adapters, 
# including initialization and a method for connecting to and preparing the data.

from abc import ABC, ABCMeta, abstractmethod

import binance.client as Client
import numpy as np
import pandas as pd
import enum
from datetime import datetime

from myimports.plotting import plot_matplotlib, plot_plotly

class CryptoPriceDatasetAdapter(metaclass=ABCMeta):
    """
    Interface to access any data source of crypto prices.
    Multiple implementations can be made to support different data sources.
    """

    DEFAULT_TICKER = "BTCUSDT"  # Changed from "BTC/USDT" to "BTCUSDT"
    DEFAULT_PRICE_USED = "high" # or "low" or "open" or "close"

    @property
    @abstractmethod
    def training_set(self, ticker=None): ...
    """
    Property to get training dataset for a given crypto pair.
    This dataset can be used to train the model. Although there
    are no such restrictions on using it.

    Args:
        ticker (str): The ticker of the crypto pair to get the training dataset for.

    Returns:
        pd.DataFrame: two columns: date and price.
    """

    @property
    @abstractmethod
    def validation_set(self, ticker=None): ...
    """
    Property to get validation dataset for a given crypto pair.
    This dataset can be used to validate the model. Although there
    are no such restrictions on using it.

    Args:
        ticker (str): The ticker of the crypto pair to get the validation dataset for.

        Returns:
            pd.DataFrame: two columns: date and price.
    """


class BaseCryptoPriceDatasetAdapter(CryptoPriceDatasetAdapter, ABC):
    """
    Base class for all crypto price dataset adapters.
    """

    def __init__(self, ticker:str=None):
        self._ticker = ticker
        self._training_set = None
        self._validation_set = None
        self._returns = None    
        self._log_returns = None

    @abstractmethod
    def _connect_and_prepare(self, date_range:tuple): ...
    """
    Abstract method to connect to the data source and prepare the data.
    """

    @property
    def training_set(self):
        return self._training_set.copy()
    
    @property
    def validation_set(self):
        return self._validation_set.copy()

    @property
    def returns(self):
        return self._returns.copy()

    @property
    def log_returns(self):
        return self._log_returns.copy()


class Frequency(enum.Enum):
    SECOND = "1s"
    MINUTE = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"

class BinanceCryptoPriceDatasetAdapter(BaseCryptoPriceDatasetAdapter):
    """
    Adapter for Binance crypto price data; Uses the Binance API.
    """

    def __init__(self, ticker:str=CryptoPriceDatasetAdapter.DEFAULT_TICKER, frequency=Frequency.DAILY,
                 training_set_date_range=('2023-06-30','2024-06-30'), validation_set_date_range=('2020-01-01','2021-06-29'),
                 price_used:str=CryptoPriceDatasetAdapter.DEFAULT_PRICE_USED):
        super().__init__(ticker)
        self._frequency = frequency
        self._binance = Client.Client()  # Initialize the Binance client without a specific ticker
        self._price_used = price_used  
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(validation_set_date_range)
        self._returns = self._compute_returns(training_set_date_range)
        self._log_returns = self._compute_log_returns(training_set_date_range)


    def _connect_and_prepare(self, date_range:tuple):
        crypto_price_records = None
        records = self._binance.get_historical_klines(self._ticker, self._frequency.value, date_range[0], date_range[1])
        crypto_price_records = pd.DataFrame(records, 
                                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        if self._price_used not in {"open", "high", "low", "close"}: # Validate the price_used value
            raise ValueError(f"Invalid price_used value: {self._price_used}")

        # Convert timestamp to datetime
        crypto_price_records['timestamp'] = pd.to_datetime(crypto_price_records['timestamp'], unit='ms')

        crypto_price_records[self._price_used] = pd.to_numeric(crypto_price_records[self._price_used], errors='coerce')

        crypto_price_records = crypto_price_records[['timestamp', self._price_used]]  
        crypto_price_records.rename(columns={"timestamp": "time", self._price_used: "price"}, inplace=True)

        return crypto_price_records


    def _compute_returns(self, date_range:tuple):
        """
        Compute the returns for a given ticker and time interval.

        Args:
            ticker (str): The ticker of the crypto pair to compute the returns for.
            date_range (tuple): The date range of the data to compute the returns for.
            time_interval (str): The time interval to compute the returns for.

        Returns:
            pd.DataFrame: The returns for the given ticker and given time intervals. 
            Default is [1d, 1w, 1M]
        """

        # period_prices = BinanceCryptoPriceDatasetAdapter(ticker, time_interval, training_set_date_range=date_range).training_set
        period_prices = self._connect_and_prepare(date_range)

        period_returns = pd.DataFrame(columns=['time', f'returns {self._ticker}'])
        period_returns['time'] = period_prices['time']  
        period_returns[f'returns {self._ticker}'] = period_prices['price']/period_prices['price'].shift(1) - 1
        period_returns.dropna(inplace=True)


        return period_returns
    

    def _compute_log_returns(self, date_range:tuple):
        """
        Compute the log returns for a given ticker and time interval.

        Args:
            ticker (str): The ticker of the crypto pair to compute the returns for.
            date_range (tuple): The date range of the data to compute the returns for.
            time_interval (str): The time interval to compute the returns for.

        Returns:
            pd.DataFrame: The returns for the given ticker and given time intervals. 
            Default is [1d, 1w, 1M]
        """

        # period_prices = BinanceCryptoPriceDatasetAdapter(ticker, time_interval, training_set_date_range=date_range).training_set
        period_prices = self._connect_and_prepare(date_range)

        period_log_returns = pd.DataFrame(columns=['time', f'log returns {self._ticker}'])
        period_log_returns['time'] = period_prices['time']  
        period_log_returns[f'log returns {self._ticker}'] = np.log(period_prices['price']/period_prices['price'].shift(1))
        period_log_returns.dropna(inplace=True)


        return period_log_returns






# # Testing the BinanceCryptoPriceDatasetAdapter
# if __name__ == "__main__":
#     # Get today's date
#     today = datetime.now().strftime("%Y-%m-%d")

#     adapter = BinanceCryptoPriceDatasetAdapter(ticker="BTCUSDT", training_set_date_range=('2025-01-01', today), frequency=Frequency.HOURLY)
#     print(adapter.training_set)  # Print the first few rows of the training set

#     print(adapter.returns)

#     plot_matplotlib(adapter.returns, "BTCUSDT")
#     plot_matplotlib(adapter.log_returns, "BTCUSDT")