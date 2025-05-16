# import ccxt
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from datetime import datetime 

import myimports.classes_financial_data as fdata


#print(ccxt.exchanges)

# binance = ccxt.binance()

portfolio = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "VETUSDT", "SCRTUSDT", "ATOMUSDT", "AVAXUSDT", "LRCUSDT", "DOTUSDT", "PEPEUSDT", "TURBOUSDT"]

portfolioAmount = [0.24974057, #btc
7.29294085, #eth
7168.973722, #ada
21959.5, #vet
4201.611758, #scrt
268.6115158, #atom
190.8155594, #avax
4079.201635, #lrc
375.6367995, #dot
648143010, #pepe
159913.6851] #turbo 

portfolio_df = pd.DataFrame()

portfolioValue = pd.DataFrame()



# for pair in portfolio:
#     # Fetch the training set data for the current pair
#     data = fdata.BinanceCryptoPriceDatasetAdapter(ticker=pair, frequency=fdata.Frequency.DAILY, training_set_date_range=('2020-01-01', '2025-05-09')).training_set
    
#     # If portfolio_df is empty, initialize it with the current data
#     if portfolio_df.empty:
#         portfolio_df = data
#     else:
#         # Join the current data with the existing portfolio_df
#         portfolio_df = portfolio_df.join(data, how='outer')

today = datetime.now().strftime("%Y-%m-%d")

for i, pair in enumerate(portfolio):
    # Fetch the data
    data = fdata.BinanceCryptoPriceDatasetAdapter(ticker=pair, frequency=fdata.Frequency.HOURLY, training_set_date_range=('2025-01-01', today)).training_set

    # Ensure time is the index and price is renamed to the asset symbol
    data = data.set_index('time')
    data = data.rename(columns={'price': pair})

    if portfolio_df.empty:
        portfolio_df = data
    else:
        portfolio_df = portfolio_df.join(data, how='outer')        

# Fill any missing values with 0.0
portfolio_df.fillna(0.0, inplace=True)

# Save the portfolio prices to a CSV file
portfolio_df.to_csv("portfolio_prices.csv", index=True)
print(portfolio_df.head())

# Calculate the portfolio value by multiplying with the amounts
portfolioValue = portfolio_df.mul(portfolioAmount, axis=1)
totalValue = portfolioValue.sum(axis=1)

# Plotting with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=totalValue.index, y=totalValue, mode='lines', name='Total Portfolio Value'))
for pair in portfolio:
    fig.add_trace(go.Scatter(x=portfolioValue[pair].index, y=portfolioValue[pair], mode='lines', name=pair))
fig.update_layout(title='Total Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value (USDT)')
fig.show()





