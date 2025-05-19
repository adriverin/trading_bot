import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# These functions require the tiker as an argument but it is only used for title and not for the data plotted
#TO SOLVE!!!


def plot_plotly(data: pd.DataFrame, ticker: str):
    """
    Plot info about crypto prices.
    args:
        data: pd.DataFrame
        ticker: str
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[data.columns[0]], y=data[data.columns[1]], mode='lines', name=f'{ticker} Price Over Time'))
    fig.update_layout(title=f'{ticker} Price Over Time', xaxis_title='Date', yaxis_title='Value (USDT)')
    fig.show()


def plot_matplotlib(data: pd.DataFrame, ticker: str):
    """
    Plot info about crypto prices.
    args:
        data: pd.DataFrame
        ticker: str
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data[data.columns[0]], data[data.columns[1]], color='royalblue', linewidth=2, label=f'{ticker} Price')
    plt.xlabel('Time', fontsize=16, fontweight='bold')
    plt.ylabel('Price (USDT)', fontsize=16, fontweight='bold')
    plt.title(f'{ticker} Price Over Time', fontsize=20, fontweight='bold', color='darkslategray')
    plt.xticks(rotation=45)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_characteristic_function(cf):
    """
    Plot characteristic function for a distribution. Just for playing around.
    args:
        cf: list of tuples (characteristic function name, data)
    """
    n = len(cf) # number of cf to plot
    row = int(n / 2)

    fig, ax = plt.subplots(row, 2, figsize=(14, 7 * row))

    i, r = 0, 0

    def _axis_plot_cf(characteristic_function, column_index):
        ax[r, column_index].set_title(characteristic_function[0])
        sns.lineplot(ax=ax[r, column_index], data=characteristic_function[1], x='omega', y='cf(omega)', lw=2, color='royalblue')


    while i < n:
        _axis_plot_cf(cf[i], 0)
        _axis_plot_cf(cf[i+1], 1)
        i += 2
        r += 1

    plt.tight_layout()
    plt.show()