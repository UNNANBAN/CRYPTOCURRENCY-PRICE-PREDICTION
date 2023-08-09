import os
import pandas as pd
import yfinance as yf #yahoo finance helpful for getting financial data
btc_ticker = yf.Ticker("BTC-USD")
btc = btc_ticker.history(period="max") #return everything from yahoo started tracking the bitcoin price
del btc["Dividends"], btc["Stock Splits"]
btc.index = pd.to_datetime(btc.index.strftime('%Y-%m-%d'))
if os.path.exists("bitcoin_price_data.csv"):
    os.remove("bitcoin_price_data.csv")
btc.to_csv("bitcoin_price_data.csv")
