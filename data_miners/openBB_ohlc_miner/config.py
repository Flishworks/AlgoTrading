from openbb import obb
import sys
sys.path.append(r'C:\Users\avido\Documents\other code\AlgoTrading')
from assets.api_credentials import openbb_pat
obb.account.login(pat=openbb_pat, remember_me=True)

# symbols = ['SPY', 'VIX', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ', 'ARKX', 'ARKF', 'ARKR', 'ARKO', 'ARKF', 'ARKW']
stocks = obb.equity.screener(provider='fmp').to_df()
us_tickers = stocks[stocks['country'] == 'US']
symbols = us_tickers.symbol.to_list()

data_folder = r'C:\Users\avido\Documents\other code\AlgoTrading\data\scraped_ohlc'
throttle = 5 #seconds in between API query for each symbol, in seconds. Also subtracted from interval.
logfile =r'C:\Users\avido\Documents\other code\AlgoTrading\data_miners\logs\openBB_miner.log'
provider = 'yfinance'
ohlc_interval = '1m'
run_time_of_day = "21:04" #'19:00' #only on the minute. Should be after market close in your time zone. 