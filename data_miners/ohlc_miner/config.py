symbols = ['SOLUSD', 'ETHUSD', 'BTCUSD', 'ADAUSD', 'ALGOUSD']
data_folder = r'C:\Users\avido\Desktop\AlgoTrading\data\scraped_OHLC'
interval = 60 * 60 #seconds between each query for new data
throttle = 1 #seconds in between API query for each symbol, in seconds. Also subtracted from interval.
logfile =r'C:\Users\avido\Desktop\AlgoTrading\logs\ohlc_miner.log'