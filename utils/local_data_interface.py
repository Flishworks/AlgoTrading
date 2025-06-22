import os 
import pandas as pd
from openbb import obb
from datetime import datetime

LOCAL_DATA_DIR = r"D:\other_data\fin_data"
PROVIDER = 'fmp' #'yfinance'

def get_all_stock_lists() -> list:
    stock_list_dir = os.path.join(LOCAL_DATA_DIR, 'stock_lists')
    return [x.split('.')[0] for x in os.listdir(stock_list_dir) if x.endswith('.csv')]

def get_stock_list(list_name: str) -> pd.DataFrame:
    if not list_name.endswith('.csv'):
        list_name = list_name + '.csv'
    stock_list_dir = os.path.join(LOCAL_DATA_DIR, 'stock_lists')
    if list_name not in os.listdir(stock_list_dir):
        raise FileNotFoundError(f'{list_name} not found in {stock_list_dir}')
    return pd.read_csv(os.path.join(stock_list_dir, list_name))['symbol'].tolist()

def get_sp500_by_date(date: str) -> pd.DataFrame:
    # data file obtained from  https://github.com/fja05680/sp500/tree/master
    sp500_by_date_df = pd.read_csv(os.path.join(LOCAL_DATA_DIR, 'S&P 500 Historical Components & Changes(12-10-2024).csv'))
    sp500_by_date_df ['date'] = pd.to_datetime(sp500_by_date_df['date'])
    #find closest match to date
    sp500_by_date_df['date_diff'] = (sp500_by_date_df['date'] - pd.to_datetime(date)).abs()
    closest_date = sp500_by_date_df.loc[sp500_by_date_df['date_diff'].idxmin()]['date']
    return sp500_by_date_df.loc[sp500_by_date_df['date'] == closest_date]['tickers'].values[0].split(',')

def get_local_ticker(symbol: str, start_date: str, interval: str, end_date:str = None) -> pd.DataFrame:
    symbol = symbol.lower() # make lower case
    stock_data_dir = os.path.join(LOCAL_DATA_DIR, 'stock_OHLC')
    if not symbol.endswith('.csv'):
        symbol = symbol + '.csv'
    if interval not in os.listdir(stock_data_dir):
        #create folder
        os.mkdir(os.path.join(stock_data_dir, interval))
    if symbol not in os.listdir(os.path.join(stock_data_dir, interval)):
        raise FileNotFoundError(f'{symbol} not found in {stock_data_dir}')
    if end_date is None:
        end_date = pd.to_datetime('today')
    stock_df = pd.read_csv(os.path.join(stock_data_dir, interval, symbol))
    #convert to date
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df = stock_df.loc[(stock_df['date'] >= pd.to_datetime(start_date)) & (stock_df['date'] <= pd.to_datetime(end_date))]
    stock_df.set_index('date', inplace=True) # convert column "date" to index
    return stock_df

def store_local_ticker(symbol: str, data: pd.DataFrame, interval: str):
    symbol = symbol.lower() # make lower case
    stock_data_dir = os.path.join(LOCAL_DATA_DIR, 'stock_OHLC')
    if not symbol.endswith('.csv'):
        symbol = symbol + '.csv'
    if interval not in os.listdir(stock_data_dir):
        #create folder
        os.mkdir(os.path.join(stock_data_dir, interval))
    data.to_csv(os.path.join(stock_data_dir, interval, symbol), index=True)
    
def get_ticker(symbol: str, start_date: str, interval: str, end_date:str = None) -> pd.DataFrame:
    '''
    Retrieves stock data from local data directory if available
    Any data not available locally will be retrieved from the internet
    '''
    if end_date is None:
        end_date = pd.to_datetime('today').date()
    try: 
        stock_df = get_local_ticker(symbol, start_date, interval, end_date)
        # Retrieve any missing data
        mod_flag = 0
        if stock_df.index[-1].date() < pd.to_datetime(end_date).date():
            try: 
                missing_data = obb.equity.price.historical(symbol=symbol, provider=PROVIDER, start_date=stock_df.index[-1]+pd.Timedelta(days=1), end_date=end_date, interval=interval).to_df()
                missing_data.index = pd.to_datetime(missing_data.index)
                stock_df = pd.concat([stock_df, missing_data])
                mod_flag = 1
            except Exception:
                pass
        if stock_df.index[0].date() > pd.to_datetime(start_date).date():
            try: 
                missing_data = obb.equity.price.historical(symbol=symbol, provider=PROVIDER, start_date=start_date, end_date=stock_df.index[0]-pd.Timedelta(days=1), interval=interval).to_df()
                missing_data.index = pd.to_datetime(missing_data.index)
                stock_df = pd.concat([missing_data, stock_df])
                mod_flag = 1
            except Exception:
                pass
        if mod_flag == 1:
            store_local_ticker(symbol, stock_df, interval)
        return stock_df
    except FileNotFoundError:
        #get data from internet
        stock_df = obb.equity.price.historical(symbol=symbol, provider=PROVIDER, start_date=start_date, end_date=end_date, interval=interval).to_df()
        store_local_ticker(symbol, stock_df, interval)
        return stock_df