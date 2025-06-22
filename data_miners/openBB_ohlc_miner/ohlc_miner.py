import os
import sys
from openbb import obb
import time
import config
import logging
from datetime import datetime

sys.path.append(r'C:\Users\avido\Documents\other code\AlgoTrading')
from assets.api_credentials import openbb_pat
obb.account.login(pat=openbb_pat, remember_me=True)

#ensure logfile exists
os.makedirs(os.path.dirname(config.logfile), exist_ok=True)

formatter = logging.Formatter('%(levelname)s|%(asctime)s|%(message)s')
filehandler = logging.FileHandler(filename=config.logfile)
filehandler.setFormatter(formatter)
filehandler.setLevel(logging.INFO)
streamhandler = logging.StreamHandler()
streamhandler.setFormatter(formatter)
streamhandler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


def main_loop():
    while True:
        if datetime.now().strftime('%H:%M') != config.run_time_of_day:
            time.sleep(30)
            continue
        for symbol in config.symbols:
            try:
                file = os.path.join(config.data_folder, symbol + '.csv')
                try:
                    last_dt = get_last_line(file).split(',')[0]
                    last_dt = datetime.strptime(last_dt, '%Y-%m-%d %H:%M:%S')
                    if last_dt.date() == datetime.now().date():
                        continue #if last line is today, that means we already ran today. skip.
                    last_dt = last_dt 
                except:
                    #if file is empty, set last_dt to 1980-01-01 to do a bulk call of as much data as possible
                    last_dt = '1980-01-01 00:00:00'

                price_data = obb.equity.price.historical(symbol=symbol,  provider=config.provider, interval=config.ohlc_interval, start_date=last_dt).to_df()
                
                # round open,high,low,close to 2 decimal places
                price_data['open'] = price_data['open'].round(2)
                price_data['high'] = price_data['high'].round(2)
                price_data['low'] = price_data['low'].round(2)
                price_data['close'] = price_data['close'].round(2)
                
                price_data.to_csv(file, mode = "a+", header = not os.path.exists(file))
                
                logger.info(f'OHLC collected for {symbol}. {config.symbols.index(symbol)/len(config.symbols)}% complete.')
                time.sleep(config.throttle) #for API throttling. Accounted for in sleep below
            except Exception as e:
                logger.error(f"Failed processing {symbol} : {e}")

def get_last_line(file):
    with open(file, 'rb') as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        return  f.readline().decode()


if __name__ == '__main__':
    print(f"{__file__} has started")
    logger.info(f"{__file__} has started")
    try:
        main_loop()
    except KeyboardInterrupt:
        sys.exit(0)