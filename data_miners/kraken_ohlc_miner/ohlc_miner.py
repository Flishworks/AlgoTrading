import os
import sys
import krakenex
from pykrakenapi import KrakenAPI
api = krakenex.API()
k = KrakenAPI(api)
import time
import config
import logging

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
        for symbol in config.symbols:
            try:
                file = os.path.join(config.data_folder, symbol + '.csv')
                try:
                    last = get_last_line(file).split(',')[1]
                except:
                    last = 0
                price_data, _last = k.get_ohlc_data(symbol, interval = 1, since = last, ascending=True)
                price_data.to_csv(file, mode = "a+", header = not os.path.exists(file))
                logger.info(f'{len(price_data)} minutes of {symbol} added.')
                time.sleep(config.throttle) #for API throttling. Accounted for in sleep below
            except Exception as e:
                logger.error(f"Failed processing {symbol} : {e}")
        time.sleep(config.interval - config.throttle*len(symbol))

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