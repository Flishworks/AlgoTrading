# %%
import pandas as pd
import numpy as np



#### Strategies ####
# strategies must be callable with the same form as strategy_base:
class strategy_base():
    def __init__(self):
        #initialize state variables
        ...

    def __call__(self, new, reserve, invested):
        # args:
        #   new: the new price
        #   reserve: the amount available to buy
        #   invested: the amount invested
        # returns:
        #    dollar amount to buy (negative means sell)
        #    (if the amount is greater than the reserve, it will be set to the reserve)
        ...

class hodl(strategy_base):
    #naive strategy for comparison
    def __init__(self):
        ...
        
    def __call__(self, new, reserve, invested):
        return 0

class DCA(strategy_base):
    #Dollar-cost averaging
    def __init__(self, period, amount):
        self.period = period #the number of iterations between buys
        self.amount = amount #the amount to buy each time
        self.count = 0
        
    def __call__(self, new, reserve, invested):
        self.count+=1
        if self.count%self.period == 0:
            return self.amount
        else:
            return 0

class randomDCA(strategy_base):
    #Dollar-cost averaging with randomness
    def __init__(self, period, amount):
        self.period = period #the number of iterations between buys
        self.amount = amount #the amount to buy each time
        self.count = 0
        
    def __call__(self, new, reserve, invested):
        self.count+=np.random.randn() + 1
        if self.count > self.period:
            self.count = 0
            return self.amount*(np.random.randn() + 1)
        else:
            return 0


class maintain_investment:
    def __init__(self, thresh_percent = .02):
        self.initial_investment = None
        self.thresh_percent = thresh_percent

    def __call__(self, new, reserve, invested):
        if self.initial_investment is None:
            self.initial_investment = invested
        percent_swing = (new - self.initial_investment) / self.initial_investment
        if np.abs(percent_swing) < self.thresh_percent:
            return 0
        return  -1 * (invested - self.initial_investment)

class percentage_from_mean(strategy_base):
    def __init__(self, avg_start = 0, beta = 0.2, limit_percent = .10, thresh_percent = .01, adjust = .5):
        self.avg = avg_start
        self.beta = beta
        self.limit_percent = limit_percent
        self.thresh_percent = thresh_percent
        self.adjust = adjust
        
    def __call__(self, new, reserve, invested):
        self.avg = exponential_moving_average(new, self.avg, self.beta)
        percent_swing = (new - self.avg) / (self.avg)
        if np.abs(percent_swing) < self.thresh_percent:
            return 0
        percent_of_limit = percent_swing / self.limit_percent
        if percent_of_limit > 0:
            trade = -1 * percent_of_limit * invested * self.adjust
        else:
            trade = -1 * percent_of_limit * reserve * self.adjust
        return trade

class percentage_from_mean_simple(strategy_base):
    def __init__(self, beta = 0.2, thresh_percent = .01, trade_percent = 10, trade_amount = 10):
        self.avg = None
        self.beta = beta
        self.thresh_percent = thresh_percent
        self.trade_percent = trade_percent
        self.trade_amount = trade_amount
        
    def __call__(self, new, reserve, invested):
        if self.avg is None:
            self.avg = new
        else:
            self.avg = exponential_moving_average(new, self.avg, self.beta)
        percent_swing = (new - self.avg) / (self.avg)
        if np.abs(percent_swing) < self.thresh_percent:
            return 0
        if percent_swing > 0:
            trade = -1 * (self.trade_percent * invested + self.trade_amount)
        else:
            trade = self.trade_percent * reserve + self.trade_amount
        return trade
    
class slow_fast_avg(strategy_base):
    def __init__(self, avg_start = 0, beta_slow = 0.05, beta_fast = 0.3, trade_amount = 1):
        self.avg_slow = avg_start
        self.avg_fast = avg_start
        self.avg_slow_ot = []
        self.avg_fast_ot = []
        self.beta_slow = beta_slow
        self.beta_fast = beta_fast
        self.trade_amount = trade_amount
        
    def __call__(self, new, reserve, invested):
        self.avg_slow = exponential_moving_average(new, self.avg_slow, self.beta_slow)
        self.avg_fast = exponential_moving_average(new, self.avg_fast, self.beta_fast)
        self.avg_slow_ot.append(self.avg_slow)
        self.avg_fast_ot.append(self.avg_fast)
        if (new > self.avg_slow) and (new < self.avg_fast):
            #sell
            return -1*self.trade_amount
        if (new < self.avg_slow) and (new > self.avg_fast):
            #buy
            return self.trade_amount
        return 0

class inverse_slow_fast_avg(strategy_base):
    def __init__(self, avg_start = 0, beta_slow = 0.05, beta_fast = 0.3, trade_amount = 1):
        self.avg_slow = avg_start
        self.avg_fast = avg_start
        self.avg_slow_ot = []
        self.avg_fast_ot = []
        self.beta_slow = beta_slow
        self.beta_fast = beta_fast
        self.trade_amount = trade_amount
        
    def __call__(self, new, reserve, invested):
        self.avg_slow = exponential_moving_average(new, self.avg_slow, self.beta_slow)
        self.avg_fast = exponential_moving_average(new, self.avg_fast, self.beta_fast)
        self.avg_slow_ot.append(self.avg_slow)
        self.avg_fast_ot.append(self.avg_fast)
        if (new > self.avg_slow) and (new < self.avg_fast):
            #buy
            return self.trade_amount
            
        if (new < self.avg_slow) and (new > self.avg_fast):
            #sell
            return -1*self.trade_amount
        
        return 0
    
class opportunistic(strategy_base):
    def __init__(self, avg_start = 0, beta = 0.1, thresh_percent = .02, trade_amount = 1):
        self.avg = avg_start
        self.avg_ot = []
        self.beta = beta
        self.thresh_percent = thresh_percent
        self.trade_amount = trade_amount
        
    def __call__(self, new, reserve, invested):
        self.avg = exponential_moving_average(new, self.avg, self.beta)
        self.avg_ot.append(self.avg)
        if (new > self.avg*(1+self.thresh_percent)):
            #sell
            return -1*self.trade_amount ##-.1*invested
        if (new < self.avg*(1-self.thresh_percent)):
            #buy
            return self.trade_amount #.1*reserve
        return 0

class momentum(strategy_base):
    def __init__(self, trade_amount = .1, trade_cap = 1):
        self.trade_amount = trade_amount
        self.momentum = 0
        self.last = 0
        self.trade_cap = trade_cap
        
    def __call__(self, new, reserve, invested):
        if (new > self.last):
            #buy
            self.momentum += self.trade_amount
        if (new < self.last):
            #sell
            self.momentum -= self.trade_amount
        self.last = new
        self.momentum = np.clip(self.momentum, -self.trade_cap, self.trade_cap)
        return self.momentum

class inverse_momentum(strategy_base):
    def __init__(self, trade_amount = .1, trade_cap = 1):
        self.trade_amount = trade_amount
        self.momentum = 0
        self.last = 0
        self.trade_cap = trade_cap
        
    def __call__(self, new, reserve, invested):
        if (new > self.last):
            #sell
            self.momentum -= self.trade_amount
        if (new < self.last):
            #buy
            self.momentum += self.trade_amount
        self.last = new
        self.momentum = np.clip(self.momentum, -self.trade_cap, self.trade_cap)
        return self.momentum

class avg_derivitive(strategy_base):
    def __init__(self, avg_start = 0, beta = 0.1):
        self.avg = avg_start
        self.avg_ot = []
        self.beta = beta
        self.last_avg = avg_start

    def __call__(self, new, reserve, invested):
        self.avg = exponential_moving_average(new, self.avg, self.beta)
        self.avg_ot.append(self.avg)
        mean_derivitive_percent = (self.avg - self.last_avg) / (self.last_avg)
        self.last_avg = self.avg
        if mean_derivitive_percent < 0:
            #sell
            return mean_derivitive_percent * invested
        else:
            #buy
            return mean_derivitive_percent * reserve
    
class copy_last(strategy_base):
    #trades based on the last diff (if it was positive, buy, if it was negative, sell)
    def __init__(self, amount = 1):
        self.amount = 1
        self.last = 0

    def __call__(self, new, reserve, invested):
        diff = new - self.last
        if diff > 0:
            #buy
            self.last = new
            return self.amount
        if diff < 0:
            #sell
            self.last = new
            return -1*self.amount
        return 0

class decorrelation_time_copy_last(strategy_base):
    #if decorrelation time of diffs over a specified window is greater than a threshold, apply copy_last
    def __init__(self, amount = 1, decor_time_thresh = 5, window = 40):
        self.amount = amount
        self.last = 0
        self.decor_time_thresh = decor_time_thresh
        self.window = window
        self.diffs = []
        
    def __call__(self, new, reserve, invested):
        if self.last == 0:
            self.last = new
        diff = new - self.last
        self.diffs.append(diff)
        if len(self.diffs) > self.window:
            self.diffs.pop(0)
            decorrelation_time = 1
            while np.sum(np.array(self.diffs[decorrelation_time:]) * np.array(self.diffs[:-decorrelation_time])) > 0:
                decorrelation_time += 1
                if (decorrelation_time > self.decor_time_thresh):
                    #buy or sell
                    self.last = new
                    return self.amount if diff > 0 else -1*self.amount
        return 0
class copy_last_all_in(strategy_base):
    #trades based on the last diff (if it was positive, buy, if it was negative, sell)
    def __init__(self):
        self.last = 0

    def __call__(self, new, reserve, invested):
        if self.last == 0:
            self.last = new
        diff = new - self.last
        if diff > 0:
            #buy
            self.last = new
            return reserve
        if diff < 0:
            #sell
            self.last = new
            return -1*invested
        return 0
    
class decorrelation_time_copy_last_all_in(strategy_base):
    #if decorrelation time of diffs over a specified window is greater than a threshold, apply copy_last_all_in
    def __init__(self, decor_time_thresh = 5, window = 40):
        self.last = 0
        self.decor_time_thresh = decor_time_thresh
        self.window = window
        self.diffs = []
        
    def __call__(self, new, reserve, invested):
        if self.last == 0:
            self.last = new
        diff = new - self.last
        self.diffs.append(diff)
        if len(self.diffs) > self.window:
            self.diffs.pop(0)
            decorrelation_time = 1
            while np.sum(np.array(self.diffs[decorrelation_time:]) * np.array(self.diffs[:-decorrelation_time])) > 0:
                decorrelation_time += 1
                if (decorrelation_time > self.decor_time_thresh):
                    #buy or sell
                    self.last = new
                    return reserve if diff > 0 else -1*invested
        return 0
    
class ema_slope(strategy_base):
    #makes a trade based on the slope of the ema
    def __init__(self, beta = 0.1):
        self.avg = 0
        self.last_avg = 0
        self.beta = beta
        
    def __call__(self, new, reserve, invested):
        self.avg = exponential_moving_average(new, self.avg, self.beta)
        ema_diff = self.avg - self.last_avg
        self.last_avg = self.avg
        return ema_diff

class ema_accel(strategy_base):
    #makes a trade based on the second derivitive of the ema
    def __init__(self, beta = 0.1, factor = 1):
        self.avg = 0
        self.last_avg = 0
        self.beta = beta
        self.last_diff = 0  
        self.factor = factor
        
    def __call__(self, new, reserve, invested):
        self.avg = exponential_moving_average(new, self.avg, self.beta)
        ema_diff = self.avg - self.last_avg
        self.last_avg = self.avg
        trade = ema_diff - self.last_diff
        self.last_diff = ema_diff
        return trade * self.factor
    
class dynamic_ema_slope(strategy_base):
    #makes the beta param based on the moving variance of the ema
    def __init__(self, beta_factor = 0.1):
        self.avg = 0
        self.sum_of_squares_short = 0
        self.sum_of_squares_long = 0
        self.last_avg = 0
        self.beta_factor = beta_factor
        
    def __call__(self, new, reserve, invested):
        if self.avg==0:
            self.avg = new
        self.sum_of_squares_short = exponential_moving_average((new-self.avg)**2, self.sum_of_squares_short, .3)
        self.sum_of_squares_long = exponential_moving_average((new-self.avg)**2, self.sum_of_squares_long, .01)
        sos_norm = self.sum_of_squares_short / self.sum_of_squares_long
        std = np.sqrt(sos_norm)
        if std == 0 or np.isnan(std) or std>1:
            std = 1
        self.avg = exponential_moving_average(new, self.avg, self.beta_factor * std)
        if np.isnan(self.avg):
            self.avg = new
        ema_diff = self.avg - self.last_avg
        self.last_avg = self.avg
        return ema_diff          
        
class ema_slope_reversal(strategy_base):
    # buy when slope of ema changes from negative to positive, sell when it changes from positive to negative
    def __init__(self, beta = 0.1):
        self.avg = 0
        self.last_avg = 0
        self.beta = beta
        self.last_diff = 0
        
    def __call__(self, new, reserve, invested):
        self.avg = exponential_moving_average(new, self.avg, self.beta)
        ema_diff = self.avg - self.last_avg
        self.last_avg = self.avg
        trade = 0
        if ema_diff > 0 and self.last_diff < 0:
            trade = reserve
        elif ema_diff < 0 and self.last_diff > 0:
            trade = -1*invested
        self.last_diff = ema_diff
        return trade
        
        
    
class threshold_buy_sell(strategy_base):
    # buys/sells on any price movement greater than a given threshold in that direction
    def __init__(self, threshold_up = 0.05, trade_percent_up = 1, trade_amount_up = 10, threshold_down = 0.05, trade_percent_down = 1, trade_amount_down = 10):
        self.threshold_up = threshold_up
        self.trade_percent_up = trade_percent_up
        self.trade_amount_up = trade_amount_up
        self.threshold_down = threshold_down
        self.trade_percent_down = trade_percent_down
        self.trade_amount_down = trade_amount_down
        self.last = 0
        
    def __call__(self, new, reserve, invested):
        trade = 0
        if (new - self.last) / self.last > self.threshold_up:
            # if movement up, sell
            trade = -1 * (self.trade_percent_up / 100 * invested + self.trade_amount_up)
        if (new - self.last) / self.last < -1*self.threshold_down:
            # if movement down, buy
            trade = (self.trade_percent_down / 100 * reserve + self.trade_amount_down)
        self.last = new
        return trade
    
#### Simulator ####
class trade_simulator:
    def __init__(self, price_data, initial_invested = 100, initial_reserve = 100, trade_cost = 0, trade_cost_percent = 0):
        self.price_data = price_data 
        self.last = None
        self.initial_invested = initial_invested
        self.initial_reserve = initial_reserve
        self.reserve = initial_reserve
        self.invested = initial_invested
        self.trades = [] #keeps track of each trade
        self.reserve_ot = [] #keeps track of how much capital is in reserve over time
        self.invested_ot = [] #keeps track of how much capital is invested over time
        self.baselines = [] #keeps track of how much initial capital is worth if left untouched
        self.trade_cost = trade_cost
        self.trade_cost_percent = trade_cost_percent #cost that is a percent of the trade amount
        
    def run(self, strategy):
        self.last = self.price_data.iloc[0]
        for i in range(len(self.price_data)):
            price = self.price_data.iloc[i]
            self.invested *= price/self.last
            self.last = price
            trade = strategy(price, self.reserve, self.invested)
            if trade > 0 and trade > self.reserve:   # if the buy is greater than the reserve, set it to the reserve
                trade = self.reserve 
            if trade < 0 and -1*trade > self.invested: # if the sell is greater than amount invested, set it to the amount invested
                trade = -1*self.invested
            self.reserve -= trade
            self.invested += trade
            if np.abs(trade) >= 0:
                #apply costs
                cost = np.abs(trade) * self.trade_cost_percent / 100 + self.trade_cost
                if self.reserve > cost:
                    self.reserve -= cost
                else:
                    self.invested -= cost
            #check for zero balance
            if self.reserve + self.invested <= 0:
                print("Balance reached zero at iteration " + str(i))
                break
            self.reserve_ot.append(self.reserve)
            self.invested_ot.append(self.invested)
            self.trades.append(trade)
            self.baselines.append((self.initial_invested + self.initial_reserve) * price / self.price_data.iloc[0])
        #print results
        print("Reserve: " + str(self.reserve))
        print("Invested: " + str(self.invested))
        print("Total: " + str(self.reserve + self.invested))
        print("Baseline: " + str((self.initial_invested + self.initial_reserve) * price / self.price_data.iloc[0]))
        print("\n")


#### helper funcs ####
def exponential_moving_average(new, old, beta = .2):
    return beta * new + (1 - beta) * old



#### Main ####
if __name__ == "__main__":
    price_data = pd.read_csv('SOL\SOLUSD.csv', names = ['time', 'price', 'volume'])
    price_data['time'] = pd.to_datetime(price_data['time'], unit = 's')
    price_data.set_index('time', inplace = True)
    price_data.dropna(inplace = True)
    price_data = price_data.resample('1H').mean().fillna(method='ffill')
    last_n = 1000
    strategy = percentage_from_mean(avg_start = price_data['price'].iloc[-last_n])
    sim = trade_simulator(price_data[-last_n:]['price'].reset_index(drop = True))
    sim.run(strategy)



