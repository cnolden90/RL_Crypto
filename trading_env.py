import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import pandas_ta as pta

import gym
from gym import spaces
from gym.utils import seeding


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)


class MarketData:
    def __init__(self, trading_days=252, coins=['EOS'], normalize=True, path='', date_min = '2020-01-01', date_max = '2021-12-31', taIndicator=False):
            self.coins = coins
            self.trading_days = trading_days
            self.normalize = normalize
            self.path = path
            self.datemin = date_min
            self.datemax = date_max
            self.taIndicator = taIndicator
            self.data = self.load_data()
            self.preprocess_data()
            self.min_values = self.data.select_dtypes(include=np.number).min()
            self.max_values = self.data.select_dtypes(include=np.number).max()
            self.step = 0
            self.offset = None


    def load_data(self):
        coin_dataframes = {}
        """ loads FEAR AND GREED for Crypto """
        fng_index = pd.read_csv('%s/data.csv' % self.path)
        fng_index['date'] = fng_index['date'].apply(lambda x: datetime.datetime.strptime(x, "%d-%m-%Y").strftime("%Y-%m-%d"))
        fng_index['date'] = pd.to_datetime(fng_index['date'])
        fng_index = fng_index[(fng_index['date'] > pd.to_datetime(self.datemin)) & (fng_index['date'] < pd.to_datetime(self.datemax)) ]
        fng_index = fng_index.set_index('date')
        fng_index = fng_index.drop('classification', axis=1)

        """ load OHCLV  for coins """
        for coin in self.coins:
            log.info('loading data for {}...'.format(coin))
            """loads OHCLV data for coin and creates key for df dic"""
            df_name = 'df_'+ coin
            ohclv = pd.read_json('%s/%s_USDT-5m.json' % (self.path, coin))
            ohclv.columns =['date_raw','open', 'high', 'low', 'close','volume']

            """prepare OHCLV data"""
            ohclv['date_time'] = ohclv['date_raw'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
            ohclv['date_time'] = pd.to_datetime(ohclv['date_time'])
            ohclv['date'] = ohclv['date_raw'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
            ohclv['date'] = pd.to_datetime(ohclv['date'])
            ohclv = ohclv[(ohclv['date'] > pd.to_datetime(self.datemin)) & (ohclv['date'] < pd.to_datetime(self.datemax))  ]
            ohclv = ohclv.set_index('date')

            """calculate returns and TA, then removes missing values"""
            if self.taIndicator:
                ohclv['returns'] = ohclv['close'].pct_change()
                ohclv['ret_2'] = ohclv['close'].pct_change(2)
                ohclv['ret_5'] = ohclv['close'].pct_change(5)
                ohclv['ret_10'] = ohclv['close'].pct_change(10)
                ohclv['ret_21'] = ohclv['close'].pct_change(21)
                ohclv['rsi'] = pta.rsi(ohclv['close'], length=14)
                ohclv = (ohclv.replace((np.inf, -np.inf), np.nan).dropna())

            """ loads marketcap data for coin """
            marketcap = pd.read_csv('%s/%s-usd-max.csv' % (self.path, coin))
            marketcap['date'] = marketcap['snapped_at'].apply(lambda x: datetime.datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S'))
            marketcap['date'] = pd.to_datetime(marketcap['date'])
            marketcap = marketcap.set_index('date')

            """ merge OHCLV & FEAR AND GREED INDEX (Crypto) """
            merged_df=pd.merge((pd.merge(ohclv,fng_index, how='inner', left_index=True, right_index=True)), marketcap, how='inner', left_index=True, right_index=True)
            #merged_df['pair'] = '%s_USDT' % coin
            coin_dataframes[df_name] = merged_df
            log.info('got data for {}...'.format(coin))

        frames = []
        for coins in coin_dataframes.keys():
            frames.append(coin_dataframes[coins])
        df = pd.concat(frames, ignore_index=False)
        return df

    def preprocess_data(self):
        """ cylicial feature encoding following: https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca"""
        self.data['dayOfWeek'] = self.data.index.dayofweek
        self.data['dayOfWeek_sin'] = np.sin(2 * np.pi * self.data['dayOfWeek']/self.data["dayOfWeek"].max())
        self.data['dayOfWeek_cos'] = np.cos(2 * np.pi * self.data['dayOfWeek']/self.data["dayOfWeek"].max())
        """ Date cos & sin """
        self.data['dayInYear'] = self.data['date_time'].apply(lambda x: int(x.strftime('%j')))
        self.data['dayInYear_sin'] = np.sin(2 * np.pi * self.data['dayInYear']/self.data["dayInYear"].max())
        self.data['dayInYear_cos'] = np.cos(2 * np.pi * self.data['dayInYear']/self.data["dayInYear"].max())
        """ Time cos & sin """
        self.data["timeOfDay"] = (self.data['date_time'].dt.hour*60) + self.data['date_time'].dt.minute
        self.data['timeOfDay_sin'] = np.sin(2 * np.pi * self.data['timeOfDay']/self.data['timeOfDay'].max())
        self.data['timeOfDay_cos'] = np.cos(2 * np.pi * self.data['timeOfDay']/self.data['timeOfDay'].max())
        """ Month cos & sin """
        self.data["monthInYear"] = self.data['date_time'].dt.month
        self.data['monthInYear_sin'] = np.sin(2 * np.pi * self.data['monthInYear']/self.data['monthInYear'].max())
        self.data['monthInYear_cos'] = np.cos(2 * np.pi * self.data['monthInYear']/self.data['monthInYear'].max())

        """ clean dataframe """
        self.data = self.data.drop('snapped_at', axis=1)
        self.data = self.data.drop('total_volume', axis=1)
        self.data = self.data.drop('price', axis=1)
        self.data = self.data.drop('date_raw', axis=1)
        self.data = self.data.drop('timeOfDay', axis=1)
        self.data = self.data.drop('dayOfWeek', axis=1)
        self.data = self.data.drop('monthInYear', axis=1)
        self.data = self.data.drop('dayInYear', axis=1)
        self.data = self.data.drop('date_time', axis=1)

        """ not scaling pairs, time, FGI, """
        df_not_scale = self.data[['dayOfWeek_sin','dayOfWeek_cos','dayInYear_sin','dayInYear_cos','timeOfDay_sin','timeOfDay_cos','monthInYear_sin','monthInYear_cos', 'classification_index']].copy()
        self.data = self.data.drop(['dayOfWeek_sin','dayOfWeek_cos','dayInYear_sin','dayInYear_cos','timeOfDay_sin','timeOfDay_cos','monthInYear_sin','monthInYear_cos', 'classification_index'], axis=1)

        """ scale OHCLV, MC & TA """
        if self.normalize:
            self.data = pd.DataFrame(scale(self.data),
                                     columns=self.data.columns,
                                     index=self.data.index)
        self.data = pd.concat([self.data, df_not_scale], axis=1)
        print(self.data.shape)

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - self.trading_days
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.trading_days
        return obs, done


class CryptoBot_Simulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """

        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action

        end_position = action - 1  # short, neutral, long
        n_trades = end_position - start_position
        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades

        # roughly value based since starting NAV = 1
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        reward = start_position * market_return - self.costs[self.step]
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])

        info = {'reward': reward,
                'nav'   : self.navs[self.step],
                'costs' : self.costs[self.step]}

        self.step += 1
        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'nav'            : self.navs,  # starting Net Asset Value (NAV)
                             'market_nav'     : self.market_navs,
                             'market_return'  : self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades})  # eod trade)


class CryptoBot_Environment(gym.Env):
    """ KUDOS: https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/22_deep_reinforcement_learning/04_q_learning_for_trading.ipynb
    A simple trading environment for reinforcement learning.
    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG
    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.
    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    If the NAV drops to 0, the episode ends with a loss.
    If the NAV hits 2.0, the agent wins.
    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """

    def __init__(self,
                 filepath,
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 coins=['EOS'],
                 date_min = '2020-01-01',
                 date_max = '2021-12-31',
                 taIndicator=False
                 ):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.coins = coins
        self.time_cost_bps = time_cost_bps
        self.data_source = MarketData(trading_days=self.trading_days,
                                      coins=coins,date_min = date_min, date_max = date_max, taIndicator=taIndicator, path=filepath)
        self.simulator = CryptoBot_Simulator(steps=self.trading_days,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.data_source.min_values,
                                            self.data_source.max_values)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        observation, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(action=action,
                                                market_return=observation[0])
        return observation, reward, done, info

    def reset(self):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]
