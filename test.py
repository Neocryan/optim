import numpy as np
import pandas as pd
import scipy.optimize as optim
import logging
import pickle

# Load price data
data = pickle.load(open('sample.pkl', 'rb'))  # a list of ten years data, each year is a list of floats.
logging.basicConfig(level=logging.INFO)

#### functions:


def pfe(close, lookback):
    """
    compute the PFE
    :param close: list or numpy.ndarray
    :param lookback: int
    :return: np.ndarray
    """
    close = np.array(close)
    btm = np.convolve((np.diff(close) ** 2 + 1) ** 0.5, np.ones(lookback, dtype='int'), 'valid')
    diff = close[lookback:] - close[:-lookback]
    top = (diff ** 2 + lookback ** 2) ** 0.5 * np.sign(diff)
    assert btm.shape == top.shape, (top.shape, btm.shape)
    return top / btm

# We make an environment using the 10 year's data to compute the yearly sharpe with the algorithm


class DiscreteEnv:
    # This Environment ingest the 10 year's data (self.__init__),
    # and given the every day's signal (1 for buy and -1 for sell)
    # this Environment also calculates the sharpe (self.act)

    """
    Class for splitted training-testing data
    :param data: list or array of continuous data
    """
    MAX_LOOKBACK = 200
    MAX_DRAWDOWN = False

    def __init__(self, close, high=None, low=None, train_size: int = 7):
        """
        :param close: list or array of continuous data, each item is a list or ndarray
        """
        # In this sample, we do not split the train test data, bc the test is on a more complicated env
        self.train_size = train_size
        assert len(close) > 1, "expected the data as a list of yearly data, but only find {} years".format(len(close))

        # Record the days in each year, and from the second year, extend the previous year
        days = [len(x) for x in close]
        self.close = []
        if high:
            self.high = []
            self.low = []

        # Padding the data
        for i in range(1, len(close)):
            self.close.append(np.array(close[i - 1][-self.MAX_LOOKBACK:] + close[i]))
            if high:
                self.high.append(
                    np.array(high[i - 1][-self.MAX_LOOKBACK:] + high[i]))  # extend the data from the second year
                self.low.append(
                    np.array(low[i - 1][-self.MAX_LOOKBACK:] + low[i]))  # extend the data from the second year

        # Initial randomly the years to be used
        self.train_index = set(np.random.choice(range(len(self.close)), self.train_size, replace=False))
        self.train_index = sorted(list(self.train_index))

        self.train_close = [np.array(self.close[x]) for x in self.train_index]
        if high:
            self.train_high = [np.array(self.high[x]) for x in self.train_index]
            self.train_low = [np.array(self.low[x]) for x in self.train_index]
        self.train_days = [days[x + 1] for x in self.train_index]  # X + 1 is because we ignore the first year

    def train(self, signals):
        """
        Function for computing sharpe for optimization functions
        :param signals: array-like signals
        :return:
        """
        total = 0

        for i in range(len(signals)):
            total += self.act(self.train_close[i], signals[i])[0]

        return total / len(signals)

    def act(self, close, signal):
        """
        :param close: array-like close price
        :param signal: array of the signals. -1:short, 0:neutral, 1:long
        :return: sharpe list, max_drawdown list
        """
        # signal -> [t_0 , t_n]
        # position -> [t_1, t_n+1]
        # real_return = diff(close) -> [t_1, t_n]
        close = np.array(close)
        lag = close.shape[0] - np.array(signal).shape[0]

        position = signal[:-1]  # from [t_1, t_n+1] ->  [t_1, t_n]
        pnl = position * np.diff(close[lag:])  # [t_1, t_n] , [lag + 1, ]
        returns = pnl / close[lag:-1]  # [lag, -1]

        max_drawdown = None
        if self.MAX_DRAWDOWN:
            index = np.zeros(returns.shape[0] + 1)
            index[0] = 100
            # Computing the list of index (portfolio value)
            for i in range(1, index.shape[0]):
                index[i] = index[i - 1] * (1 + returns[i - 1])

            # Computing Max Drawdown
            lowest = np.argmax(np.maximum.accumulate(index) - index)

            # Make sure the day with lowest index is not the first day
            if lowest == 0:
                peak = 0
            else:
                peak = np.argmax(index[:lowest])

            max_drawdown = (index[lowest] - index[peak]) / index[peak]
        sharpe = returns.mean() / (returns.std() + 1e-8) * 252 ** 0.5
        return sharpe, max_drawdown


class PFEVOptim:
    # This class it the algorithm it self. it is very simple: compute the pfe and volatility
    # And given the pfe and volatility and their threshold, return the signal to the environment
    # The objective function is self.__call__, which is the average sharpe ratio of training years
    def __init__(self, env):
        self.env = env

    @staticmethod
    def optim_range():
        # Optim range for brute optimizing. (start, end, step)
        return [slice(2, 30, 4),
               slice(-0.5, 1, 0.3),
               slice(-1, 1, 0.3),
                slice(10,70,30),
                slice(0,0.1,0.03),
                slice(0,0.8,0.4)]

    @staticmethod
    def headers():
        return ['pfe_lookback', 'buypfe', 'sellpfe',
                'volatility_lookback', 'volatility_threshold',
                'size_shrink']

    @staticmethod
    def process(high, low, close, x, *args, **kwargs):
        # Compute the signal with the algorithm
        pfe_lookback, buypfe, sellpfe = int(x[0]), float(x[1]), float(x[2])
        volatility_lookback, volatility_threshold = int(x[3]), float(x[4])
        size_shrink = float(x[5])

        returns = np.diff(close) / close[:-1]

        PFE = pfe(close, pfe_lookback)
        volatility = np.array(pd.Series(returns).rolling(volatility_lookback).std()) * np.sqrt(252 / volatility_lookback)

        if volatility.shape[0] >= PFE.shape[0]:
            volatility = volatility[-PFE.shape[0]:]
        else:
            PFE = PFE[-volatility.shape[0]:]

        signal_pfe = (PFE > buypfe).astype('int') - (PFE < sellpfe).astype('int')
        signal_volatility = 1 - (volatility >= volatility_threshold).astype('int') * size_shrink

        return signal_pfe * signal_volatility

    def __call__(self, x: list):
        signals = [self.process(None,
                                None,
                                self.env.train_close[i],
                                x) for i in range(len(self.env.train_close))]


        return -self.env.train(signals)


# Make the environment
env_optim = DiscreteEnv(data, high=None, low=None, train_size=6)
obj_func = PFEVOptim(env=env_optim)

# Optimizing
x = optim.brute(func=obj_func,
                ranges=obj_func.optim_range(), full_output=True)


logging.info("Training Performance: Average yearly sharpe\t{}, parameters\t{}".format(-x[1], x[0]))