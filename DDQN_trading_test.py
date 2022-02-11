import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from time import time
from collections import deque
from random import sample

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import gym
from gym.envs.registration import register

np.random.seed(42)
tf.random.set_seed(42)
sns.set_style('whitegrid')

gpu_devices = False
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

results_path = Path('results', 'trading_bot')
if not results_path.exists():
    results_path.mkdir(parents=True)

trading_days = 252

register(
    id='trading-v0',
    entry_point='trading_env:CryptoBot_Environment',
    max_episode_steps=trading_days
)
trading_cost_bps = 1e-3
time_cost_bps = 1e-4

trading_environment = gym.make('trading-v0',
                               coins=['EOS'],
                               trading_days=trading_days,
                               trading_cost_bps=trading_cost_bps,
                               time_cost_bps=time_cost_bps)
trading_environment.seed(42)

state_dim = trading_environment.observation_space.shape[0]
print(trading_environment.observation_space.shape)
num_actions = trading_environment.action_space.n
max_episode_steps = trading_environment.spec.max_episode_steps
print(max_episode_steps)
