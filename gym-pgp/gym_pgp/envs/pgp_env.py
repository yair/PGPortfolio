import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import sys
import os
sys.path.append(os.path.abspath("/home/yair/w/PGPortfolio/pgportfolio"))
sys.path.append(os.path.abspath("/home/yair/w/PGPortfolio"))
#from pgportfolio.tools.configprocess import preprocess_config
from pgportfolio.tools.configprocess import load_config
from pgportfolio.marketdata.datamatrices import DataMatrices
import json
import numpy as np
import time
from datetime import datetime

class PGPEnv(gym.Env):

    def __init__(self):
        self.config = load_config()
        self.observation_space = spaces.Tuple((spaces.Box(low=-float('Inf'), high=float('Inf'),
                                                          shape=(self.config["input"]["coin_number"], # coin
                                                                 self.config["input"]["feature_number"], # HLC (or permutation)
                                                                 self.config["input"]["window_size"]), # time periods
                                                          dtype=np.float32),
                                               spaces.Box(low=0., high=1.,
                                                          shape=(self.config["input"]["coin_number"] + 1,), # Omega (cash and coins)
                                                          dtype=np.float32)))
        self.action_space = spaces.Box(low=0., high=1.,
                                       shape=(self.config["input"]["coin_number"] + 1,), # Omega (cash and coins)
                                       dtype=np.float32)
        self.load_data_matrices()
        self.cv = self.dm.calculate_consumption_vector (self.config)
#        self.reset()

    def load_data_matrices(self):
        config = self.config
        start = time.mktime(datetime.strptime(config["input"]["start_date"], "%Y/%m/%d").timetuple())
        end = time.mktime(datetime.strptime(config["input"]["end_date"], "%Y/%m/%d").timetuple())
        self.dm = DataMatrices(start=start,
                               end=end,
                               feature_number=config["input"]["feature_number"],
                               window_size=config["input"]["window_size"],
                               online=True,
                               period=config["input"]["global_period"],
                               volume_average_days=config["input"]["volume_average_days"],
                               coin_filter=config["input"]["coin_number"],
                               is_permed=config["input"]["is_permed"],
                               test_portion=config["input"]["test_portion"],
                               portion_reversed=config["input"]["portion_reversed"],
                               market=config['input']['market'])

    def seed(self, a_seed):
        self._my_seed = a_seed

#    def _seed(self, a_seed):
#        self._my_seed = a_seed
 
    def softmax(self, x):
        xmax = np.max(x)
        e_x = np.exp(x - xmax)
        return e_x / e_x.sum()

    def step(self, action):
        new_omega = action
        self.idx = self.idx + 1
        sample = self.sample()

#        loss6 = -tf.reduce_mean(tf.log(self.pv_vector))
#        pv_vector = tf.reduce_sum(self.__net.output * self.__future_price, reduction_indices=[1]) * (tf.concat([tf.ones(1), self.__pure_pc()], axis=0))
#        future_price = tf.concat([tf.ones([self.__net.input_num, 1]), self.__y[:, 0, :]], 1) #WAT
#        mu = 1 - tf.reduce_sum(tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv), axis=1)
#        cv = self.consumption_vector

#        current_portfolio_value = tf.reduce_sum(new_omega * tf.concat(tf.ones(1), sample[:,0,-1])) # 0 is close, right?
#        print("new_omega = " + str(new_omega) + " np.ones(1) = " + str(np.ones(1)) + " sample[0,:,-1] = " + str(sample[0,:,-1]))
#        prices = np.concatenate(np.ones(1), sample[0,:,-1])
#        prices = np.concatenate((np.ones(1), np.asarray(sample[0,:,-1])))
#        current_portfolio_value = np.dot(new_omega, np.concatenate((np.ones(1), sample[0,:,-1]))) # 0 is close, right?
#        logger.debug("Step " + str(self.idx) + " current_portfolio_value = " + str(current_portfolio_value))
#        logger.debug("latest price vector (v_t) = " + str(sample[0,:,-1]))
#        logger.debug("Previous price vector (v_t-1) = " + str(sample[0,:,-2]))
        price_relative_vector = np.concatenate((np.ones(1), np.divide(sample[0,:,-1], sample[0,:,-2]))) # y_t
        free_portfolio_value_change = np.dot(price_relative_vector, self.omega)
        free_reward = np.log(free_portfolio_value_change)
#        current_portfolio_value = np.dot(new_omega, np.concatenate(np.ones(1), sample[0,:,-1])) # 0 is close, right?
#        previous_portfolio_value = tf.reduce_sum(self.omega * tf.concat(tf.ones(1), sample[:,0,-2]))
#        previous_portfolio_value = np.dot(self.omega, np.concatenate((np.ones(1), sample[0,:,-2])))
#        trading_loss = 1 - tf.reduce_sum(tf.matmul(new_omega[1:]-self.omega[1:]), self.cv), axis=1)
#        traded_volume = np.absolute(new_omega[1:]-self.omega[1:])
#        trading_loss = 1 - np.dot(traded_volume, self.cv)
#        trading_loss = 1 - np.sum(np.matmul(new_omega[1:]-self.omega[1:], self.cv), axis=1)
#        reward = (current_portfolio_value * trading_loss) / previous_portfolio_value - 1
        self.cumulative_free_reward += free_reward
        print ("Step " + str(self.idx) + " -- Free reward = " + str(free_reward) + ". Cumulative free reward = " + str(self.cumulative_free_reward))
        print("price_relative_vector = " + str(price_relative_vector) + " free_portfolio_value_change = " + str(free_portfolio_value_change))

        evolved_old_omega = np.multiply(price_relative_vector, self.omega) / np.dot(price_relative_vector, self.omega) # current omega if no trading was made (w'_t)
        trading_cost = 1. - np.dot(np.absolute(new_omega[1:] - evolved_old_omega[1:]), self.cv) # Linear approximation of mu
        real_portfolio_value_change = free_portfolio_value_change * trading_cost
        print("trading_cost = " + str(trading_cost) + " real_portfolio_value_change = " + str(real_portfolio_value_change))
        reward = np.log(real_portfolio_value_change)
        self.cumulative_reward += reward
        print ("Reward = " + str(reward) + ". Cumulative_reward = " + str(self.cumulative_reward))

        self.omega = new_omega

        ob = (sample, self.omega)
        episode_over = False
        if (self.idx - self.ep_start == self.config["training"]["batch_size"]):
            logger.info("Episode over. Final cumulative free reward is " + str(self.cumulative_free_reward) + ". Cumulative reward is " + str(self.cumulative_reward))
            episode_over = True
        info = {}
        return ob, free_reward, episode_over, info

    def reset(self):
        # Select a starting point on train set
        #  Win Size         Sample from here                                min ep. size      test ep.
        # [=========|====================================================|================][================]
        train_set_size = self.dm._num_train_samples # All minus win size
        self.ep_start = int(np.random.uniform(low=0, high=train_set_size - self.config["training"]["batch_size"]))   # TODO: verify we get indices from end of episode
                                                                                                                     #       (we don't)
                                                                                                                     # TODO: Allow partial episodes at the end
                                                                                                                     # TODO: Exponential instead of uniform
        self.idx = self.ep_start
        self.cumulative_free_reward = 0.
        self.cumulative_reward = 0.
        print("Reset called: self.idx = " + str(self.idx))
        sample = self.sample()
        # softmax a starting omega
        noof = self.config["input"]["coin_number"] + 1
        randy = 20 * np.random.uniform(size=noof)
        self.omega = self.softmax(randy)
#        self.omega = self.softmax(np.random(self.config["input"]["coin_number"]))
        return (sample, self.omega)

    def sample(self):
        # [features, coins, time]
        return self.dm.get_submatrix(self.idx)[:,:,:-1]

    def _render(self, mode='human', close=False):
        pass
