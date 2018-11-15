import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

class PGPEnv(gym.Env):

    def __init__(self):
        pass

    def _step(self, action):
        ob = {}
        reward = 0
        episode_over = False
        info = {}
        return ob, reward, episode_over, info

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass
