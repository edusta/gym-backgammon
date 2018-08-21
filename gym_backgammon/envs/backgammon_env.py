import gym
import numpy as np

import itertools
import random

from game import Game

from gym.spaces import Box


class BackgammonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    playing_agent = 'x'

    OFF_MOVE = 24
    ON_MOVE = 25

    STEP = 10000

    info = \
    {
        'reward_valid_move': 1,
        'reward_invalid_move': -1,
        'reward_winner': 0,
        'reward_loser': 0,
        'enable_double': False,
    }

    current_step = 0

    def __init__(self):
        '''
        action_space: It's represented with [tokenFirstPlace, tokenSecondPlace] * 2 (or 4, if enable_double).
                    : There can be min of 0 moves and max of 2 (or 4, if enable_double) moves.
                    : 24 is 'off'
                    : 25 is 'on'

        observation_space : Same with Fomorians. Will be detailed later.
        '''

        self.action_count = 8 if BackgammonEnv.info['enable_double'] else 4
        self.current_action = np.asarray([0.0] * self.action_count)
        self.game = Game.new(BackgammonEnv.info)

        self.action_space = Box(low=np.array([0.0] * self.action_count), high=np.array([1.0] * self.action_count), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(153, ), dtype=np.float32)

    def step(self, action):
        assert len(action) == self.action_count
        self.current_action = action
        #self.current_action = np.clip(self.current_action, [0.0] * self.action_count, [23.0] * self.action_count)

        integer_actions = self.current_action.astype(int)

        zipped_action = ((integer_actions[0], integer_actions[1]), (integer_actions[2], integer_actions[3]))
        actual_action_set = list(self.game.get_actions(self.game.last_roll, self.playing_agent))

        picked_action = None
        target_action = actual_action_set[0]

        if zipped_action == target_action:
            picked_action = zipped_action
            print("Picked!", picked_action)
            self._take_action(picked_action)

        if random.uniform(0, 1) < 0.002:
            print action
            print self.current_action
            print target_action
            print (self.game.last_roll)

        ob = self._get_obs()
        episode_over = self.game.is_over()

        if picked_action is not None:
            self.game.play_random()
            reward = self.info['reward_valid_move'] * ob[-1]
            episode_over = True
        else:
            target_action_values = [target_action[0][0] / 25.0, target_action[0][1] / 25.0, target_action[1][0] / 25.0,
                                    target_action[1][1] / 25.0]

            target_action_values = np.asarray(target_action_values)

            distance = np.sum(np.abs(target_action_values - self.current_action))

            ob = np.append(ob, np.asarray(target_action_values - self.current_action))

            #print ob.shape

            reward = self.info['reward_invalid_move'] * distance

        return ob, reward, episode_over, {}

    def reset(self):
        self.game.reset()
        self.current_action = np.asarray([0.0] * self.action_count)

        return np.append(self._get_obs(), self.current_action)

    def render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        self.game.take_action(action, self.playing_agent)

    def _get_reward(self):
        return 1

    def _get_obs(self):
        return self.game.extract_features(self.playing_agent)
