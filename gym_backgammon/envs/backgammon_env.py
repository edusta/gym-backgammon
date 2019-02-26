import gym
import numpy as np

import itertools
import random

from game import Game

from gym.spaces import Box, Discrete


class BackgammonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    playing_agent = 'x'

    OFF_MOVE = 24
    ON_MOVE = 25

    STEP = 10000

    info = \
    {
        'reward_valid_move': 0.1,
        'reward_invalid_move': -0.01,
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

        self.action_space = Box(low=np.array([0] * self.action_count), high=np.array([1] * self.action_count), dtype=np.float32)
        #self.action_space = Discrete(7)
        self.observation_space = Box(low=0.0, high=1.0, shape=(294, ), dtype=np.float32)

    def toBinary(self, n):
        return ''.join(str(1 & int(n) >> i) for i in range(5)[::-1])

    def step(self, action):
        '''
        action = int(action)
        action_index = action / 2
        action_type = action % 2

        if action_type == 0:
            # Decrease
            self.current_action[action_index] -= 1
        else:
            # Increase
            self.current_action[action_index] += 1
        '''

        '''
        formatted_action = self.toBinary(action)

        assert len(formatted_action) == 5
        formatted_action_list = []

        for i in range(1, len(formatted_action)):
            action_value = int(formatted_action[i])
            if formatted_action[0] == 1:
            # Inverse them
                action_value *= -1
            formatted_action_list.append(action_value)

        #print formatted_action_list

        for i in range(len(formatted_action_list)):
            self.current_action[i] += formatted_action_list[i]

        #print self.current_action
        '''
        self.current_action += action

        #integer_actions = np.abs(self.current_action.astype(int))
        integer_actions = (self.current_action * 25).astype(int)
        '''
        for action in integer_actions:
            if action < 0 or action > 25:
                return np.append(self._get_obs(), self.current_action), -5, True, {}
        '''

        zipped_action = ((integer_actions[0], integer_actions[1]), (integer_actions[2], integer_actions[3]))
        actual_action_set = list(self.game.get_actions(self.game.last_roll, self.playing_agent))

        picked_action = None

        if zipped_action in actual_action_set:
            picked_action = zipped_action
            print("Picked!", picked_action)
            self._take_action(picked_action)

        if random.uniform(0, 1) < 0.001:
            print action
            print self.current_action
            print zipped_action
            print (self.game.last_roll)

        ob = self._get_obs()
        episode_over = self.game.is_over()

        if picked_action is not None:
            self.game.play_random()

        if len(actual_action_set) > 0:
            target_action = actual_action_set[0]

            target_action_values = []
            for i in range(2):
                for j in range(2):
                    try:
                        value = target_action[i][j]

                        if value == "off":
                            value = 25.0 / 25.0
                        elif value == 'on':
                            value = 24.0 / 25.0
                        else:
                            value = value / 25.0
                        target_action_values.append(value)
                    except:
                        target_action_values.append(0)

            difference = np.abs(target_action_values - self.current_action)
            distance = np.sum(difference)

            # print "Current action:", self.current_action
            # print "Difference:", difference
            # print "Distance:", distance

            reward = self.info['reward_invalid_move'] * distance

            if picked_action is not None:
                reward += self.info['reward_valid_move']

        return ob, reward, episode_over, {}

    def reset(self):
        self.game.reset()
        self.current_action = np.asarray([0.0] * self.action_count)
        self.current_step = 0

        return self._get_obs()
        #return np.append(self._get_obs(), self.current_action)

    def render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        self.game.take_action(action, self.playing_agent)

    def _get_reward(self):
        return 1

    def _get_obs(self):
        return self.game.extract_features(self.playing_agent)
