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
        'reward_invalid_move': -0.01,
        'reward_winner': 0,
        'reward_loser': 0,
        'enable_double': False,
        'is_action_scaled': False
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
        self.is_action_scaled = BackgammonEnv.info['is_action_scaled']

        self.game = Game.new(BackgammonEnv.info)

        self.action_space = Box(low=np.array([0.0] * self.action_count), high=np.array([1.0] * self.action_count), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(149, ), dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        #action = action[0]

        assert len(action) == self.action_count

        parsed_action_list = []

        for token_place in action:
            if self.is_action_scaled:
                token_place = token_place
            else:
                # Convert to int
                # Clip the actions
                token_place = int(token_place * 23)
                #token_place = 0 if token_place < 0 else token_place
                #token_place = 25 if token_place > 25 else token_place

            if token_place < 0 or token_place > 23:
                continue

            parsed_action_list.append(token_place)

            '''
            if token_place == BackgammonEnv.OFF_MOVE:
                parsed_action_list.append('off')
            elif token_place == BackgammonEnv.ON_MOVE:
                parsed_action_list.append('on')
            else:
            '''
        if len(parsed_action_list) != self.action_count:
            ob = self._get_obs()
            episode_over = self.game.is_over()
            reward = self.info['reward_invalid_move']

            return ob, reward, episode_over or self.current_step == self.STEP, {}

        printed = False
        if random.uniform(0, 1) < 0.005:
            printed = True
            print (action)
            print (parsed_action_list)

        zipped_action = ((parsed_action_list[0], parsed_action_list[1]), (parsed_action_list[2], parsed_action_list[3]))
        #zipped_actions = list(itertools.combinations(parsed_action_list, 2))
        actual_action_set = self.game.get_actions(self.game.last_roll, self.playing_agent)
        #print "hey", zipped_actions

        '''
        for action in zipped_actions:
            first_action = action[0]
            second_action = action[1]

            if first_action == 'off' or first_action == 'on' or second_action == 'off' or second_action == 'on':
                continue

            diff = abs(int(first_action) - int(second_action))
            if diff > 6 or diff == 0:
                zipped_actions.remove(action)
        '''
        if printed:
            print(self.game.last_roll)
            #print actual_action_set

        '''
        constructed_actions = []

        for i in range(0, self.action_count / 2 + 1):  # Cross product all moves possible (min of 0, max of action_count/ 2)
            constructed_actions += list(itertools.product(zipped_actions, repeat=i))

        if len(actual_action_set) == 0:
            # No possible moves.
            #print("No possible moves. Skipping...")
            reward = 0
            self.game.play_random()
        else:
            # There are some possible moves.
            #picked_action = list(actual_action_set)[action % len(actual_action_set)]
            #picked_action = random.choice(list(actual_action_set))

            picked_action = None

            for constructed_action in constructed_actions:


            if picked_action is not None:
                self._take_action(picked_action)
            else:
                # No moves done, hence the state didn't change.
                reward = self.info['reward_invalid_move']

        '''

        picked_action = None

        if zipped_action in actual_action_set:
            picked_action = zipped_action
            print("Picked!", picked_action)
            self._take_action(picked_action)

        ob = self._get_obs()
        episode_over = self.game.is_over() or self.current_step == self.STEP

        if picked_action is not None:
            self.game.play_random()
            reward = self.info['reward_valid_move'] * ob[-1]
            #print ob[-1]
        else:
            reward = self.info['reward_invalid_move']


        '''
        if episode_over:
            winner = self.game.winner()
            print("Episode over! Winner: ", winner)
            if winner == self.playing_agent:
                reward = self.info['reward_winner']
            else:
                reward = self.info['reward_loser']
        '''

        return ob, reward, episode_over, {}

    def reset(self):
        self.game.reset()
        self.current_step = 0
        return self._get_obs()

    def render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        self.game.take_action(action, self.playing_agent)

    def _get_reward(self):
        return 1

    def _get_obs(self):
        return self.game.extract_features(self.playing_agent)
