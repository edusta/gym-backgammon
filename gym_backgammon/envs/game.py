import copy
import random
import numpy as np


class Game:
    LAYOUT = "0-2-o,5-5-x,7-3-x,11-5-o,12-5-x,16-3-o,18-5-o,23-2-x"
    NUMCOLS = 24
    QUAD = 6
    OFF = 'off'
    ON = 'on'
    TOKENS = ['x', 'o']

    current_win = 0.0

    def __init__(self, layout=LAYOUT, grid=None, off_pieces=None, bar_pieces=None, num_pieces=None, players=['x', 'o'], enable_double=True):
        """
        Define a new game object
        """
        self.die = Game.QUAD
        self.layout = layout
        self.enable_double = enable_double

        self.players = Game.TOKENS

    @staticmethod
    def new(info):
        game = Game(enable_double=info['enable_double'])
        game.reset()
        return game

    # Original
    def extract_features(self, player):
        features = []

        for p in self.players:
            for col in self.grid:
                feats = [0.] * 6
                if len(col) > 0 and col[0] == p:
                    for i in range(len(col)):
                        feats[min(i, 5)] += 0.2
                features += feats
            features.append(float(len(self.bar_pieces[p])) / 10.0)
            features.append(float(len(self.off_pieces[p])) / self.num_pieces[p])

        # Adding the rolled dice to the features
        features += [self.last_roll[0] / 6.0, self.last_roll[1] / 6.0]

        # Adding current round
        # features += [self.current_win / 10.0]

        return np.array(features)

    def roll_dice(self):
        firstRoll = random.randint(1, self.die)
        secondRoll = random.randint(1, self.die)

        if not self.enable_double:
            while firstRoll == secondRoll:
                secondRoll = random.randint(1, self.die)

        return (firstRoll, secondRoll)

    def play(self, players, draw=False):
        player_num = random.randint(0, 1)
        while not self.is_over():
            self.next_step(players[player_num], player_num, draw=draw)
            player_num = (player_num + 1) % 2

        return self.winner()

    def play_random(self):
        random_token = 'o'
        self.last_roll = self.roll_dice()
        moves = self.get_actions(self.last_roll, random_token, nodups=False)
        move = random.choice(list(moves)) if moves else None

        if move:
            self.take_action(move, random_token)

        # Return to first player
        self.last_roll = self.roll_dice()

    def clone(self):
        """
        Return an exact copy of the game. Changes can be made
        to the cloned version without affecting the original.
        """
        return Game(None, self.grid, self.off_pieces,
                    self.bar_pieces, self.num_pieces, self.players)

    def take_action(self, action, token):
        #print self.last_roll, token, "has played", action
        """
        Makes given move for player, assumes move is valid,
        will remove pieces from play
        """

        self.current_win += 1

        ateList = [0] * 4
        for i, (s, e) in enumerate(action):
            if s == Game.ON:
                piece = self.bar_pieces[token].pop()
            else:
                piece = self.grid[s].pop()
            if e == Game.OFF:
                self.off_pieces[token].append(piece)
                continue
            if len(self.grid[e]) > 0 and self.grid[e][0] != token:
                bar_piece = self.grid[e].pop()
                self.bar_pieces[bar_piece].append(bar_piece)
                ateList[i] = 1
            self.grid[e].append(piece)
        return ateList

    def undo_action(self, action, player, ateList):
        """
        Reverses given move for player, assumes move is valid,
        will remove pieces from play
        """
        for i, (s, e) in enumerate(reversed(action)):
            if e == Game.OFF:
                piece = self.off_pieces[player].pop()
            else:
                piece = self.grid[e].pop()
                if ateList[len(action) - 1 - i]:
                    bar_piece = self.bar_pieces[self.opponent(player)].pop()
                    self.grid[e].append(bar_piece)
            if s == Game.ON:
                self.bar_pieces[player].append(piece)
            else:
                self.grid[s].append(piece)

    def get_actions(self, roll, player, nodups=False):
        """
        Get set of all possible move tuples
        """
        moves = set()
        if nodups:
            start = 0
        else:
            start = None

        r1, r2 = roll
        if r1 == r2:  # doubles
            i = 4
            # keep trying until we find some moves
            while not moves and i > 0:
                self.find_moves(tuple([r1] * i), player, (), moves, start)
                i -= 1
        else:
            self.find_moves(roll, player, (), moves, start)
            self.find_moves((r2, r1), player, (), moves, start)
            # has no moves, try moving only one piece
            if not moves:
                for r in roll:
                    self.find_moves((r,), player, (), moves, start)

        return moves

    def find_moves(self, rs, player, move, moves, start=None):
        if len(rs) == 0:
            moves.add(move)
            return
        r, rs = rs[0], rs[1:]
        # see if we can remove a piece from the bar
        if self.bar_pieces[player]:
            if self.can_onboard(player, r):
                if player == self.players[0]:
                    piece = self.bar_pieces[player].pop()
                    bar_piece = None
                    if len(self.grid[24 - r]) == 1 and self.grid[24 - r][-1] != player:
                        bar_piece = self.grid[24 - r].pop()

                    self.grid[24 - r].append(piece)

                    self.find_moves(rs, player, move + ((Game.ON, 24 - r),), moves, start)
                    self.grid[24 - r].pop()
                    self.bar_pieces[player].append(piece)
                    if bar_piece:
                        self.grid[24 - r].append(bar_piece)
                elif player == self.players[1]:
                    piece = self.bar_pieces[player].pop()
                    bar_piece = None
                    if len(self.grid[r - 1]) == 1 and self.grid[r - 1][-1] != player:
                        bar_piece = self.grid[r - 1].pop()

                    self.grid[r - 1].append(piece)

                    self.find_moves(rs, player, move + ((Game.ON, r - 1),), moves, start)
                    self.grid[r - 1].pop()
                    self.bar_pieces[player].append(piece)
                    if bar_piece:
                        self.grid[r - 1].append(bar_piece)
            return

        # otherwise check each grid location for valid move using r
        offboarding = self.can_offboard(player)

        for i in range(len(self.grid)):
            if start is not None:
                start = i

            if player == self.players[0]:
                if self.is_valid_move(i, i - r, player):
                    piece = self.grid[i].pop()
                    bar_piece = None
                    if len(self.grid[i - r]) == 1 and self.grid[i - r][-1] != player:
                        bar_piece = self.grid[i - r].pop()
                    self.grid[i - r].append(piece)
                    self.find_moves(rs, player, move + ((i, i - r),), moves, start)
                    self.grid[i - r].pop()
                    self.grid[i].append(piece)
                    if bar_piece:
                        self.grid[i - r].append(bar_piece)

            elif player == self.players[1]:
                if self.is_valid_move(i, i + r, player):
                    piece = self.grid[i].pop()
                    bar_piece = None
                    if len(self.grid[i + r]) == 1 and self.grid[i + r][-1] != player:
                        bar_piece = self.grid[i + r].pop()
                    self.grid[i + r].append(piece)
                    self.find_moves(rs, player, move + ((i, i + r),), moves, start)
                    self.grid[i + r].pop()
                    self.grid[i].append(piece)
                    if bar_piece:
                        self.grid[i + r].append(bar_piece)

            # If we can't move on the board can we take the piece off?
            if offboarding and self.remove_piece(player, i, r):
                piece = self.grid[i].pop()
                self.off_pieces[player].append(piece)
                self.find_moves(rs, player, move + ((i, Game.OFF),), moves, start)
                self.off_pieces[player].pop()
                self.grid[i].append(piece)

    def opponent(self, token):
        """
        Retrieve opponent players token for a given players token.
        """
        for t in self.players:
            if t != token:
                return t

    def is_won(self, player):
        """
        If game is over and player won, return True, else return False
        """
        return self.is_over() and player == self.players[self.winner()]

    def is_lost(self, player):
        """
        If game is over and player lost, return True, else return False
        """
        return self.is_over() and player != self.players[self.winner()]

    def reverse(self):
        """
        Reverses a game allowing it to be seen by the opponent
        from the same perspective
        """
        self.grid.reverse()
        self.players.reverse()

    def reset(self):
        """
        Resets game to original layout.
        """
        self.current_win = 0
        random.seed(1992)

        self.grid = [[] for _ in range(Game.NUMCOLS)]

        for col in self.layout.split(','):
            loc, num, token = col.split('-')
            self.grid[int(loc)] = [token for _ in range(int(num))]

        self.off_pieces = {}
        self.bar_pieces = {}
        self.num_pieces = {}

        for t in self.players:
            self.bar_pieces[t] = []
            self.off_pieces[t] = []
            self.num_pieces[t] = 15

        self.last_roll = self.roll_dice()

    def winner(self):
        """
        Get winner.
        """
        return self.players[0] if len(self.off_pieces[self.players[0]]) == self.num_pieces[self.players[0]] else self.players[1]

    def is_over(self):
        """
        Checks if the game is over.
        """
        for t in self.players:
            if len(self.off_pieces[t]) == self.num_pieces[t]:
                return True
        return False

    def can_offboard(self, player):
        if player == self.players[0]:  # x-x-x-x-x
            count = 0
            for i in range(0, 6):  # 0'dan 5'e.
                if len(self.grid[i]) > 0 and self.grid[i][0] == player:
                    count += len(self.grid[i])
            if count + len(self.off_pieces[player]) == self.num_pieces[player]:
                return True
            return False
        elif player == self.players[1]:  # o-o-o-o-o
            count = 0
            for i in range(Game.NUMCOLS - self.die, Game.NUMCOLS):  # 18'den 23'e.
                if len(self.grid[i]) > 0 and self.grid[i][0] == player:
                    count += len(self.grid[i])
            if count + len(self.off_pieces[player]) == self.num_pieces[player]:
                return True
            return False

    def can_onboard(self, player, r):
        """
        Can we take a players piece on the bar to a position
        on the grid given by roll-1?
        """
        if player == self.players[0]:  # x-x-x-x-x
            if len(self.grid[24 - r]) <= 1 or self.grid[24 - r][0] == player:
                return True
            else:
                return False
        elif player == self.players[1]:  # o-o-o-o-o
            if len(self.grid[r - 1]) <= 1 or self.grid[r - 1][0] == player:
                return True
            else:
                return False

    def remove_piece(self, player, start, r):
        """
        Can we remove a piece from location start with roll r ?
        In this function we assume we are cool to offboard,
        i.e. no pieces on the bar and all are in the home quadrant.
        """
        if player == self.players[0]:  # x-x-x-x-x
            if start > 5:
                return False
            if len(self.grid[start]) == 0 or self.grid[start][0] != player:
                return False
            if start - r == 0:
                return True
            if start - r < 0:
                for i in range(start + 1, 6, +1):
                    if len(self.grid[i]) != 0 and self.grid[i][0] == player:
                        return False
                return True
            return False
        elif player == self.players[1]:  # o-o-o-o-o
            if start < Game.NUMCOLS - self.die:
                return False
            if len(self.grid[start]) == 0 or self.grid[start][0] != player:
                return False
            if start + r == Game.NUMCOLS:
                return True
            if start + r > Game.NUMCOLS:
                for i in range(start - 1, Game.NUMCOLS - self.die, -1):
                    if len(self.grid[i]) != 0 and self.grid[i][0] == player:
                        return False
                return True
            return False

    def is_valid_move(self, start, end, token):
        if len(self.grid[start]) > 0 and self.grid[start][0] == token:
            if end < 0 or end >= len(self.grid):
                return False
            if len(self.grid[end]) <= 1:
                return True
            if len(self.grid[end]) > 1 and self.grid[end][-1] == token:
                return True
        return False