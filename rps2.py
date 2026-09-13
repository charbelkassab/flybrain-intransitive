"""Intransitive (meaf.us/rps2) rules, ported from the site's own client code.

9x9 board, files a-i, ranks 1-9. Blue base a1, red base i9. Blue moves first.
Pieces move one square in any direction (chess king). A piece may move to an empty
square or capture an enemy piece it beats: rock > scissors > paper > rock.
Win by moving onto the enemy base; a side with no legal move loses.
The server-side no-capture draw limit isn't in the client; STAGNATION_PLIES is our assumption.
"""

import numpy as np

FILES = 'abcdefghi'
BASE = {'blue': (0, 0), 'red': (8, 8)}
BEATS = {'R': 'S', 'S': 'P', 'P': 'R'}
SETUP_BLUE = {'R': ['b4', 'c3', 'd2'], 'P': ['b5', 'c4', 'd3', 'e2'], 'S': ['c5', 'd4', 'e3']}
STAGNATION_PLIES = 200
FEATURES = ['goal', 'capture', 'progress', 'danger', 'base_threat']


def sq(name):
    return FILES.index(name[0]), int(name[1:]) - 1


def name(p):
    return f'{FILES[p[0]]}{p[1] + 1}'


def other(player):
    return 'red' if player == 'blue' else 'blue'


def dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class Game:
    def __init__(self):
        self.board = {}
        for t, squares in SETUP_BLUE.items():
            for s in squares:
                x, y = sq(s)
                self.board[(x, y)] = ('blue', t)
                self.board[(8 - y, 8 - x)] = ('red', t)  # site mirrors across the anti-diagonal
        self.turn = 'blue'
        self.ply = 0
        self.quiet = 0
        self.winner = None
        self.reason = None
        self.history = []

    def moves_from(self, p, board=None):
        board = self.board if board is None else board
        player, t = board[p]
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = (p[0] + dx, p[1] + dy)
                if (dx or dy) and 0 <= q[0] < 9 and 0 <= q[1] < 9:
                    target = board.get(q)
                    if target is None or (target[0] != player and BEATS[t] == target[1]):
                        out.append((p, q))
        return out

    def legal_moves(self, player=None, board=None):
        board = self.board if board is None else board
        player = self.turn if player is None else player
        return [m for p, (pl, _) in board.items() if pl == player for m in self.moves_from(p, board)]

    def features(self, move):
        """What this move 'looks like' from the mover's point of view (all binary)."""
        p, q = move
        me, t = self.board[p]
        foe = other(me)
        goal = q == BASE[foe]
        after = dict(self.board)
        del after[p]
        after[q] = (me, t)
        danger = False
        if not goal:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    e = after.get((q[0] + dx, q[1] + dy))
                    if e and e[0] == foe and BEATS[e[1]] == t:
                        danger = True
        base_threat = not goal and any(m[1] == BASE[me] for m in self.legal_moves(foe, after))
        return {
            'goal': float(goal),
            'capture': float(q in self.board),
            'progress': float(dist(q, BASE[foe]) < dist(p, BASE[foe])),
            'danger': float(danger),
            'base_threat': float(base_threat),
        }

    def play(self, move):
        p, q = move
        captured = self.board.get(q)
        self.history.append((self.turn, move, self.board[p][1], captured))
        self.board[q] = self.board.pop(p)
        self.ply += 1
        self.quiet = 0 if captured else self.quiet + 1
        if q == BASE[other(self.turn)]:
            self.winner, self.reason = self.turn, 'base capture'
            return
        self.turn = other(self.turn)
        if not self.legal_moves():
            self.winner, self.reason = other(self.turn), 'no legal moves'
        elif self.quiet >= STAGNATION_PLIES:
            self.winner, self.reason = 'draw', 'stagnation'

    @property
    def over(self):
        return self.winner is not None


# Reference evaluation the learned readout imitates (and the "greedy" opponent uses).
REFERENCE_VALUE = {'goal': 1000., 'base_threat': -500., 'capture': 10., 'danger': -8., 'progress': 2.}


def reference_value(f):
    return sum(REFERENCE_VALUE[k] * f[k] for k in FEATURES)


def pick_best(values, rng):
    values = np.asarray(values, float)
    best = np.flatnonzero(values >= values.max() - 1e-9)
    return int(rng.choice(best))
