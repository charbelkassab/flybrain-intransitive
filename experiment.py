"""Can the fly connectome play Intransitive (rock-paper-scissors chess)?

For every legal move, the fly "sees" the move: its binary features stimulate real visual
neurons for STIM_MS, and the whole CNS responds. The fly plays the move its brain rates best:

  * zero training: approach neurons (DNa01, DNa02, DNp09) minus escape neurons
    (giant fiber DNp01, DNp02, DNp04, DNp06, DNp11)
  * readout: linear readout of all 1,314 descending neurons, fit to a reference evaluation

Brain responses for each of the 32 feature patterns are recorded REPEATS times; readouts are
fit on half and games sample the other half. Same for a scrambled-wiring brain.
"""

import itertools
import json
import os
import time

import numpy as np

from brain import Brain, load, scramble, stabilise
from rps2 import FEATURES, Game, pick_best, reference_value

STIM_MS = 80
REPEATS = 12
GAMES = 100


def pattern(f):
    return tuple(int(f[k]) for k in FEATURES)


def record_bank(brain):
    bank = {}
    for key in itertools.product([0, 1], repeat=len(FEATURES)):
        drive = dict(zip(FEATURES, key))
        dn, val = [], []
        for _ in range(REPEATS):
            brain.reset()
            c = brain.run(drive, STIM_MS)
            dn.append(c[brain.descending])
            val.append(c[brain.approach].sum() - c[brain.escape].sum())
        bank[key] = {'dn': np.array(dn), 'wired': np.array(val, float)}
    return bank


class Readout:
    """Ridge regression from log descending-neuron counts to move value."""

    def fit(self, X, y, lam=1.0):
        X = np.log1p(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.ym = y.mean()
        self.w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - self.ym))
        return self

    def predict(self, X):
        Z = (np.log1p(np.atleast_2d(X)) - self.mu) / self.sd
        return Z @ self.w + self.ym


def fit_readout(bank):
    X = np.concatenate([v['dn'][:REPEATS // 2] for v in bank.values()])
    y = np.concatenate([[reference_value(dict(zip(FEATURES, k)))] * (REPEATS // 2) for k in bank])
    r = Readout().fit(X, y)
    Xt = np.concatenate([v['dn'][REPEATS // 2:] for v in bank.values()])
    yt = np.concatenate([[reference_value(dict(zip(FEATURES, k)))] * (REPEATS - REPEATS // 2) for k in bank])
    pred = r.predict(Xt)
    r2 = 1 - ((pred - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    return r, float(r2)


def brain_player(bank, mode, readout=None):
    def choose(game, rng):
        moves = game.legal_moves()
        vals = []
        for m in moves:
            entry = bank[pattern(game.features(m))]
            i = REPEATS // 2 + rng.integers(REPEATS - REPEATS // 2)  # held-out responses
            vals.append(entry['wired'][i] if mode == 'wired' else readout.predict(entry['dn'][i])[0])
        return moves[pick_best(vals, rng)]
    return choose


def random_player(game, rng):
    moves = game.legal_moves()
    return moves[rng.integers(len(moves))]


def greedy_player(game, rng):
    moves = game.legal_moves()
    return moves[pick_best([reference_value(game.features(m)) for m in moves], rng)]


def match(p_name, player, opponent, rng):
    tally = {'win': 0, 'draw': 0, 'loss': 0}
    reasons = {}
    for g_i in range(GAMES):
        fly_color = 'blue' if g_i % 2 == 0 else 'red'
        g = Game()
        while not g.over and g.ply < 1000:
            mover = player if g.turn == fly_color else opponent
            g.play(mover(g, rng))
        result = 'draw' if g.winner in (None, 'draw') else 'win' if g.winner == fly_color else 'loss'
        tally[result] += 1
        reasons[g.reason] = reasons.get(g.reason, 0) + 1
    return tally, reasons


def main():
    os.makedirs('results', exist_ok=True)
    rng = np.random.default_rng(0)
    W, neurons = load()
    W_fly = stabilise(W, neurons)
    brains = {'fly': Brain(W_fly, neurons, seed=1),
              'scrambled': Brain(scramble(W_fly, seed=0), neurons, seed=1)}

    players, results = {}, {'patterns': {}, 'matches': {}}
    for name, b in brains.items():
        t = time.time()
        bank = record_bank(b)
        print(f'{name}: {len(bank) * REPEATS} brain runs in {time.time() - t:.0f}s', flush=True)
        readout, r2 = fit_readout(bank)
        np.savez(f'results/readout_{name}.npz', w=readout.w, mu=readout.mu, sd=readout.sd, ym=readout.ym)
        results[f'{name}_readout_r2_heldout'] = r2
        players[f'{name} brain, zero training'] = brain_player(bank, 'wired')
        players[f'{name} brain + readout'] = brain_player(bank, 'readout', readout)
        for f in FEATURES:  # single-feature responses
            key = tuple(int(k == f) for k in FEATURES)
            e = bank[key]
            results['patterns'][f'{name}:{f}'] = {
                'approach_minus_escape': float(e['wired'].mean()),
                'descending_active': int(np.count_nonzero(e['dn'].mean(0))),
            }
        print(f'  readout held-out R^2 {r2:.3f}; single-feature approach-escape:',
              {f: round(results['patterns'][f'{name}:{f}']['approach_minus_escape'], 1) for f in FEATURES})

    players['random moves'] = random_player
    players['greedy reference rule (no brain)'] = greedy_player

    for opp_name, opp in [('random', random_player), ('greedy', greedy_player)]:
        for name, p in players.items():
            t = time.time()
            tally, reasons = match(name, p, opp, rng)
            results['matches'][f'{name} vs {opp_name}'] = {**tally, 'reasons': reasons}
            print(f'{name:38s} vs {opp_name:7s} W{tally["win"]:3d} D{tally["draw"]:3d} L{tally["loss"]:3d}  '
                  f'{reasons}  ({time.time() - t:.0f}s)', flush=True)
    json.dump(results, open('results/scoreboard.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
