"""Render videos/fly_plays_intransitive.mp4: the zero-training fly brain (blue) vs the greedy bot (red).

For every legal move the fly's eyes are stimulated with what that move looks like, the whole
connectome runs for 80 ms, and the move's value is approach neurons minus escape neurons.
The video sweeps through those evaluations, then plays the move the brain liked best.
"""

import json
import os
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from brain import Brain, load, stabilise
from experiment import STIM_MS, greedy_player
from rps2 import BASE, FEATURES, FILES, Game, pick_best

W_VID, H_VID, FPS = 1920, 1080, 30
BG, PANEL, GRID = (9, 12, 18), (17, 22, 32), (30, 38, 52)
TEXT, MUTED = (228, 232, 240), (132, 142, 160)
BLUE, RED = (75, 111, 217), (217, 75, 75)
GREEN, LOOM, CYAN, GOLD, AMBER = (92, 220, 120), (255, 90, 90), (70, 210, 255), (244, 197, 66), (255, 176, 64)
FONT = '/System/Library/Fonts/HelveticaNeue.ttc'

FEATURE_LABELS = {
    'goal': ('lands on enemy base', 'LC9', GREEN),
    'capture': ('takes a piece', 'LC10a', GREEN),
    'progress': ('closer to enemy base', 'LC18', GREEN),
    'danger': ('can be taken next turn', 'LPLC2', LOOM),
    'base_threat': ('leaves our base open', 'LC4', LOOM),
}


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


class BrainMap:
    """Dorsal view of the CNS from soma coordinates, with decaying spike glow."""

    def __init__(self, neurons, brain, box):
        self.x0, self.y0, self.w, self.h = box
        ok = neurons[['x', 'z']].notna().all(axis=1).values
        x, z = neurons.x.values, neurons.z.values
        xr, zr = np.nanmax(x) - np.nanmin(x), np.nanmax(z) - np.nanmin(z)
        scale = min(self.w, self.h * xr / zr)
        wpx, hpx = scale, scale * zr / xr
        ox, oy = (self.w - wpx) / 2, (self.h - hpx) / 2
        self.px = np.where(ok, ox + (np.nanmax(x) - x) / xr * (wpx - 1), -1).astype(int)
        self.py = np.where(ok, oy + (z - np.nanmin(z)) / zr * (hpx - 1), -1).astype(int)
        self.ok = ok
        col = np.tile(np.array(AMBER, np.float32) / 255, (len(neurons), 1))
        for k, idx in brain.inputs.items():
            col[idx] = np.array(FEATURE_LABELS[k][2]) / 255
        col[brain.approach] = np.array(CYAN) / 255
        col[brain.escape] = np.array((255, 80, 200)) / 255
        self.col = col
        self.weight = np.ones(len(neurons), np.float32)
        self.weight[brain.approach] = 6
        self.weight[brain.escape] = 6
        base = np.zeros((self.h, self.w), np.float32)
        np.add.at(base, (self.py[ok], self.px[ok]), 1)
        base = np.clip(gaussian_filter(base, 0.7) / np.percentile(base[base > 0], 99), 0, 1) ** 0.6
        self.base = base[..., None] * np.array([40, 52, 72], np.float32)
        self.glow = np.zeros((self.h, self.w, 3), np.float32)

    def add(self, idx, counts, decay=0.45):
        self.glow *= decay
        keep = self.ok[idx]
        idx, counts = idx[keep], counts[keep].astype(np.float32)
        for c in range(3):
            np.add.at(self.glow[..., c], (self.py[idx], self.px[idx]),
                      self.col[idx, c] * self.weight[idx] * np.minimum(counts, 4) * 1.2)

    def image(self):
        g = gaussian_filter(self.glow, (1.3, 1.3, 0))
        return Image.fromarray(np.clip(self.base + 255 * (1 - np.exp(-1.6 * g)), 0, 255).astype(np.uint8))


class Renderer:
    def __init__(self, neurons, brain):
        self.map = BrainMap(neurons, brain, (1110, 150, 470, 780))
        self.bx, self.by, self.cell = 110, 110, 96
        self.icons = {}
        for t, fname in [('R', 'rock'), ('P', 'paper'), ('S', 'scissors')]:
            m = Image.open(f'assets/{fname}.png').split()[-1]
            s = int(self.cell * 0.74)
            m.thumbnail((s, s), Image.LANCZOS)
            for player, col in [('blue', BLUE), ('red', RED)]:
                icon = Image.new('RGBA', m.size, col + (0,))
                icon.putalpha(m)
                self.icons[(player, t)] = icon
        fly = Image.open('assets/fly_top.png').rotate(90, expand=True)
        fly.thumbnail((70, 70), Image.LANCZOS)
        self.fly = fly
        self.f = {s: font(s) for s in (17, 21, 26)}
        self.fb = {s: font(s, True) for s in (26, 34, 40, 52)}

    def cell_box(self, p):
        x = self.bx + p[0] * self.cell
        y = self.by + (8 - p[1]) * self.cell
        return x, y

    def center(self, p):
        x, y = self.cell_box(p)
        return x + self.cell / 2, y + self.cell / 2

    def draw_board(self, img, d, game, last=None, arrows=(), tints=None):
        c = self.cell
        d.rounded_rectangle([self.bx - 6, self.by - 6, self.bx + 9 * c + 6, self.by + 9 * c + 6], 8, fill=(51, 51, 51))
        for i in range(9):
            for j in range(9):
                x, y = self.cell_box((i, j))
                fill = (255, 255, 255)
                if (i, j) == BASE['blue']:
                    fill = (160, 182, 240)
                elif (i, j) == BASE['red']:
                    fill = (234, 128, 128)
                if last and (i, j) in last:
                    fill = (255, 247, 206)
                if tints and (i, j) in tints:
                    fill = tints[(i, j)]
                d.rectangle([x + 1, y + 1, x + c - 2, y + c - 2], fill=fill)
        for i, f in enumerate(FILES):
            d.text((self.bx + i * c + c / 2 - 6, self.by + 9 * c + 14), f, font=self.f[21], fill=MUTED)
            d.text((self.bx - 34, self.by + (8 - i) * c + c / 2 - 12), str(i + 1), font=self.f[21], fill=MUTED)
        for (p, t), icon in [((p, v), self.icons[v]) for p, v in game.board.items()]:
            x, y = self.cell_box(p)
            img.alpha_composite(icon, (int(x + (c - icon.width) / 2), int(y + (c - icon.height) / 2)))
        for frm, to, col, width in arrows:
            (x1, y1), (x2, y2) = self.center(frm), self.center(to)
            d.line([x1, y1, x2, y2], fill=col, width=width)
            r = width * 1.6
            d.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=col)

    def panel(self, d, st):
        x = 1620
        d.text((x, 150), 'WHAT THE FLY SEES', font=self.fb[26], fill=MUTED)
        for i, k in enumerate(FEATURES):
            lab, neuron, col = FEATURE_LABELS[k]
            y = 195 + i * 50
            on = st and st['features'][k] > 0
            d.ellipse([x, y + 5, x + 20, y + 25], fill=col if on else GRID)
            d.text((x + 32, y), lab, font=self.f[21], fill=TEXT if on else MUTED)
            d.text((x + 32, y + 24), f'{neuron} neurons', font=self.f[17], fill=MUTED)
        d.text((x, 470), 'BRAIN OUTPUT', font=self.fb[26], fill=MUTED)
        a = st['approach'] if st else 0
        e = st['escape'] if st else 0
        for i, (lab, v, col) in enumerate([('approach  DNa01 DNa02 DNp09', a, CYAN),
                                           ('escape  giant fiber DNp01 + 4 more', e, (255, 80, 200))]):
            y = 515 + i * 70
            d.text((x, y), f'{lab}', font=self.f[17], fill=MUTED)
            d.text((x + 240, y + 24), str(int(v)), font=self.fb[26], fill=TEXT)
            d.rounded_rectangle([x, y + 28, x + 230, y + 50], 6, fill=GRID)
            wv = int(230 * min(1, v / 250))
            if wv > 4:
                d.rounded_rectangle([x, y + 28, x + wv, y + 50], 6, fill=col)
        d.text((x, 670), 'MOVE VALUE', font=self.fb[26], fill=MUTED)
        if st:
            v = st['approach'] - st['escape']
            d.text((x, 705), f'{v:+.0f}', font=self.fb[52], fill=GREEN if v > 0 else LOOM if v < 0 else TEXT)
            d.text((x, 775), st.get('note', ''), font=self.f[21], fill=MUTED)

    def frame(self, game, title, sub, st=None, **board_kw):
        img = Image.new('RGBA', (W_VID, H_VID), BG + (255,))
        d = ImageDraw.Draw(img)
        self.draw_board(img, d, game, **board_kw)
        img.alpha_composite(self.fly, (110, 18))
        d.text((190, 26), 'Blue: fruit fly connectome (zero training)', font=self.fb[26], fill=BLUE)
        d.text((190, 62), 'Red: greedy bot', font=self.f[21], fill=RED)
        d.text((1110, 40), title, font=self.fb[40], fill=TEXT)
        d.text((1110, 94), sub, font=self.f[21], fill=MUTED)
        img.paste(self.map.image(), (self.map.x0, self.map.y0))
        ly = self.map.y0 + self.map.h + 12
        lx = self.map.x0
        for lab, col in [('eyes: appetitive', GREEN), ('eyes: looming', LOOM), ('approach', CYAN),
                         ('escape', (255, 80, 200))]:
            d.ellipse([lx, ly + 5, lx + 12, ly + 17], fill=col)
            d.text((lx + 18, ly), lab, font=self.f[17], fill=MUTED)
            lx += 34 + d.textlength(lab, font=self.f[17])
        d.text((self.map.x0, ly + 26), 'brain (top) and nerve cord (bottom), real neuron positions',
               font=self.f[17], fill=MUTED)
        self.panel(d, st)
        return np.array(img.convert('RGB'))


def value_color(v):
    if v > 0:
        return (120, 220, 140)
    if v < 0:
        return (240, 110, 110)
    return (190, 196, 210)


def tint_color(v):
    return (205, 240, 212) if v > 0 else (250, 208, 208) if v < 0 else (232, 234, 240)


def play_live(brain, seed):
    rng = np.random.default_rng(seed)
    g = Game()
    turns = []
    while not g.over and g.ply < 400:
        snapshot = Game()
        snapshot.board, snapshot.turn, snapshot.ply = dict(g.board), g.turn, g.ply
        if g.turn == 'blue':
            evals = []
            for m in g.legal_moves():
                f = g.features(m)
                brain.reset()
                c = brain.run(f, STIM_MS)
                idx = np.flatnonzero(c)
                evals.append({'move': m, 'features': f, 'idx': idx, 'counts': c[idx],
                              'approach': int(c[brain.approach].sum()), 'escape': int(c[brain.escape].sum())})
            choice = pick_best([e['approach'] - e['escape'] for e in evals], rng)
            move = evals[choice]['move']
            turns.append({'player': 'blue', 'board': snapshot, 'evals': evals, 'choice': choice, 'move': move})
        else:
            move = greedy_player(g, rng)
            turns.append({'player': 'red', 'board': snapshot, 'move': move})
        g.play(move)
    return g, turns


def card(lines, sizes, colors, extra=None):
    img = Image.new('RGB', (W_VID, H_VID), BG)
    d = ImageDraw.Draw(img)
    y = H_VID // 2 - sum(s + 22 for s in sizes) // 2
    for text, s, col in zip(lines, sizes, colors):
        f = font(s, s >= 40)
        d.text(((W_VID - d.textlength(text, font=f)) / 2, y), text, font=f, fill=col)
        y += s + 22
    if extra:
        extra(d)
    return np.array(img)


def main():
    os.makedirs('videos', exist_ok=True)
    W, neurons = load()
    brain = Brain(stabilise(W, neurons), neurons, seed=3)
    board = json.load(open('results/scoreboard.json'))
    r = Renderer(neurons, brain)

    print('choosing a game...')
    chosen = None
    for seed in range(12):
        t = time.time()
        g, turns = play_live(brain, seed)
        n_fly = sum(tt['player'] == 'blue' for tt in turns)
        print(f'  seed {seed}: winner {g.winner} by {g.reason}, {n_fly} fly moves ({time.time() - t:.0f}s)', flush=True)
        if g.winner == 'blue' and 10 <= n_fly <= 35:
            chosen = (g, turns)
            break
        if chosen is None:
            chosen = (g, turns)
    g_final, turns = chosen

    out = 'videos/fly_plays_intransitive.mp4'
    writer = imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=1)

    def hold(frame, seconds):
        for _ in range(int(FPS * seconds)):
            writer.append_data(frame)

    hold(card(['A fruit fly brain plays Intransitive',
               'rock-paper-scissors chess from meaf.us/rps2',
               f'MaleCNS connectome  ·  {len(neurons):,} neurons  ·  simulated live  ·  zero training'],
              [64, 30, 28], [TEXT, AMBER, MUTED]), 4)
    hold(card(['How the fly chooses a move',
               'Every legal move is shown to the fly\'s eyes as what it would mean:',
               'take a piece / get closer / win  =  object-chasing visual neurons (LC10a, LC18, LC9)',
               'could be taken / leaves the base open  =  looming detectors (LPLC2, LC4)',
               'The whole nervous system runs 80 ms per move. The fly plays the move that drives',
               'its approach neurons most and its escape neurons (the giant fiber) least.',
               'No training, no learned weights: just the wiring.'],
              [52, 28, 28, 28, 28, 28, 30], [TEXT, MUTED, GREEN, LOOM, MUTED, MUTED, CYAN]), 8)

    for ti, turn in enumerate(turns):
        game = turn['board']
        last = None
        if ti > 0:
            last = set(turns[ti - 1]['move'])
        if turn['player'] == 'red':
            for _ in range(10):
                r.map.add(np.zeros(0, int), np.zeros(0))
                writer.append_data(r.frame(game, 'Red is thinking', 'greedy bot, one move ahead', last=last))
            continue
        evals = turn['evals']
        n = len(evals)
        for k, e in enumerate(evals):
            v = e['approach'] - e['escape']
            r.map.add(e['idx'], e['counts'])
            quiet = not any(e['features'].values())
            st = dict(e, note=f'move {k + 1} of {n}' + ('\nnothing notable: eyes quiet' if quiet else ''))
            writer.append_data(r.frame(game, 'The fly is looking at every move',
                                       f'{n} legal moves · 80 ms of brain activity each', st,
                                       last=last, arrows=[(e['move'][0], e['move'][1], value_color(v), 8)]))
        best = {}
        for e in evals:
            v = e['approach'] - e['escape']
            best[e['move'][1]] = max(best.get(e['move'][1], -1e9), v)
        tints = {p: tint_color(v) for p, v in best.items()}
        ch = evals[turn['choice']]
        st = dict(ch, note='chosen: highest value')
        for _ in range(14):
            r.map.add(np.zeros(0, int), np.zeros(0), decay=0.9)
            writer.append_data(r.frame(game, 'Decision', 'green = brain approaches · red = brain escapes', st,
                                       last=last, tints=tints, arrows=[(ch['move'][0], ch['move'][1], GOLD, 12)]))

    end_game = Game()
    end_game.board = g_final.board
    result = ('The fly wins' if g_final.winner == 'blue' else 'The bot wins' if g_final.winner == 'red'
              else 'Draw') + f' by {g_final.reason}'
    for i in range(int(FPS * 3)):
        r.map.add(np.zeros(0, int), np.zeros(0))
        f = Image.fromarray(r.frame(end_game, result, f'{g_final.ply} moves', last=set(turns[-1]['move'])))
        d = ImageDraw.Draw(f)
        fnt = font(44, True)
        w = d.textlength(result, font=fnt)
        cx = 110 + 9 * 96 / 2
        d.rounded_rectangle([cx - w / 2 - 30, 500, cx + w / 2 + 30, 580], 16, fill=PANEL)
        d.text((cx - w / 2, 514), result, font=fnt, fill=GOLD)
        writer.append_data(np.array(f))

    m = board['matches']
    rows = [('fly wiring, zero training', 'fly brain, zero training', BLUE),
            ('scrambled wiring, zero training', 'scrambled brain, zero training', MUTED),
            ('random moves', 'random moves', MUTED),
            ('greedy bot (reference, no brain)', 'greedy reference rule (no brain)', (90, 96, 110))]

    def chart(d):
        x0, wmax = 820, 700
        for block, (opp, y0) in enumerate([('random', 210), ('greedy', 520)]):
            d.text((x0 - 420, y0 - 50), f'wins out of 100 vs {opp} bot', font=font(30, True), fill=TEXT)
            for i, (lab, key, col) in enumerate(rows):
                y = y0 + i * 58
                v = m[f'{key} vs {opp}']['win']
                f = font(24)
                d.text((x0 - 20 - d.textlength(lab, font=f), y + 6), lab, font=f, fill=TEXT)
                d.rounded_rectangle([x0, y, x0 + max(8, int(wmax * v / 100)), y + 38], 8, fill=col)
                d.text((x0 + max(8, int(wmax * v / 100)) + 14, y + 4), str(v), font=font(26, True), fill=TEXT)

    hold(card(['', '', '', '', '', '', '', '', '', '', '', '',
               'Scrambled = same neurons and synapse counts, connections rewired at random: eye signals never reach',
               'approach or escape neurons, so it plays like random. The fly wiring matches the greedy bot without training.'],
              [28] * 14, [MUTED] * 14, extra=chart), 9)
    writer.close()
    print('wrote', out)


if __name__ == '__main__':
    main()
