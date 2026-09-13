"""Leaky integrate-and-fire simulation of the whole MaleCNS v1.0 connectome (165k neurons).

Base model follows Shiu et al. 2024 (Nature): identical LIF neurons, each synaptic
contact adds a fixed voltage kick, signed by the presynaptic neuron's transmitter.

Changes needed to keep the whole CNS from locking into runaway activity (it self-sustains
indefinitely with the paper's settings on this dataset):
  * synapse gain lowered 0.275 -> 0.1 mV per contact
  * spike-frequency adaptation (real neurons have it; the base model doesn't)
  * dopamine / serotonin / octopamine treated as slow modulators, not fast synapses
  * Kenyon cell <-> Kenyon cell contacts dropped (dense axo-axonic, not spike-driving)
  * sensory neurons are pure inputs (their axo-axonic inputs dropped)
Time step is 1 ms (paper: 0.1 ms).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

V_REST = -52.0   # mV (also reset)
V_TH = -45.0     # mV
TAU_M = 20.0     # ms
TAU_SYN = 5.0    # ms
TAU_ADAPT = 200.0  # ms
ADAPT_INC = 3.0  # mV per spike
T_REF = 2        # ms refractory
DELAY = 2        # ms synaptic delay
W_SCALE = 0.1    # mV per synaptic contact
DT = 1.0         # ms
W_INPUT = 68.75  # mV per Poisson drive spike (Shiu et al.)

# What a candidate move looks like to the fly -> real visual neuron types (both eyes).
CHANNELS = {
    'goal': ['LC9'],        # move lands on the enemy base
    'capture': ['LC10a'],   # move takes an enemy piece: small-object pursuit neurons
    'progress': ['LC18'],   # move gets closer to the enemy base
    'danger': ['LPLC2'],    # destination can be taken next turn: looming detector
    'base_threat': ['LC4'], # after the move the enemy can take our base: looming/escape
}
APPROACH_TYPES = ['DNa01', 'DNa02', 'DNp09']                 # steering + forward-approach
ESCAPE_TYPES = ['DNp01', 'DNp02', 'DNp04', 'DNp06', 'DNp11']  # giant fiber + looming escape


def load(path='data/'):
    W = sp.load_npz(path + 'brain.npz').tocsc()
    neurons = pd.read_parquet(path + 'neurons.parquet')
    return W, neurons


def stabilise(W, neurons):
    """Apply the connection-level changes listed in the module docstring."""
    kc = neurons.type.fillna('').str.startswith('KC').values
    modulatory = neurons.nt.isin(['dopamine', 'serotonin', 'octopamine']).values
    sensory = neurons.superclass.fillna('').str.contains('sensory').values
    coo = W.tocoo()
    keep = ~(kc[coo.row] & kc[coo.col]) & ~modulatory[coo.col] & ~sensory[coo.row]
    return sp.csc_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=W.shape)


def scramble(W, seed=0):
    """Same neurons, same number/sign/strength of outgoing contacts per neuron, but every
    connection lands on a random target: destroys the wiring diagram, keeps its statistics."""
    rng = np.random.default_rng(seed)
    W = W.tocsc(copy=True)
    W.indices = rng.integers(0, W.shape[0], size=W.nnz).astype(W.indices.dtype)
    W.sum_duplicates()
    return W


class Brain:
    def __init__(self, W, neurons, seed=0):
        self.W = W.tocsc()
        self.n = W.shape[0]
        self.neurons = neurons
        self.rng = np.random.default_rng(seed)

        types = neurons.type.values
        self.inputs = {k: np.flatnonzero(np.isin(types, v)) for k, v in CHANNELS.items()}
        self.approach = np.flatnonzero(np.isin(types, APPROACH_TYPES))
        self.escape = np.flatnonzero(np.isin(types, ESCAPE_TYPES))
        self.descending = np.flatnonzero((neurons.superclass == 'descending_neuron').values)
        self.reset()

    def reset(self):
        self.v = np.full(self.n, V_REST, np.float32)
        self.g = np.zeros(self.n, np.float32)
        self.a = np.zeros(self.n, np.float32)
        self.ref = np.zeros(self.n, np.int16)
        self.queue = [np.zeros(0, np.int64) for _ in range(DELAY)]

    def run(self, drive, ms, rate=150., record=None):
        """Simulate `ms` milliseconds with Poisson drive to the named input groups.

        drive: dict input name -> strength in [0, 1] (scales the 150 Hz max rate).
        record: optional list that receives the spiking neuron indices of every step.
        Returns spike counts per neuron over the window.
        """
        counts = np.zeros(self.n, np.int32)
        dm = np.float32(DT / TAU_M)
        ds = np.float32(np.exp(-DT / TAU_SYN))
        da = np.float32(np.exp(-DT / TAU_ADAPT))
        drives = [(self.inputs[k], s * rate * DT / 1000.) for k, s in drive.items() if s > 0]
        for _ in range(ms):
            delayed = self.queue.pop(0)
            if delayed.size:
                self.g += W_SCALE * np.asarray(self.W[:, delayed].sum(axis=1)).ravel()
            for idx, p in drives:
                self.g[idx[self.rng.random(idx.size) < p]] += W_INPUT
            self.v += ((V_REST - self.v) + self.g - self.a) * dm
            self.g *= ds
            self.a *= da
            refractory = self.ref > 0
            self.v[refractory] = V_REST
            self.ref[refractory] -= 1
            spk = np.flatnonzero(self.v >= V_TH)
            self.v[spk] = V_REST
            self.ref[spk] = T_REF
            self.a[spk] += ADAPT_INC
            counts[spk] += 1
            self.queue.append(spk)
            if record is not None:
                record.append(spk)
        return counts
