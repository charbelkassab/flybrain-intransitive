"""Compress the MaleCNS v1.0 connectome into a signed sparse matrix for simulation.

Keeps traced neurons, signs each connection by the presynaptic neuron's predicted
neurotransmitter (ACh excitatory, GABA/glutamate/histamine inhibitory, others excitatory),
and saves data/brain.npz + data/neurons.parquet.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import scipy.sparse as sp

D = 'data/'
ann = pd.read_feather(D + 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
ann = ann[ann.status == 'Traced'].reset_index(drop=True)
nt = pd.read_feather(D + 'body-neurotransmitters-male-cns-v1.0.feather',
                     columns=['body', 'consensus_nt'])
ann = ann.merge(nt.rename(columns={'body': 'bodyId'}), on='bodyId', how='left')
ann['consensus_nt'] = ann.consensus_nt.fillna('unclear')
sign = np.where(ann.consensus_nt.isin(['gaba', 'glutamate', 'histamine']), -1, 1)

# Soma position for visualisation; sensory neurons have no soma in the CNS.
loc = ann.somaLocation.apply(lambda v: v if v is not None and len(v) == 3 else [np.nan] * 3)
xyz = np.array(loc.tolist(), dtype=float)
neurons = pd.DataFrame({
    'bodyId': ann.bodyId, 'type': ann.type, 'instance': ann.instance,
    'superclass': ann.superclass, 'class': ann['class'],
    'side': ann.rootSide.fillna(ann.somaSide), 'nt': ann.consensus_nt,
    'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
})
print('neurons kept:', len(neurons))

ids = pa.array(neurons.bodyId.values)
index = pd.Series(np.arange(len(neurons)), index=neurons.bodyId.values)
table = feather.read_table(D + 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
                           memory_map=True)
pre_l, post_l, w_l = [], [], []
for batch in table.to_batches(max_chunksize=10_000_000):
    mask = pc.and_(pc.is_in(batch['body_pre'], ids), pc.is_in(batch['body_post'], ids))
    b = batch.filter(mask)
    pre_l.append(index.loc[b['body_pre'].to_numpy()].to_numpy().astype(np.int32))
    post_l.append(index.loc[b['body_post'].to_numpy()].to_numpy().astype(np.int32))
    w_l.append(b['weight'].to_numpy().astype(np.float32))
    print(f'  kept {sum(map(len, w_l)):,} edges', flush=True)
pre, post, w = map(np.concatenate, (pre_l, post_l, w_l))
w *= sign[pre]
# Column = presynaptic neuron, row = postsynaptic: current = W[:, spiking].sum(1).
W = sp.csc_matrix((w, (post, pre)), shape=(len(neurons),) * 2, dtype=np.float32)
sp.save_npz(D + 'brain.npz', W)
neurons.to_parquet(D + 'neurons.parquet')
print('edges', W.nnz, 'synapses', int(np.abs(W.data).sum()))
