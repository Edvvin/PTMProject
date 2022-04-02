import h5py
import numpy as np
import torch as pt


from src.dataset import load_sparse_mask, collate_batch_features


def load_interface_labels(hgrp, t0, t1_l):
    # load stored data
    shape = tuple(hgrp.attrs['Y_shape'])
    ids = pt.from_numpy(np.array(hgrp['Y']).astype(np.int64))

    # matching contacts type for receptor and ligand
    y_ctc_r = pt.any((ids[:,2].view(-1,1) == t0), dim=1).view(-1,1)
    y_ctc_l = pt.stack([pt.any((ids[:,3].view(-1,1) == t1), dim=1) for t1 in t1_l], dim=1)
    y_ctc = (y_ctc_r & y_ctc_l)

    # save contacts of right type
    y = pt.zeros((shape[0], len(t1_l)), dtype=pt.bool)
    y[ids[:,0], pt.where(y_ctc)[1]] = True

    return y


def collate_batch_data(batch_data):
    # collate features
    X, ids_topk, q, M, mut, batch_M = collate_batch_features(batch_data)

    # collate targets
    y = pt.tensor([data[5] for data in batch_data])

    return X, ids_topk, q, M, mut, batch_M, y
