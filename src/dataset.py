import numpy as np
import torch as pt
import pandas as pd
import re
import os

from .structure_io import read_pdb
from .structure import clean_structure, concatenate_chains, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits
from .data_encoding import std_reschar, encode_features, encode_structure, extract_topology


def select_by_sid(dataset, sids_sel):
    # extract sids of dataset
    sids = np.array(['_'.join([s.split(':')[0] for s in key.split('/')[1::2]]) for key in dataset.keys])

    # create selection mask
    m = np.isin(sids, sids_sel)

    return m


def select_by_max_ba(dataset, max_ba):
    # extract aids of dataset
    aids = np.array([int(key.split('/')[2]) for key in dataset.keys])

    # create selection mask
    m = (aids <= max_ba)

    return m


def select_complete_assemblies(dataset, m):
    # get non-selected subunits
    rmkeys = np.unique(dataset.keys[~m])

    # select all assemblies not containing non-selected subunits
    return ~np.isin(dataset.rkeys, rmkeys)


def select_by_interface_types(dataset, l_types, r_types):
    # get types id
    t0 = np.where(np.isin(dataset.mids, l_types))[0]
    t1 = np.where(np.isin(dataset.mids, r_types))[0]

    # ctypes selection mask
    cm = (np.isin(dataset.ctypes[:,1], t0) & np.isin(dataset.ctypes[:,2], t1))

    # apply selection on dataset
    m = np.isin(np.arange(dataset.keys.shape[0]), dataset.ctypes[cm,0])

    return m


def load_sparse_mask(hgrp, k):
    # get shape
    shape = tuple(hgrp.attrs[k+'_shape'])

    # create map
    M = pt.zeros(shape, dtype=pt.float)
    ids = pt.from_numpy(np.array(hgrp[k]).astype(np.int64))
    M.scatter_(1, ids[:,1:], 1.0)

    return M


def save_data(hgrp, attrs={}, **data):
    # store data
    for key in data:
        hgrp.create_dataset(key, data=data[key], compression="lzf")

    # save attributes
    for key in attrs:
        hgrp.attrs[key] = attrs[key]


def load_data(hgrp, keys=None):
    # define keys
    if keys is None:
        keys = hgrp.keys()

    # load data
    data = {}
    for key in keys:
        # read data
        data[key] = np.array(hgrp[key])

    # load attributes
    attrs = {}
    for key in hgrp.attrs:
        attrs[key] = hgrp.attrs[key]

    return data, attrs


def collate_batch_features(batch_data, max_num_nn=64):
    # pack coordinates and charges
    X = pt.cat([data[0] for data in batch_data], dim=0)
    q = pt.cat([data[2] for data in batch_data], dim=0)
    mut = pt.cat([data[4] for data in batch_data], dim=0)

    # extract sizes
    sizes = pt.tensor([data[3].shape for data in batch_data])
    mut_size = sum([data[4].shape[0] for data in batch_data])

    # pack nearest neighbors indices and residues masks
    ids_topk = pt.zeros((X.shape[0], max_num_nn), dtype=pt.long, device=X.device)
    M = pt.zeros(pt.Size(pt.sum(sizes, dim=0)), dtype=pt.float, device=X.device)
    batch_M = pt.zeros((mut.shape[0], len(batch_data)), dtype=pt.float, device=X.device)
    
    batch_elm_num = 0
    for size, data in zip(pt.cumsum(sizes, dim=0), batch_data):
        # get indices of slice location
        ix1 = size[0]
        ix0 = ix1-data[3].shape[0]
        iy1 = size[1]
        iy0 = iy1-data[3].shape[1]
        
        # store data
        ids_topk[ix0:ix1, :data[1].shape[1]] = data[1]+ix0+1
        M[ix0:ix1,iy0:iy1] = data[3]
        mut[iy0:iy1] = data[4]
        batch_M[iy0:iy1, batch_elm_num] = pt.ones((iy1-iy0,))
        batch_elm_num += 1
        

    return X, ids_topk, q, M, mut, batch_M

def parse_skempi_entry(entry):
    
    # parse muts
    muts = []
    for mut in entry['Mutation(s)_cleaned'].split(','):
        aa_old, su, pos, aa_new = re.match(r'([A-Z])([A-Z])([0-9]+)([A-Z])', mut).groups()
        muts.append({'su': su, 'pos': int(pos) , 'wt_aa': aa_old , 'mut_aa': aa_new })
        
    parse = {
        'wt_pdb' : re.match(r'([A-Z0-9]+)_.+', entry['#Pdb']).group(1),
        'muts' : muts
    }
    
    return parse


class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, wtpdb_dir, skempi_path):
        super(StructuresDataset).__init__()
        # store dataset filepath
        self.wtpdb_dir = wtpdb_dir
        skempi = pd.read_csv(skempi_path, sep=';')
        self.skempi = skempi[['#Pdb', 'Mutation(s)_cleaned',
                         'iMutation_Location(s)', 'Affinity_mut_parsed',
                         'Affinity_wt_parsed']]

    def __len__(self):
        return len(self.skempi)

    def __getitem__(self, i):
        # find pdb filepath
        
        mut = self.skempi.iloc[i]
        mut_parse = parse_skempi_entry(mut)
        pdb_filepath = os.path.join(self.wtpdb_dir, mut_parse['wt_pdb'] + ".pdb")
        

        # parse pdb
        try:
            structure = read_pdb(pdb_filepath)
        except Exception as e:
            print(f"ReadError: {pdb_filepath}: {e}")
            return None, None, None, None, None, None

        # label
        deltaG = mut['Affinity_mut_parsed'] - mut['Affinity_wt_parsed']
        label = (deltaG >= 0).astype(np.float32)
        label = pt.tensor(label)
        

        # process structure
        structure = clean_structure(structure)

        # update molecules chains
        structure = tag_hetatm_chains(structure)

        # split structure
        subunits = split_by_chain(structure)

        # remove non atomic structures
        subunits = filter_non_atomic_subunits(subunits)

        # remove duplicated molecules and ions
        subunits = remove_duplicate_tagged_subunits(subunits)
        

        # generate residue mutation onehot array
        for su_name, su in subunits.items():
            mut_onehot = np.zeros((len(np.unique(su['resid'])), 20))
            for mu in mut_parse['muts']:
                if mu['su'] == su_name[:-2]:
                    mut_onehot[mu['pos'] - 1] = (std_reschar == mu['mut_aa'])
            mut_onehot = np.concatenate([mut_onehot, ~np.any(mut_onehot, axis=1).reshape(-1,1)], axis=1)
            su['mut_onehot'] = mut_onehot
        
        structure = concatenate_chains(subunits)
        X, M = encode_structure(structure)
        q = pt.cat(encode_features(structure), dim=1)
        q = encode_features(structure)[0]
        ids_topk, _, _, _, _ = extract_topology(X, 64)
        mut = pt.tensor(structure['mut_onehot'], dtype=pt.torch.float32)

        return X, ids_topk, q, M, mut, label
