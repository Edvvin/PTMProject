# import sys
from datetime import datetime
from src.data_encoding import categ_to_resnames


config_data = {
    'dataset_filepath': "datasets/contacts_rr5A_64nn_8192.h5",
    'train_selection_filepath': {"skempi" : "datasets/skempi_v3_sizes_train.csv",
                                 "wt_pdbs" : "datasets/skempi_pdbs"
                                },
    'test_selection_filepath': {"skempi" : "datasets/skempi_v3_sizes_test.csv",
                                 "wt_pdbs" : "datasets/skempi_pdbs"
                               },
    'max_ba': 1,
    'max_size': 1024*8,
    'min_num_res': 48,
    'max_chain_size' : 3000,
    'l_types': categ_to_resnames['protein'],
    'r_types': [
        categ_to_resnames['protein'],
        categ_to_resnames['dna']+categ_to_resnames['rna'],
        categ_to_resnames['ion'],
        categ_to_resnames['ligand'],
        categ_to_resnames['lipid'],
    ],
    # 'r_types': [[c] for c in categ_to_resnames['protein']],
}

config_model = {
    "em": {'N0': 30, 'N1': 32},
    "sum": [
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
    ],
    "spl": {'N0': 32, 'N1': 32, 'Nh': 4},
    "dm": {'N0': 32, 'N1': 32},
    "mutm" : {'N0':32, 'Nmut':21, 'N1':32, 'N2':32, 'h':2},
    "aggr" : {'N0':32, 'N1':32},
    "out": {'N0': 32}
}

# define run name tag
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'mut2ptm_v0_1_actual_training'+tag,
    'output_dir': 'save',
    'reload': False,
    'device': 'cuda',
    'num_epochs': 100,
    'batch_size': 1,
    'optimizer_step' : 64,
    'log_step': 256,
    'eval_step': 2048,
    'eval_size': 512,
    'learning_rate': 1e-3,
    'pos_weight_factor': 1,
    'dyn_pos_ratio' : True,
    'pos_ratios' : [0.72, 0.28],
    'comment': "Training with higher optimizer step of simplified model",
}
