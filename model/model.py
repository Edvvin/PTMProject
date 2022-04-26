import torch as pt

from src.model_operations import StateUpdateLayer, StatePoolLayer, unpack_state_features


class Model(pt.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # features encoding models for structures and library
        self.em = pt.nn.Sequential(
            pt.nn.Linear(config['em']['N0'], config['em']['N1']),
            pt.nn.ELU(),
        )
        # atomic level state update model
        self.sum = pt.nn.Sequential(*[StateUpdateLayer(layer_params) for layer_params in config['sum']])

        # atomic to residue reduction layer
        self.spl = StatePoolLayer(config['spl']['N0'], config['spl']['N1'], config['spl']['Nh'])

        # decoding mlp
        self.dm = pt.nn.Sequential(
            pt.nn.Linear(2*config['dm']['N0'], config['dm']['N1']),
            pt.nn.ELU(),
        )
        
        mutm_layers = [
            pt.nn.Linear(config['mutm']['N0'] + config['mutm']['Nmut'], config['mutm']['N1']),
            pt.nn.ELU()
        ]
        
        for i in range(config['mutm']['h']):
            mutm_layers.append(
                pt.nn.Linear(config['mutm']['N1'], config['mutm']['N1'])
            )
            mutm_layers.append(
                pt.nn.ELU()
            )
            
        mutm_layers.append(
            pt.nn.Linear(config['mutm']['N1'], config['mutm']['N2'])
        )
        
        self.mutm = pt.nn.Sequential(*mutm_layers)
        
        self.out = pt.nn.Sequential(
            pt.nn.Linear(config['out']['N0'], config['out']['N0']),
            pt.nn.ELU(),
            pt.nn.Linear(config['out']['N0'], config['out']['N0']),
            pt.nn.ELU(),
            pt.nn.Linear(config['out']['N0'], 1),
        )

    def forward(self, X, ids_topk, q0, M, mut, batch_M):
        # encode features
        q = self.em.forward(q0)

        # initial state vectors
        p0 = pt.zeros((q.shape[0]+1, X.shape[1], q.shape[1]), device=X.device)

        # unpack state features with sink
        q, ids_topk, D_nn, R_nn = unpack_state_features(X, ids_topk, q)

        # atomic tsa layers
        qa, pa, _, _, _ = self.sum.forward((q, p0, ids_topk, D_nn, R_nn))

        # atomic to residue attention pool (without sink)
        qr, pr = self.spl.forward(qa[1:], pa[1:], M)

        # decode state
        zr = pt.cat([qr, pt.norm(pr, dim=1)], dim=1)
        z = self.dm.forward(zr)
        
        zm = pt.cat([z, mut], dim = 1)
        zm = self.mutm.forward(zm)
        agg = zm.unsqueeze(dim=2) * batch_M.unsqueeze(dim = 1)
        agg = agg.mean(dim = 0)
        out = self.out.forward(agg.T)
        out = out.reshape(-1)

        return out
