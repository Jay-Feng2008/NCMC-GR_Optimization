import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BrillLindquistMetric(nn.Module):
    def __init__(self, num_sources, init_masses_presoftplus=None, init_positions=None):
        super().__init__()

        
        if init_masses_presoftplus is not None:
            masses_presoftplus = torch.tensor(init_masses_presoftplus, dtype=torch.float32)
        else:
            masses_presoftplus = torch.abs(torch.rand(num_sources, dtype=torch.float32))

        self.masses_presoftplus = nn.Parameter(masses_presoftplus) 

        if init_positions is not None:
            positions = torch.tensor(init_positions, dtype=torch.float32)
        else:
            positions = torch.rand(num_sources, 3)  

        self.positions = nn.Parameter(positions, requires_grad=False)

    def forward(self, x):
        assert x.shape[-1] == 4, "Each point must be 4D"
        x1 = x[..., 1]
        x2 = x[..., 2]
        x3 = x[..., 3]

        points_spatial = torch.stack((x1, x2, x3), dim=-1)  
        diffs = points_spatial.unsqueeze(-2) - self.positions  
        r = torch.norm(diffs, dim=-1)  

        r = r + (r == 0).float() * 1e-8

        masses = torch.nn.functional.softplus(self.masses_presoftplus)
        psi = 1 + torch.sum(masses / (2 * r), dim=-1)  
        psi4 = psi ** 4  

        shape = x.shape[:-1] + (4, 4)  
        g = torch.zeros(shape, dtype=x.dtype, device=x.device)

        g[..., 0, 0] = -1.0
        g[..., 1, 1] = psi4
        g[..., 2, 2] = psi4
        g[..., 3, 3] = psi4

        return g
    

class BrillLindquistChristfSym(nn.Module):
    def __init__(self, metric_field):
        super().__init__()

        self.metric_field = metric_field

    def forward(self, x):
        B, n = x.shape
        x_dtype = x.dtype
        x_device = x.device
        C_batch = torch.zeros(B, n, n, n, dtype=x_dtype, device=x_device)

        eps = 1e-4 

        for b in range(B):
            position = x[b]  
            G = self.metric_field(position)  
            G_inv = torch.inverse(G)

            dG = torch.zeros(n, n, n, dtype=x_dtype, device=x_device)

            for k in range(n):
                delta = torch.zeros_like(position)
                delta[k] = eps

                pos_plus = position + delta
                pos_minus = position - delta

                G_plus = self.metric_field(pos_plus)
                G_minus = self.metric_field(pos_minus)

                G_diff = (G_plus - G_minus) / (2 * eps)  
                dG[:, :, k] = G_diff  

            C = torch.zeros(n, n, n, dtype=x_dtype, device=x_device)
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        sum_term = 0.0
                        for l in range(n):
                            term = dG[i, l, j] + dG[j, l, i] - dG[i, j, l]
                            sum_term += G_inv[k, l] * term
                        C[k, i, j] = 0.5 * sum_term

            C_batch[b] = C

        return C_batch
    



class NeuralODEfunc(nn.Module):
    def __init__(self, positions, masses_presoftplus):
        super().__init__()
        self.metric_field = BrillLindquistMetric(num_sources=len(masses_presoftplus), 
                                                 init_masses_presoftplus=masses_presoftplus, init_positions=positions)
        self.christfSym_field = BrillLindquistChristfSym(self.metric_field)

    def forward(self, t, x):
        x, v = x        
        C = self.christfSym_field(x)
        dv = -torch.einsum('Blmn,Bm,Bn->Bl', C, v, v)
        dx = v
        return torch.stack([dx, dv])
    

        

if __name__ == '__main__':
    # Tests here...
    ...
