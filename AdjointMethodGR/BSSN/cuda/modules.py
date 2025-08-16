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
import string
import gc, os, shutil


    



class NeuralODEfunc(nn.Module):
    def __init__(self, BH_positions, BH_masses_presoftplus, BH_masses=None, res=64, max_x=1, CF_factor=0.5,
                 device='cpu'):
        super().__init__()
        self.device = device
        positions = torch.tensor(BH_positions, dtype=torch.float32, requires_grad=True, device=self.device)
        masses_presoftplus = torch.tensor(BH_masses_presoftplus, dtype=torch.float32, requires_grad=True, device=self.device)
        self.res = res
        self.max_x = max_x
        self.dx = torch.tensor(self.max_x / (self.res / 2 - 1), requires_grad=True, device=self.device)

        self.damping = torch.tensor([2.], requires_grad=True, device=self.device) # Eta
        self.lambdas = torch.tensor([1., 1, 1, 1], requires_grad=True, device=self.device)
        self.epsilon = torch.tensor([1e-5], requires_grad=True, device=self.device)    # For avoiding /0 of ln(0)

        self.BH_masses_presoftplus = nn.Parameter(masses_presoftplus)
        if BH_masses is not None:
            masses = torch.tensor(BH_masses, dtype=torch.float32, requires_grad=True, device=self.device)
            self.BH_masses = nn.Parameter(masses)
        else:
            self.BH_masses = None
        self.BH_positions = nn.Parameter(positions)

        


        # Gauge
        # self.alpha = torch.zeros((self.res, self.res, self.res), requires_grad=True, device=self.device)
        # self.beta = torch.zeros((self.res, self.res, self.res, 3), requires_grad=True, device=self.device)
        # self.B = torch.zeros((self.res, self.res, self.res, 3), requires_grad=True, device=self.device)

        # Metric
        # self.chi = torch.zeros((self.res, self.res, self.res), requires_grad=True, device=self.device)
        # self.phi = torch.zeros((self.res, self.res, self.res), requires_grad=True, device=self.device)
        # self.gamma_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)
        # self.gamma_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)
        # self.gamma_bar_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)
        # self.gamma_bar_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)

        # Connection Coefficients
        # self.Gamma_bar = torch.zeros((self.res, self.res, self.res, 3), requires_grad=True, device=self.device)
        # self.Gamma_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3, 3), requires_grad=True, device=self.device)
        # self.Gamma_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3, 3), requires_grad=True, device=self.device)
        # self.Gamma_bar_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3, 3), requires_grad=True, device=self.device)
        # self.Gamma_bar_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3, 3), requires_grad=True, device=self.device)

        # Extrinsic Curvature
        # self.A_bar_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)
        # self.A_bar_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)
        # self.A_TT_bar_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)
        # self.A_L_bar_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)

        # self.K = torch.zeros((self.res, self.res, self.res), requires_grad=True, device=self.device)
        # self.K_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)

        # Ricci tensor
        # self.R_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)
        # self.R_bar_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)
        # self.R_phi_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3), requires_grad=True, device=self.device)

        # # Stress Energy
        # self.rho = torch.zeros((self.res, self.res, self.res))
        # self.S = torch.zeros((self.res, self.res, self.res, 3))
        # self.S_mat_lo = torch.zeros((self.res, self.res, self.res, 3, 3))    # Not evolving to save memory and time


        self.x = torch.linspace(self.dx * (1 - self.res/2), self.dx * (self.res/2 - 1), self.res, requires_grad=True, device=self.device)
        self.y = torch.linspace(self.dx * (1 - self.res/2), self.dx * (self.res/2 - 1), self.res, requires_grad=True, device=self.device)
        self.z = torch.linspace(self.dx * (1 - self.res/2), self.dx * (self.res/2 - 1), self.res, requires_grad=True, device=self.device)

        self.X, self.Y, self.Z = torch.meshgrid(self.x, self.y, self.z, indexing="xy")

        self.CF_factor = torch.tensor(CF_factor, device=self.device)
        self.CF_dt = self.CF_factor * self.dx

        


    def init_non_evolve_vars(self, phi):
        print("Preparing Variables")
        start_time = time.time()
        gamma_bar_mat_lo = self.calc_gamma_bar_mat_lo()
        gamma_bar_mat_up = self.calc_gamma_bar_mat_up(gamma_bar_mat_lo)
        gamma_mat_lo = self.calc_gamma_mat_lo(phi, gamma_bar_mat_lo)
        gamma_mat_up = self.calc_gamma_mat_up(phi, gamma_bar_mat_up)
        # print(f"Calculating the metric took: {time.time() - start_time} s")

        start_time = time.time()
        Gamma_mat_up = self.calc_Gamma_mat_up(gamma_mat_lo, gamma_mat_up)
        Gamma_mat_lo = self.calc_Gamma_mat_lo(gamma_mat_lo, Gamma_mat_up)
        Gamma_bar_mat_up = self.calc_Gamma_bar_mat_up(gamma_bar_mat_lo, gamma_bar_mat_up)
        Gamma_bar_mat_lo = self.calc_Gamma_bar_mat_lo(gamma_bar_mat_lo, Gamma_bar_mat_up)
        Gamma_bar = self.calc_Gamma_bar(gamma_bar_mat_up)
        # print(f"Calculating the connection coefficients took: {time.time() - start_time} s")

        start_time = time.time()
        A_L_bar_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3), device=self.device)
        A_TT_bar_mat_up = torch.zeros((self.res, self.res, self.res, 3, 3), device=self.device)
        K = torch.zeros((self.res, self.res, self.res), device=self.device)
        A_bar_mat_up, A_bar_mat_lo = self.calc_A_bar_mat_lo(A_L_bar_mat_up, A_TT_bar_mat_up, gamma_bar_mat_lo)
        A_bar_mat_up = self.calc_A_bar_mat_up(gamma_bar_mat_up, A_bar_mat_lo)
        K_mat_lo = self.calc_K_mat_lo(phi, A_bar_mat_lo, gamma_mat_lo, K)
        # print(f"Calculating the extrinsic curvature took: {time.time() - start_time} s")

        start_time = time.time()
        R_bar_mat_lo = self.calc_R_bar_mat_lo(Gamma_bar, Gamma_bar_mat_lo, Gamma_bar_mat_up,
                                                   gamma_bar_mat_lo, gamma_bar_mat_up)
        R_phi_mat_lo = self.calc_R_phi_mat_lo(phi, Gamma_bar_mat_up, gamma_bar_mat_lo, 
                                                   gamma_bar_mat_up)
        R_mat_lo = self.calc_R_mat_lo(R_bar_mat_lo, R_phi_mat_lo)
        # print(f"Calculating the Ricci tensor took: {time.time() - start_time} s")

        return (gamma_bar_mat_lo, gamma_bar_mat_up, gamma_mat_lo, gamma_mat_up, 
                Gamma_mat_up, Gamma_mat_lo, Gamma_bar_mat_up, Gamma_bar_mat_lo, Gamma_bar,
                K, A_bar_mat_lo, A_bar_mat_up, K_mat_lo,
                R_bar_mat_lo, R_phi_mat_lo, R_mat_lo)


    def BrillLindquist_init(self):
        # Brill-Lindquist initial data.
        psi = 1.0
        if self.BH_masses is not None:
            BH_masses = self.BH_masses
        else:
            BH_masses = F.softplus(self.BH_masses_presoftplus)

        print("BH_masses stats: min, max, mean:", BH_masses.min().item(), BH_masses.max().item(), BH_masses.mean().item())
        for loc, mass in zip(self.BH_positions, BH_masses):
            psi += mass/(2 * torch.sqrt((self.X - loc[0])**2 + (self.Y - loc[1])**2 + (self.Z - loc[2])**2))

        phi = torch.log(psi)
        chi = torch.exp(-4 * phi)
        alpha = torch.exp(-2 * phi)                 

        return phi, chi, alpha


    def one_spatial_deriv(self, func):
        """
        Calculate the spatial derivation d_i.

        Returns a tensor with the shape of (*func.shape, 3)
        """

        diff = torch.stack(torch.gradient(func, spacing=[self.x, self.y, self.z], dim=[0,1,2], edge_order=1), dim=0)

        n_indices = len(diff.shape)-3
        labels = string.ascii_lowercase[:n_indices]

        diff_ordered = torch.einsum(f"{labels[-1]}xyz{labels[:n_indices-1]}->xyz{labels}",diff)

        return diff_ordered


    def two_spatial_deriv(self, func):
        """
        Calculate the spatial derivation d_i d_j.

        Returns a tensor with the shape of (*func.shape, 3, 3)
        """

        diff = torch.stack(torch.gradient(self.one_spatial_deriv(func), spacing=[self.x, self.y, self.z], dim=[0,1,2], edge_order=1), dim=0)

        n_indices = len(diff.shape)-3
        labels = string.ascii_lowercase[:n_indices]

        diff_ordered = torch.einsum(f"{labels[-1]}xyz{labels[:n_indices-1]}->xyz{labels}",diff)

        return diff_ordered
    

    def lopsided_one_spatial_deriv(self, func, side):
        """
        Calculate the lopsided spatial derivation d_i.

        The forward scheme computes F_{i+3,j,k} - ... and
        the backward scheme computes -F_{i-3,j,k} + ...

        Returns a tensor with the shape of (*func.shape, 3)
        """

        diff = torch.zeros((*func.shape, 3), device=self.device)

        if side == "forward":
            roll = [-1, -2, -3, 1]
            idx = [0, -1, -2, -3]
            coeff = 1
        elif side == "backward":
            roll = [1, 2, 3, -1]
            idx = [1, 2, 3, -1]
            coeff = -1

        for j in range(3):
            func_s1 = torch.roll(func, roll[0], dims=j) # Shift 1
            func_s2 = torch.roll(func, roll[1], dims=j) # Shift 2
            func_s3 = torch.roll(func, roll[2], dims=j) # Shift 3
            func_o1 = torch.roll(func, roll[3], dims=j) # Other shift 1

            diff[...,j] = coeff * 1/(12 * self.dx) * (func_s3 - 6 * func_s2 + 18 * func_s1 - 10 * func - 3 * func_o1)

            other_1 = coeff * 1/self.dx * (func_s1 - func)
            shift_1 = coeff * torch.zeros_like(other_1, device=self.device)
            shift_2 = coeff * 1/self.dx * (func_s1 - func)
            shift_3 = coeff * 1/(2 * self.dx) * (-func_s2 + 4 * func_s1 - 3 * func)

            if j == 0:
                diff[idx[0], :, :, ..., j] = other_1[idx[0], :, :, ...]
                diff[idx[1], :, :, ..., j] = shift_1[idx[1], :, :, ...]
                diff[idx[2], :, :, ..., j] = shift_2[idx[2], :, :, ...]
                diff[idx[3], :, :, ..., j] = shift_3[idx[3], :, :, ...]
            elif j == 1:
                diff[:, idx[0], :, ..., j] = other_1[:, idx[0], :, ...]
                diff[:, idx[1], :, ..., j] = shift_1[:, idx[1], :, ...]
                diff[:, idx[2], :, ..., j] = shift_2[:, idx[2], :, ...]
                diff[:, idx[3], :, ..., j] = shift_3[:, idx[3], :, ...]
            elif j == 2:
                diff[:, :, idx[0], ..., j] = other_1[:, :, idx[0], ...]
                diff[:, :, idx[1], ..., j] = shift_1[:, :, idx[1], ...]
                diff[:, :, idx[2], ..., j] = shift_2[:, :, idx[2], ...]
                diff[:, :, idx[3], ..., j] = shift_3[:, :, idx[3], ...]

        return diff

    def advection_one_spatial_deriv(self, func, beta):
        """
        Masaru Shibata - Numerical Relativity Section 3.2 Handling advection terms.
        """
        diff = self.one_spatial_deriv(func)
        diff_forward = self.lopsided_one_spatial_deriv(func, side="forward")
        diff_backward = self.lopsided_one_spatial_deriv(func, side="backward")

        # Advection part
        diff_advection = torch.zeros_like(diff, device=self.device)
        for i in range(3):
            diff_advection[...,i][beta[...,i] <= 0] = diff_backward[...,i][beta[...,i] <= 0]
            diff_advection[...,i][beta[...,i] >= 0] = diff_forward[...,i][beta[...,i] >= 0]
            diff_advection[...,i][beta[...,i] == 0] = diff[...,i][beta[...,i] == 0]

        return diff_advection
    

    def calc_gamma_bar_mat_lo(self):
        """
        This calculation will only be performed once for the initial data
        because it's part of the evolution d/dt gamma_bar_mat_lo.
        """
        minkowski = torch.tensor([[1., 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]], device=self.device)

        gamma_bar_mat_lo = minkowski.repeat(self.res, self.res, self.res, 1, 1)
        return gamma_bar_mat_lo

    def calc_gamma_bar_mat_up(self, gamma_bar_mat_lo):
        # self.gamma_bar_mat_up[:] = torch.linalg.inv(self.gamma_bar_mat_lo)
        return torch.linalg.inv(gamma_bar_mat_lo)

    def calc_gamma_mat_lo(self, phi, gamma_bar_mat_lo):
        # self.gamma_mat_lo[:] = torch.einsum('xyz,xyzij->xyzij', torch.exp(4*self.phi), self.gamma_bar_mat_lo)
        return torch.einsum('xyz,xyzij->xyzij', torch.exp(4*phi), gamma_bar_mat_lo)

    def calc_gamma_mat_up(self, phi, gamma_bar_mat_up):
        # self.gamma_mat_up[:] = torch.einsum('xyz,xyzij->xyzij', torch.exp(-4*self.phi), self.gamma_bar_mat_up)
        return torch.einsum('xyz,xyzij->xyzij', torch.exp(-4*phi), gamma_bar_mat_up)

    def calc_Gamma_mat_up(self, gamma_mat_lo, gamma_mat_up):
        diff_gamma_mat_lo = self.one_spatial_deriv(gamma_mat_lo)

        summation_1 = torch.einsum('xyzil,xyzjlk->xyzijk',gamma_mat_up,diff_gamma_mat_lo)
        summation_2 = torch.einsum('xyzil,xyzlkj->xyzijk',gamma_mat_up,diff_gamma_mat_lo)
        summation_3 = torch.einsum('xyzil,xyzjkl->xyzijk',gamma_mat_up,diff_gamma_mat_lo)

        return 1/2 * (summation_1 + summation_2 - summation_3)

    def calc_Gamma_mat_lo(self, gamma_mat_lo, Gamma_mat_up):
        return torch.einsum('xyzil,xyzljk->xyzijk', gamma_mat_lo, Gamma_mat_up)

    def calc_Gamma_bar_mat_up(self, gamma_bar_mat_lo, gamma_bar_mat_up):
        diff_gamma_bar_mat_lo = self.one_spatial_deriv(gamma_bar_mat_lo)

        summation_1 = torch.einsum('xyzil,xyzjlk->xyzijk', gamma_bar_mat_up, diff_gamma_bar_mat_lo)
        summation_2 = torch.einsum('xyzil,xyzlkj->xyzijk', gamma_bar_mat_up, diff_gamma_bar_mat_lo)
        summation_3 = torch.einsum('xyzil,xyzjkl->xyzijk', gamma_bar_mat_up, diff_gamma_bar_mat_lo)

        return 1/2 * (summation_1 + summation_2 - summation_3)

    def calc_Gamma_bar_mat_lo(self, gamma_bar_mat_lo, Gamma_bar_mat_up):
        return torch.einsum('xyzil,xyzljk->xyzijk', gamma_bar_mat_lo, Gamma_bar_mat_up)

    def calc_Gamma_bar(self, gamma_bar_mat_up):
        """
        This calculation will only be performed once for the initial data
        because it's part of the evolution d/dt Gamma_bar.

        This also uses the assumption that det(gamma_bar_mat_lo) = 1.
        """

        return -torch.einsum("xyzijj->xyzi", self.one_spatial_deriv(gamma_bar_mat_up))


    def calc_A_bar_mat_lo(self, A_L_bar_mat_up, A_TT_bar_mat_up, gamma_bar_mat_lo):
        """
        This calculation will only be performed once for the initial data
        because it's part of the evolution d/dt A_bar_mat_lo.

        A_bar_mat_up is calculated here for initial data sake.
        """
        A_bar_mat_up = A_L_bar_mat_up + A_TT_bar_mat_up
        A_bar_mat_lo = torch.einsum('xyzjl,xyzik,xyzkl->xyzij', gamma_bar_mat_lo, gamma_bar_mat_lo, A_bar_mat_up)
        return A_bar_mat_up, A_bar_mat_lo

    def calc_A_bar_mat_up(self, gamma_bar_mat_up, A_bar_mat_lo):
        return torch.einsum('xyzjl,xyzik,xyzkl->xyzij', gamma_bar_mat_up, gamma_bar_mat_up, A_bar_mat_lo)

    def calc_K_mat_lo(self, phi, A_bar_mat_lo, gamma_mat_lo, K):
        return torch.einsum('xyz,xyzij->xyzij', torch.exp(4*phi), A_bar_mat_lo) + 1/3 * torch.einsum('xyzij,xyz->xyzij', gamma_mat_lo, K)

    def calc_R_bar_mat_lo(self, Gamma_bar, Gamma_bar_mat_lo, Gamma_bar_mat_up, gamma_bar_mat_lo, gamma_bar_mat_up):
        diff_Gamma_bar = self.one_spatial_deriv(Gamma_bar)
        diff2_gamma_bar_mat_lo = self.two_spatial_deriv(gamma_bar_mat_lo)

        used_Gamma_bar = Gamma_bar
        used_Gamma_bar = -torch.einsum("xyzijj->xyzi", self.one_spatial_deriv(gamma_bar_mat_up))

        term1 = -1/2 * torch.einsum("xyzkl,xyzijkl->xyzij", gamma_bar_mat_up, diff2_gamma_bar_mat_lo)
        term2 = 1/2 * (torch.einsum("xyzki,xyzkj->xyzij", gamma_bar_mat_lo, diff_Gamma_bar)
                    + torch.einsum("xyzkj,xyzki->xyzij", gamma_bar_mat_lo, diff_Gamma_bar))
        term3 = 1/2 * (torch.einsum("xyzk,xyzijk->xyzij", used_Gamma_bar, Gamma_bar_mat_lo)
                    + torch.einsum("xyzk,xyzjik->xyzij", used_Gamma_bar, Gamma_bar_mat_lo))
        term4 = 1/2 * 2 * (torch.einsum("xyzlm,xyzkli,xyzjkm->xyzij", gamma_bar_mat_up, Gamma_bar_mat_up, Gamma_bar_mat_lo)
                        + torch.einsum("xyzlm,xyzklj,xyzikm->xyzij", gamma_bar_mat_up, Gamma_bar_mat_up, Gamma_bar_mat_lo))
        term5 = torch.einsum("xyzkl,xyzmil,xyzmkj->xyzij", gamma_bar_mat_up, Gamma_bar_mat_up, Gamma_bar_mat_lo)

        return term1 + term2 + term3 + term4 + term5

    def calc_R_phi_mat_lo(self, phi, Gamma_bar_mat_up, gamma_bar_mat_lo, gamma_bar_mat_up):
        diff_phi = self.one_spatial_deriv(phi)
        diff2_phi = self.two_spatial_deriv(phi)

        Diff2_bar_phi = diff2_phi - torch.einsum("xyzk,xyzkji->xyzij", diff_phi, Gamma_bar_mat_up)

        term1 = -2 * torch.einsum("xyzji->xyzij", Diff2_bar_phi)
        term2 = -2 * torch.einsum("xyzij,xyzlm,xyzml->xyzij", gamma_bar_mat_lo, gamma_bar_mat_up, Diff2_bar_phi)
        term3 = 4 * torch.einsum("xyzi,xyzj->xyzij", diff_phi, diff_phi)
        term4 = -4 * torch.einsum("xyzij,xyzlm,xyzl,xyzm->xyzij", gamma_bar_mat_lo, gamma_bar_mat_up, diff_phi, diff_phi)

        return term1 + term2 + term3 + term4

    def calc_R_mat_lo(self, R_bar_mat_lo, R_phi_mat_lo):
        return R_bar_mat_lo + R_phi_mat_lo


    # Evolution
    def calc_chi_dot(self, alpha, beta, chi, K):
        diff_beta = self.one_spatial_deriv(beta)

        term1 = 2/3 * chi * alpha * K
        term2 = - 2/3 * chi * alpha * torch.einsum("xyzaa->xyz", diff_beta)

        adv_diff_chi = self.advection_one_spatial_deriv(chi, beta)
        advection_term = torch.einsum("xyzi,xyzi->xyz", beta, adv_diff_chi)

        return term1 + term2 + advection_term
    
    def calc_gamma_bar_mat_lo_dot(self, alpha, beta, gamma_bar_mat_lo, A_bar_mat_lo):
        diff_gamma_bar_mat_lo = self.one_spatial_deriv(gamma_bar_mat_lo)
        diff_beta = self.one_spatial_deriv(beta)

        term1 = -2 * torch.einsum("xyz,xyzij->xyzij", alpha, A_bar_mat_lo)
        term2 = torch.einsum("xyzki,xyzkj->xyzij", gamma_bar_mat_lo, diff_beta)
        term3 = torch.einsum("xyzkj,xyzki->xyzij", gamma_bar_mat_lo, diff_beta)
        term4 = -2/3 * torch.einsum("xyzij,xyzkk->xyzij", gamma_bar_mat_lo, diff_beta)

        adv_diff_gamma_bar_mat_lo = self.advection_one_spatial_deriv(gamma_bar_mat_lo, beta)
        advection_term = torch.einsum("xyzk,xyzijk->xyzij", beta, adv_diff_gamma_bar_mat_lo)

        return term1 + term2 + term3 + term4 + advection_term
    
    def calc_K_dot(self, alpha, beta, gamma_mat_up, Gamma_mat_up, A_bar_mat_lo, A_bar_mat_up, K):
        diff_alpha = self.one_spatial_deriv(alpha)
        diff2_alpha = self.two_spatial_deriv(alpha)
        Diff2_alpha = diff2_alpha - torch.einsum("xyzk,xyzkji->xyzji", diff_alpha, Gamma_mat_up)

        term1 = - torch.einsum("xyzij,xyzji->xyz", gamma_mat_up, Diff2_alpha)
        term2 = torch.einsum("xyz,xyzij,xyzij->xyz", alpha, A_bar_mat_lo, A_bar_mat_up) + 1/3 * alpha * K**2

        adv_diff_K = self.advection_one_spatial_deriv(K, beta)
        advection_term = torch.einsum("xyzi,xyzi->xyz", beta, adv_diff_K)

        return term1 + term2 + advection_term
    
    def calc_A_bar_mat_lo_dot(self, alpha, beta, phi, gamma_mat_lo, gamma_mat_up, gamma_bar_mat_up, 
                              Gamma_mat_up, A_bar_mat_lo, K, R_mat_lo):
        diff_beta = self.one_spatial_deriv(beta)
        diff_alpha = self.one_spatial_deriv(alpha)
        diff2_alpha = self.two_spatial_deriv(alpha)
        Diff2_alpha = diff2_alpha - torch.einsum("xyzk,xyzkji->xyzij", diff_alpha, Gamma_mat_up)

        term1 = - torch.einsum("xyz,xyzji->xyzij", torch.exp(-4 * phi), Diff2_alpha)
        term2 = torch.einsum("xyz,xyz,xyzij->xyzij", torch.exp(-4 * phi), alpha, R_mat_lo)
        term123_TF = (term1 + term2) - 1/3 * torch.einsum('xyzij,xyz->xyzij', gamma_mat_lo, torch.einsum('xyzij,xyzij->xyz', gamma_mat_up, (term1 + term2)))

        term4 = torch.einsum('xyz,xyz,xyzij->xyzij', alpha, K, A_bar_mat_lo)

        A_bar_mat_uplo = torch.einsum("xyzil,xyzlj->xyzij", gamma_bar_mat_up, A_bar_mat_lo)
        term5 = - 2 * torch.einsum("xyz,xyzil,xyzlj->xyzij", alpha, A_bar_mat_lo, A_bar_mat_uplo)

        term5 = torch.einsum("xyzki,xyzkj->xyzij", A_bar_mat_lo, diff_beta)
        term6 = torch.einsum("xyzkj,xyzki->xyzij", A_bar_mat_lo, diff_beta)
        term7 = - 2/3 * torch.einsum("xyzij,xyzkk->xyzij", A_bar_mat_lo, diff_beta)

        adv_diff_A_bar_mat_lo = self.advection_one_spatial_deriv(A_bar_mat_lo, beta)
        advection_term = torch.einsum("xyzk,xyzijk->xyzij", beta, adv_diff_A_bar_mat_lo)

        return term123_TF + term4 + term5 + term6 + term7 + advection_term
    
    def calc_Gamma_bar_dot(self, alpha, beta, phi, gamma_bar_mat_up, Gamma_bar_mat_up, Gamma_bar, A_bar_mat_up, K):
        diff_alpha = self.one_spatial_deriv(alpha)
        diff_K = self.one_spatial_deriv(K)
        diff_phi = self.one_spatial_deriv(phi)
        diff_beta = self.one_spatial_deriv(beta)
        diff2_beta = self.two_spatial_deriv(beta)

        used_Gamma_bar = Gamma_bar
        used_Gamma_bar = -torch.einsum("xyzijj->xyzi",self.one_spatial_deriv(gamma_bar_mat_up))

        term1 = -2 * torch.einsum("xyzij,xyzj->xyzi", A_bar_mat_up, diff_alpha)
        term2 = 2 * torch.einsum("xyz,xyzijk,xyzkj->xyzi", alpha, Gamma_bar_mat_up, A_bar_mat_up)
        term3 = 2 * (-2/3) * torch.einsum("xyz,xyzij,xyzj->xyzi", alpha, gamma_bar_mat_up, diff_K)
        term5 = 2 * 6 * torch.einsum("xyz,xyzij,xyzj->xyzi", alpha, A_bar_mat_up, diff_phi)

        term6 = - torch.einsum("xyzj,xyzij->xyzi", used_Gamma_bar, diff_beta)
        term7 = 2/3 * torch.einsum("xyzi,xyzjj->xyzi", used_Gamma_bar, diff_beta)
        term8 = 1/3 * torch.einsum("xyzli,xyzjjl->xyzi", gamma_bar_mat_up, diff2_beta)
        term9 = torch.einsum("xyzlj,xyzilj->xyzi", gamma_bar_mat_up, diff2_beta)

        adv_diff_Gamma_bar = self.advection_one_spatial_deriv(Gamma_bar, beta)
        advection_term = torch.einsum("xyzj,xyzij->xyzi",beta,adv_diff_Gamma_bar)

        return term1 + term2 + term3 + term5 + term6 + term7 + term8 + term9 + advection_term
    
    def calc_alpha_dot(self, alpha, beta, K, lambdas):
        f_used = 2/alpha

        term1 = - alpha**2 * f_used * K

        adv_diff_alpha = self.advection_one_spatial_deriv(alpha, beta)
        advection_term = lambdas[0] * torch.einsum("xyzj,xyzj->xyz", beta, adv_diff_alpha)

        return term1 + advection_term
    
    def calc_beta_dot(self, alpha, beta, B, lambdas):
        f_used = torch.ones_like(alpha, device=self.device)

        term1 = 3/4 * torch.einsum("xyz,xyzi->xyzi", f_used, B)

        adv_diff_beta = self.advection_one_spatial_deriv(beta, beta)
        advection_term = lambdas[1] * torch.einsum("xyzj,xyzij->xyzi", beta, adv_diff_beta)

        return term1 + advection_term
    
    def calc_B_dot(self, alpha, beta, phi, gamma_bar_mat_up, Gamma_bar_mat_up, 
                   Gamma_bar, A_bar_mat_up, K, B, lambdas, damping):
        term1 = self.calc_Gamma_bar_dot(alpha, beta, phi, gamma_bar_mat_up, 
                                        Gamma_bar_mat_up, Gamma_bar, A_bar_mat_up, K)
        term2 = -damping * B

        adv_diff_B = self.advection_one_spatial_deriv(B, beta)
        advection_term1 = lambdas[2] * torch.einsum("xyzj,xyzij->xyzi", beta, adv_diff_B)
        adv_diff_Gamma = self.advection_one_spatial_deriv(Gamma_bar, beta)
        advection_term2 = -lambdas[3] * torch.einsum("xyzj,xyzij->xyzi", beta, adv_diff_Gamma)

        return term1 + term2 + advection_term1 + advection_term2


    # Constraints
    def calc_hamiltonian_constraint(self, phi, gamma_bar_mat_up, Gamma_bar_mat_up, 
                                    A_bar_mat_lo, A_bar_mat_up, K, R_bar_mat_lo):
        diff_exp_phi = self.one_spatial_deriv(torch.exp(phi))
        diff2_exp_phi = self.two_spatial_deriv(torch.exp(phi))

        Diff2_bar_exp_phi = diff2_exp_phi - torch.einsum("xyzk,xyzkji->xyzij",diff_exp_phi, Gamma_bar_mat_up)

        term1 = torch.einsum("xyzij,xyzji->xyz",gamma_bar_mat_up,Diff2_bar_exp_phi)
        term2 = - torch.exp(phi)/8 * torch.einsum("xyzij,xyzij->xyz",gamma_bar_mat_up, R_bar_mat_lo)
        term3 = torch.exp(5*phi)/8 * torch.einsum("xyzij,xyzij->xyz",A_bar_mat_lo, A_bar_mat_up)
        term4 = - torch.exp(5*phi)/12*K**2

        return term1 + term2 + term3 + term4

    def calc_momentum_constraint(self, phi, Gamma_bar_mat_up, A_bar_mat_up, K):
        phi_A = torch.einsum("xyz,xyzji->xyzji",torch.exp(6*phi),A_bar_mat_up)
        diff_phi_A = self.one_spatial_deriv(phi_A)
        Diff_phi_A = torch.einsum("xyzjij->xyzi",diff_phi_A) + torch.einsum("xyzli,xyzjlj->xyzi",phi_A,Gamma_bar_mat_up) + torch.einsum("xyzjl,xyzilj->xyzi",phi_A,Gamma_bar_mat_up)

        diff_K = self.one_spatial_deriv(K)

        term1 = Diff_phi_A
        term2 = -2/3 * torch.einsum("xyz,xyzi->xyzi",torch.exp(6*phi),diff_K)

        return term1 + term2
    

    def update_funcs(self, chi, gamma_bar_mat_lo_raw, A_bar_mat_lo_raw, Gamma_bar, K):
        """
        This function updates all the non-evolution variables using the newly
        evolved variables.
        """

        print("Updating functions")

        # Replace the values of "chi" or "W" to avoid ln(0)
        if torch.min(chi) < self.epsilon:
            chi = chi + self.epsilon
        phi = -torch.log(chi)/4

        # Calculate the variables that are not part of the evolution.
        gamma_bar_mat_lo = torch.einsum('xyz,xyzij->xyzij',torch.linalg.det(gamma_bar_mat_lo_raw)**(-1/3),gamma_bar_mat_lo_raw)     # Actuall an evolving varible; enforce constraints

        # calc_gamma_bar_mat_lo()
        gamma_bar_mat_up = self.calc_gamma_bar_mat_up(gamma_bar_mat_lo)
        gamma_mat_lo = self.calc_gamma_mat_lo(phi, gamma_bar_mat_lo)
        gamma_mat_up = self.calc_gamma_mat_up(phi, gamma_bar_mat_up)

        A_bar_mat_lo = A_bar_mat_lo_raw - 1/3 * torch.einsum('xyzij,xyz->xyzij',gamma_bar_mat_lo,torch.einsum('xyzij,xyzij->xyz',gamma_bar_mat_up,A_bar_mat_lo_raw)) 
        # ↑ Actuall an evolving varible; enforce constraints

        Gamma_mat_up = self.calc_Gamma_mat_up(gamma_mat_lo, gamma_mat_up)
        Gamma_mat_lo = self.calc_Gamma_mat_lo(gamma_mat_lo, Gamma_mat_up)
        Gamma_bar_mat_up = self.calc_Gamma_bar_mat_up(gamma_bar_mat_lo, gamma_bar_mat_up)
        Gamma_bar_mat_lo = self.calc_Gamma_bar_mat_lo(gamma_bar_mat_lo, Gamma_bar_mat_up)
        # calc_Gamma_bar()

        # calc_A_bar_mat_lo()
        A_bar_mat_up = self.calc_A_bar_mat_up(gamma_bar_mat_up, A_bar_mat_lo)
        K_mat_lo = self.calc_K_mat_lo(phi, A_bar_mat_lo, gamma_mat_lo, K)

        R_bar_mat_lo = self.calc_R_bar_mat_lo(Gamma_bar, Gamma_bar_mat_lo, Gamma_bar_mat_up,
                                                   gamma_bar_mat_lo, gamma_bar_mat_up)
        R_phi_mat_lo = self.calc_R_phi_mat_lo(phi, Gamma_bar_mat_up, gamma_bar_mat_lo, 
                                                   gamma_bar_mat_up)
        R_mat_lo = self.calc_R_mat_lo(R_bar_mat_lo, R_phi_mat_lo)

        return chi, phi, gamma_bar_mat_lo, gamma_bar_mat_up, gamma_mat_lo, gamma_mat_up, A_bar_mat_lo, Gamma_mat_up, \
            Gamma_mat_lo, Gamma_bar_mat_up, Gamma_bar_mat_lo, A_bar_mat_up, K_mat_lo, R_bar_mat_lo, R_phi_mat_lo, R_mat_lo
    
    def interpolate_gridfield_at_positions(self, grid_field, positions, max_x):
        n = grid_field.shape[0]
        B = positions.shape[0]

        pos_grid = (positions + max_x) * ((n - 1) / (2 * max_x))

        trailing_dims = grid_field.shape[3:]
        C = 1
        for d in trailing_dims:
            C *= d

        grid_field_flat = grid_field.reshape(n, n, n, C)  # (n, n, n, C)
        grid_field_reshaped = grid_field_flat.permute(3, 0, 1, 2)  # (C, n, n, n)
        grid_field_reshaped = grid_field_reshaped.unsqueeze(0)  # (1, C, n, n, n)

        pos_norm = 2 * (pos_grid / (n - 1)) - 1  # (B, 3)

        grid = pos_norm.view(B, 1, 1, 1, 3)

        grid_field_batch = grid_field_reshaped.expand(B, -1, -1, -1, -1)  # (B, C, n, n, n)

        sampled = F.grid_sample(grid_field_batch, grid, align_corners=True, mode='bilinear', padding_mode='border')

        sampled = sampled.view(B, C)  # (B, C)
        sampled = sampled.view(B, *trailing_dims)  # (B, dim1, dim2, ...)

        return sampled

    def forward(self, t, x):
        print("time: ", t.item())
        alpha, beta, chi_raw, gamma_bar_mat_lo_raw, Gamma_bar, A_bar_mat_lo_raw, K, B, hc, mc, X, V = x

        chi, phi, gamma_bar_mat_lo, gamma_bar_mat_up, gamma_mat_lo, gamma_mat_up, A_bar_mat_lo, \
        Gamma_mat_up, Gamma_mat_lo, Gamma_bar_mat_up, Gamma_bar_mat_lo, A_bar_mat_up, K_mat_lo, \
        R_bar_mat_lo, R_phi_mat_lo, R_mat_lo = self.update_funcs(chi=chi_raw, gamma_bar_mat_lo_raw=gamma_bar_mat_lo_raw,
                                                                A_bar_mat_lo_raw=A_bar_mat_lo_raw, Gamma_bar=Gamma_bar, K=K)

        hamiltonian_constraint = torch.mean(torch.abs(self.calc_hamiltonian_constraint(phi, gamma_bar_mat_up, Gamma_bar_mat_up, A_bar_mat_lo,
                                                                                    A_bar_mat_up, K, R_bar_mat_lo)))
        momentum_constraint = torch.mean(torch.abs(self.calc_momentum_constraint(phi, Gamma_bar_mat_up, A_bar_mat_up, K)))

        gamma_bar_mat_lo_dot = self.calc_gamma_bar_mat_lo_dot(alpha, beta, gamma_bar_mat_lo, A_bar_mat_lo)
        K_dot = self.calc_K_dot(alpha, beta, gamma_mat_up, Gamma_mat_up, A_bar_mat_lo, A_bar_mat_up, K)
        A_bar_mat_lo_dot = self.calc_A_bar_mat_lo_dot(alpha, beta, phi, gamma_mat_lo, gamma_mat_up, gamma_bar_mat_up,
                                                    Gamma_mat_up, A_bar_mat_lo, K, R_mat_lo)
        Gamma_bar_dot = self.calc_Gamma_bar_dot(alpha, beta, phi, gamma_bar_mat_up, Gamma_bar_mat_up, Gamma_bar,
                                                A_bar_mat_up, K)
        alpha_dot = self.calc_alpha_dot(alpha, beta, K, self.lambdas)
        beta_dot = self.calc_beta_dot(alpha, beta, B, self.lambdas)
        B_dot = self.calc_B_dot(alpha, beta, phi, gamma_bar_mat_up, Gamma_bar_mat_up,
                                Gamma_bar, A_bar_mat_up, K, B, self.lambdas, self.damping)
        chi_dot = self.calc_chi_dot(alpha, beta, chi, K)

        print("Hamiltonian Constraint: ", hamiltonian_constraint.item())
        print("Momentum Constraint: ", momentum_constraint.item())

        device = X.device
        dtype = X.dtype
        Bbatch = X.shape[0]

        grid_spacing = (2.0 * self.max_x) / (float(self.res) - 1.0)
        eps = grid_spacing * 1e-2  

        out_of_bounds = (X.abs() > self.max_x).any(dim=1)  # (B,)
        self.last_X_out_of_bounds = out_of_bounds

        X_clamped = X       # For gradient, perform no clamp


        def sample_field(grid_field):
            """
            sample grid_field at X_clamped, returning a tensor with trailing dims preserved.
            """
            return self.interpolate_gridfield_at_positions(grid_field, X_clamped, max_x=self.max_x)


        def grad_scalar_field(grid_field):
            grads = []
            for axis in range(3):
                shift = torch.zeros_like(X_clamped)
                shift[:, axis] = eps
                plus = self.interpolate_gridfield_at_positions(grid_field, (X_clamped + shift).clamp(-self.max_x, self.max_x), max_x=self.max_x)
                minus = self.interpolate_gridfield_at_positions(grid_field, (X_clamped - shift).clamp(-self.max_x, self.max_x), max_x=self.max_x)
                plus = plus.view(Bbatch, -1).squeeze(-1) if plus.dim() > 1 else plus
                minus = minus.view(Bbatch, -1).squeeze(-1) if minus.dim() > 1 else minus
                grads.append(((plus - minus) / (2.0 * eps)).unsqueeze(-1))  # (B,1)
            grad = torch.cat(grads, dim=1)  # (B,3)
            return grad

        def grad_vector_field(grid_field):  
            grads_comp = []
            for i_comp in range(3):
                component_grid = lambda: None
                grads = []
                for axis in range(3):
                    shift = torch.zeros_like(X_clamped)
                    shift[:, axis] = eps
                    plus = self.interpolate_gridfield_at_positions(grid_field, (X_clamped + shift).clamp(-self.max_x, self.max_x), max_x=self.max_x)
                    minus = self.interpolate_gridfield_at_positions(grid_field, (X_clamped - shift).clamp(-self.max_x, self.max_x), max_x=self.max_x)
                    plus_c = plus[:, i_comp]
                    minus_c = minus[:, i_comp]
                    grads.append(((plus_c - minus_c) / (2.0 * eps)).unsqueeze(-1))  # (B,1)
                grads_comp.append(torch.cat(grads, dim=1).unsqueeze(1))  # (B,1,3)
            grad_vec = torch.cat(grads_comp, dim=1)
            return grad_vec

    
        alpha_b = sample_field(alpha)  # shape (B,) or (B,1)
        if alpha_b.dim() > 1:
            alpha_b = alpha_b.view(Bbatch, -1).squeeze(-1)

        if torch.min(alpha_b) < self.epsilon:
            alpha_b = alpha_b + self.epsilon

        beta_b = sample_field(beta)
        if beta_b.dim() == 1:
            beta_b = beta_b.unsqueeze(-1)  

        Gamma_b = sample_field(Gamma_mat_up) 
        K_lo_b = sample_field(K_mat_lo)
        gamma_up_b = sample_field(gamma_mat_up)
        gamma_lo_b = sample_field(gamma_mat_lo)
        K_mixed = torch.einsum('bik,bkj->bij', gamma_up_b, K_lo_b)
        grad_alpha = grad_scalar_field(alpha)          
        grad_beta = grad_vector_field(beta)            
        alpha_safe = alpha_b.clamp(min=self.epsilon)
        grad_ln_alpha = grad_alpha / alpha_safe.unsqueeze(1)  # (B,3)
        KdotV = torch.einsum('bjk,bk->bj', K_lo_b, V)
        GammaV = torch.einsum('bijk,bk->bij', Gamma_b, V)
        term1 = grad_ln_alpha - KdotV  # (B,j)
        Vi_term = V.unsqueeze(2) * term1.unsqueeze(1)  
        bracket = Vi_term + 2.0 * K_mixed - GammaV  # (B,i,j)
        bracket_contract = torch.einsum('bij,bj->bi', bracket, V)
        alpha_factor = alpha_b.unsqueeze(1)  # (B,1)
        first_big_term = alpha_factor * bracket_contract  # (B,3)
        gamma_grad_alpha = torch.einsum('bij,bj->bi', gamma_up_b, grad_alpha)
        V_grad_beta = torch.einsum('bj,bij->bi', V, grad_beta)


        dVdt = first_big_term - gamma_grad_alpha - V_grad_beta  # (B,3)
        dXdt = alpha_b.unsqueeze(1) * V - beta_b  # (B,3)


        return alpha_dot, beta_dot, chi_dot, gamma_bar_mat_lo_dot, Gamma_bar_dot, A_bar_mat_lo_dot, K_dot, B_dot, \
            hamiltonian_constraint, momentum_constraint, dXdt, dVdt




if __name__ == '__main__':
    # Tests here...
    BH_locs = [[0.0, 0.0, 0.25], [0.0, 0.0, -0.25], [0.0, 0.0, 0.125], [0.0, 0.0, -0.125]]
    BH_masses = np.array([1, 1, 1., 1]) * -4.6
    ODEfunc = NeuralODEfunc(BH_masses_presoftplus=BH_masses, BH_positions=BH_locs)
    ODEfunc.forward(10, dt=0.01, folder="./out")

