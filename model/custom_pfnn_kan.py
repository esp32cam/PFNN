import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator
from .koopman_base import *

# Minimal KAN block (replace with your advanced KAN if available)
class KANBlock(nn.Module):
    def __init__(self, dim):
        super(KANBlock, self).__init__()
        self.fc1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=1)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class KoopmanAE_2d_kan(nn.Module):
    def __init__(self, in_channel, out_channel, dim = 4, num_blocks = [2, 2, 2, 2], steps = 1, steps_back = 1, init_scale=1, grid_info = True):
        super(KoopmanAE_2d_kan, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.latent_dim = dim * 2 * 2 * 2
        if grid_info:
            self.grid_dim = 2
        else:
            self.grid_dim = 0
        self.patch_embed = nn.Conv2d(self.grid_dim + in_channel, dim, kernel_size=3, stride=1, padding=1)
        self.encoder_level1 = nn.Sequential(*[KANBlock(dim) for _ in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2)
        self.encoder_level2 = nn.Sequential(*[KANBlock(dim*2) for _ in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(dim*2, dim*4, kernel_size=4, stride=4)
        self.encoder_level3 = nn.Sequential(*[KANBlock(dim*4) for _ in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(dim*4, dim*8, kernel_size=4, stride=4)
        self.latent = nn.Sequential(*[KANBlock(dim*8) for _ in range(num_blocks[3])])
        self.up4_3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=4, stride=4)
        self.decoder_level3 = nn.Sequential(*[KANBlock(dim*4) for _ in range(num_blocks[2])])
        self.up3_2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=4, stride=4)
        self.decoder_level2 = nn.Sequential(*[KANBlock(dim*2) for _ in range(num_blocks[1])])
        self.up2_1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.decoder_level1 = nn.Sequential(*[KANBlock(dim) for _ in range(num_blocks[0])])
        self.output = nn.Conv2d(dim, out_channel, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            self.patch_embed,
            self.encoder_level1,
            self.down1_2,
            self.encoder_level2,
            self.down2_3,
            self.encoder_level3,
            self.down3_4,
            self.latent)
        self.decoder = nn.Sequential(
            self.up4_3,
            self.decoder_level3,
            self.up3_2,
            self.decoder_level2,
            self.up2_1,
            self.decoder_level1,
            self.output)
        self.dynamics = dynamics(self.latent_dim * 2 * 2, init_scale)
        self.backdynamics = dynamics_back(self.latent_dim * 2 * 2, self.dynamics)
    def forward(self, x, mode='forward'):
        out = []
        out_id = []
        out_back = []
        out_back_id = []
        if self.grid_dim > 0:
            grid = self.get_grid(x.shape[1], x.shape[0], x.device)
            x_grid = torch.cat((x, grid.permute(0, 2, 3, 1)), dim=-1)
            x_grid = x_grid.permute(0, 3, 1, 2)
        else:
            x_grid = x
        z = self.encoder(x_grid.contiguous())
        qt = z.contiguous().reshape(z.size(0), -1)
        if mode == 'forward':
            for _ in range(self.steps):
                q_t1 = self.dynamics(qt)
                q_t1 = q_t1.view(q_t1.size(0), self.latent_dim, 2, 2)
                out.append(self.decoder(q_t1))
            out_id.append(self.decoder(z.contiguous()))
            return out, out_id
        if mode == 'backward':
            for _ in range(self.steps_back):
                q_1t = self.backdynamics(qt)
                q_1t = q_1t.view(q_1t.size(0), self.latent_dim, 2, 2)
                out_back.append(self.decoder(q_1t))
            out_back_id.append(self.decoder(z.contiguous()))
            return out_back, out_back_id
    def get_grid(self, S, batchsize, device):
        gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridx = gridx.reshape(1, 1, S, 1).repeat([batchsize, 1, 1, S])
        gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, S).repeat([batchsize, 1, S, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
