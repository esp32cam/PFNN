import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator
from koopman_base import *
from restormer_arch import *


torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)

######### Transformer based Koopman/PFNN(meansure-invariant) model #########
class KoopmanAE_2d_trans(nn.Module):
    def __init__(self, in_channel, out_channel, dim = 4, num_blocks = [2, 2, 2, 2], heads = [1, 2, 4, 4], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', steps = 1, steps_back = 1, init_scale=1, grid_info = True):
        super(KoopmanAE_2d_trans, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.latent_dim = dim * 2 * 2 * 2
        
        if grid_info:
            self.grid_dim = 2
        else:
            self.grid_dim = 0

        self.patch_embed = OverlapPatchEmbed(self.grid_dim + in_channel, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample_4x(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample_4x(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])



        self.up4_3 = Upsample_4x(int(dim*2**3)) ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample_4x(int(dim*2**2)) ## From Level 3 to Level 2
        # self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        # self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_channel, kernel_size=3, stride=1, padding=1, bias=bias)

        self.encoder = nn.Sequential(
            self.patch_embed,  # [batch, dim, 64, 64]
            self.encoder_level1, # [batch, dim, 64, 64]
            self.down1_2, # [batch, dim*2, 32, 32]
            self.encoder_level2, # [batch, dim*2, 32, 32]
            self.down2_3, # [batch, dim*2*2, 8, 8]
            self.encoder_level3, # [batch, dim*2*2, 8, 8]
            self.down3_4, # [batch, dim*2*2*2, 2, 2]
            self.latent) # [batch, dim*2*2*2, 2, 2]

        self.decoder = nn.Sequential(
            self.up4_3, # [batch, dim*2*2, 8, 8]
            self.decoder_level3, # [batch, dim*2*2, 8, 8]
            self.up3_2, # [batch, dim*2, 32, 32]
            self.decoder_level2, # [batch, dim*2, 32, 32]
            self.up2_1, # [batch, dim, 64, 64]
            self.decoder_level1, # [batch, dim, 64, 64]
            self.output) # [batch, out_channel, 64, 64]

        self.dynamics = dynamics(self.latent_dim * 2 * 2, init_scale)
        self.backdynamics = dynamics_back(self.latent_dim  * 2 * 2, self.dynamics)

    def forward(self, x, mode='forward'):

        out = []
        out_id = []
        out_back = []
        out_back_id = []
        if self.grid_dim > 0:
            grid = self.get_grid(x.shape[2], x.shape[0], x.device)
            x_grid = torch.cat((x, grid), dim=1)
        else:
            x_grid = x
        z = self.encoder(x_grid.contiguous())
        qt = z.contiguous().reshape(z.size(0), -1) # flatten to [batch, self.latent_dim * 2 * 2] 

        if mode == 'forward':
            for _ in range(self.steps):
                q_t1 = self.dynamics(qt)
                # q_t = q_t1 # if steps > 1, latent update should be autoregressive 
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