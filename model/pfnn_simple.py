import torch
import torch.nn as nn
import torch.nn.functional as F

class PFNN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Operators
        self.Gc = nn.Linear(latent_dim, latent_dim, bias=False)
        self.Gm = nn.Linear(latent_dim, latent_dim, bias=False)
        self.Gm_star = nn.Linear(latent_dim, latent_dim, bias=False)

        # Encoder-Decoder: tanh activation
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z, mode='contract'):
        z_encoded = self.encoder(z)
        if mode == 'contract':
            z_next = self.Gc(z_encoded)
        elif mode == 'invariant':
            z_next = self.Gm(z_encoded)
        elif mode == 'inverse':
            z_next = self.Gm_star(z_encoded)
        else:
            raise ValueError("Invalid mode")
        return self.decoder(z_next)

    def get_unitary_regularization(self):
        I = torch.eye(self.latent_dim).to(self.Gm.weight.device)
        Gm = self.Gm.weight
        Gm_star = self.Gm_star.weight
        return (
            torch.norm(Gm @ Gm_star - I) +
            torch.norm(Gm_star @ Gm - I)
        )

    def get_contraction_regularization(self):
        I = torch.eye(self.latent_dim).to(self.Gc.weight.device)
        eigvals = self.Gc.weight.T @ self.Gc.weight - I
        return F.relu(eigvals).norm()
