import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
from omegaconf import OmegaConf
from bnt_model import BNTContrastiveEncoder
import yaml

# Get directories from config.yaml
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

corr_matrices_file = config["corr_matrices_file"]


# Step 1: Load Dataset with On-the-Fly Augmentation
# Applies two different augmentations to the same sample (dilate/shrink + noise)
# Returns the pair (for contrastive loss)
class ContrastiveGraphDataset(Dataset):
    def __init__(self, npy_path, num_aug_nodes=(5, 20), epsilon=0.05, noise_std=0.01):
        self.data = np.load(npy_path)  # shape: (N, 200, 200)
        self.num_aug_nodes = num_aug_nodes
        self.epsilon = epsilon
        self.noise_std = noise_std

    def __len__(self):
        return len(self.data)

    def dilate_shrink(self, matrix, mode='dilate'):
        mat = matrix.copy()
        num_nodes = random.randint(*self.num_aug_nodes)
        nodes = random.sample(range(mat.shape[0]), num_nodes)

        for i in nodes:
            if mode == 'dilate':
                mat[i, :] += self.epsilon * np.abs(mat[i, :])
                mat[:, i] += self.epsilon * np.abs(mat[:, i])
            elif mode == 'shrink':
                mat[i, :] -= self.epsilon * np.abs(mat[i, :])
                mat[:, i] -= self.epsilon * np.abs(mat[:, i])
        
        np.fill_diagonal(mat, 1.0)
        mat += np.random.normal(0, self.noise_std, mat.shape)  # small Gaussian noise

        # Check for NaNs BEFORE conversion to torch
        if np.isnan(mat).any():
            print("⚠️ NaNs detected before returning augmented matrix")
            print("Matrix stats:", np.min(mat), np.max(mat), np.mean(mat))

        return np.clip(mat, -1.0, 1.0)  # keep correlations in range

    def __getitem__(self, idx):
        matrix = self.data[idx].astype(np.float32)
        aug1 = self.dilate_shrink(matrix, mode='dilate')
        aug2 = self.dilate_shrink(matrix, mode='shrink')
        return torch.tensor(aug1), torch.tensor(aug2)

# Step 2: Contrastive Loss (InfoNCE / MoCo-style)

import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.07):
    """
    Compute NT-Xent loss between two batches of embeddings.
    """
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # (2B, D)
    similarity_matrix = torch.matmul(representations, representations.T)

    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.sum(z1 * z2, dim=-1)
    positives = torch.cat([positives, positives], dim=0)

    numerator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)

    loss = -torch.log(numerator / (denominator + 1e-8))
    return loss.mean()


# Step 4: Training Loop
from torch.utils.data import DataLoader
import torch.optim as optim

# Initialize
raw = np.load(corr_matrices_file)
print("NaNs in dataset:", np.isnan(raw).any())

dataset = ContrastiveGraphDataset(corr_matrices_file)
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
device = torch.device("cpu")  # since CUDA is not available

model = BNTContrastiveEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for aug1, aug2 in loader:
        aug1, aug2 = aug1.to(device), aug2.to(device)
        z1 = model(aug1)
        z2 = model(aug2)

        # If norm collapses to zero or very small values → embedding collapse.
        print("z1 mean norm:", z1.norm(dim=1).mean().item())

        loss = contrastive_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(loader):.4f}")

# Step 5: Save Pretrained Weights
torch.save(model.state_dict(), "bnt_contrastive_pretrained.pth")