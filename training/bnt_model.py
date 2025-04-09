import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sys
from omegaconf import OmegaConf
import yaml

# Get directories from config.yaml
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

bnt_dir = config["bnt_dir"]

sys.path.append(bnt_dir)
from models.BNT.bnt import BrainNetworkTransformer

def get_bnt_config():
    return OmegaConf.create({
        "dataset": {
            "node_sz": 200  # Number of ROIs
        },
        "model": {
            "sizes": [200, 100],  # You can adjust this
            "pos_encoding": "none",  # or "identity" if you want PE
            "pos_embed_dim": 16,
            "pooling": [True, True],
            "orthogonal": True,
            "freeze_center": False,
            "project_assignment": True
        }
    })

class BNTContrastiveEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = get_bnt_config()
        self.bnt = BrainNetworkTransformer(config)

        # Output from BNT = (B, 2), from `self.fc`
        # Instead, we’ll capture intermediate embeddings before `fc`
        self.projector = nn.Sequential(
            nn.Linear(8 * config.model.sizes[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, corr_matrix):
        """
        corr_matrix: Tensor of shape (B, 200, 200)
        """
        x = corr_matrix  # fake as node feature input
        # Forward through BNT (we only need embeddings, not predictions)
        bz = x.shape[0]
        node_feature = x  # shape: (B, 200, 200)
        time_series = None  # not used here

        if self.bnt.pos_encoding == 'identity':
            pos_emb = self.bnt.node_identity.expand(bz, *self.bnt.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []
        for atten in self.bnt.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.bnt.dim_reduction(node_feature)
        flat = node_feature.reshape(bz, -1)
        z = self.projector(flat)
        return z
    