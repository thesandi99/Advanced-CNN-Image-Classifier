%%writefile dict.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import deque, defaultdict
import time
import datetime


def dct_matrix(size):
    """Generates the DCT transformation matrix."""
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == 0:
                m[i, j] = np.sqrt(1. / size)
            else:
                m[i, j] = np.sqrt(2. / size) * np.cos((2 * j + 1) * i * np.pi / (2 * size))
    return torch.tensor(m, dtype=torch.float32)


class DCTFeatureExtractor(nn.Module):
    def __init__(self, window_size=32, stride=16, num_patches=16, levels=1):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.num_patches = num_patches  # Number of top/bottom patches to consider
        self.levels = levels

        self.dct_basis = nn.Parameter(dct_matrix(window_size), requires_grad=False)
        self.dct_basis_t = nn.Parameter(self.dct_basis.T, requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)

        # No fold operation needed; we'll keep the patches separate.
        self.grade_filters = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False) for _ in range(6)  # Example: 6 learnable grade filters
        ])
        for filter_layer in self.grade_filters:
          nn.init.kaiming_normal_(filter_layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        N, C, H, W = x.shape

        # --- Padding ---
        pad_h = (self.stride - (H - self.window_size) % self.stride) % self.stride
        pad_w = (self.stride - (W - self.window_size) % self.stride) % self.stride
        # Pad symmetrically (reflection padding)
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
        # Now unfold
        patches = self.unfold(x)  # (N, C*window_size*window_size, L)

        L = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(N, L, C, self.window_size, self.window_size)

        # Apply DCT:  (N, L, C, w, w) @ (w, w) -> (N, L, C, w, w)
        dct_coeffs = torch.einsum('nlcwh,xy->nlcwy', patches, self.dct_basis)
        dct_coeffs = torch.einsum('nlcwh,yw->nlcyh', dct_coeffs, self.dct_basis_t)

        # Calculate energy for each patch (sum of squared DCT coefficients).
        energy = torch.sum(dct_coeffs ** 2, dim=(3, 4))  # (N, L, C)

        # --- Handle topk edge cases ---
        num_patches = min(self.num_patches, L)  # Ensure we don't select more patches than exist

        # Get top-k and bottom-k indices *before* combining channels.
        _, topk_indices = torch.topk(energy, num_patches, dim=1)  # (N, num_patches, C)
        _, bottomk_indices = torch.topk(energy, num_patches, dim=1, largest=False)  # (N, num_patches, C)

        # Combine energies across color channels (NOW we sum across channels)
        # energy = energy.sum(dim=2)  # MOVED this line down

        # Gather the top-k and bottom-k patches.
        # Expand the indices for gathering.
        dct_top_patches = torch.gather(patches, 1, topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.window_size, self.window_size))
        dct_bottom_patches = torch.gather(patches, 1, bottomk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.window_size, self.window_size))

        return dct_top_patches, dct_bottom_patches