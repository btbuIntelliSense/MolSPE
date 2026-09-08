# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from torch_geometric.data import Data
import numpy as np
import torch
from scipy.spatial import distance_matrix

from MaskModel.coord2model import coords2unimol
# from Motif.MaskModel.coord2modelalllll import coords2unimol

# from torchmdnet.datasets.coord2modelall import coords2unimol


def Distance(pos):
    dist = distance_matrix(pos, pos).astype(np.float32)
    return torch.from_numpy(dist)


def mask_points(
        data,
        vocab,
        pad_idx,
        mask_idx,
        noise_type,
        noise,
        mask_prob,
        max_atoms,
):
    assert 0.0 < mask_prob < 1.0


    if noise_type == "trunc_normal":
        noise_f = lambda num_mask: np.clip(
            np.random.randn(num_mask, 3) * noise,
            a_min=-noise * 2.0,
            a_max=noise * 2.0,
        )
    elif noise_type == "normal":
        noise_f = lambda num_mask: np.random.randn(num_mask, 3) * noise
    elif noise_type == "uniform":
        noise_f = lambda num_mask: np.random.uniform(
            low=-noise, high=noise, size=(num_mask, 3)
        )
    else:
        noise_f = lambda num_mask: 0.0

    z = np.array(data["z"])
    pos = np.array(data["pos"])  # pos
    pos_denoise = np.array(data["pos_denoise"]) #pos+noise
    pos_noise = np.array(data["noise"])
        # pos_target = pos_target - pos_target.mean(axis=0)
    sz = len(z)
        # z1_list.append(z)
    assert sz > 0

    # num_mask = int(mask_prob * sz + np.random.rand())
    num_mask = 1
    mask_idc = np.random.choice(sz, num_mask, replace=False)
    mask = np.full(sz, False)
    mask[mask_idc] = True  # 生成掩码

    targets = np.full(len(mask), pad_idx)
    targets[mask] = z[mask]

    masked_z = np.copy(z)
    masked_z[mask] = mask_idx

    num_mask = mask.astype(np.int32).sum()
    noisy_pos = np.copy(pos)
    noisy_pos[mask, :] += noise_f(num_mask)

    z, mask_z, coord, mask_coord, dist, mask_dist, targets, src_edge_type,bond_adj = coords2unimol(
            torch.from_numpy(z).long(), torch.from_numpy(masked_z).long(),
            torch.from_numpy(pos).float(), torch.from_numpy(noisy_pos).float(),
            torch.from_numpy(targets).long(),data["bond_adj"],vocab, max_atoms)
        # return result


    return  z, mask_z, coord, mask_coord, dist, mask_dist, targets, src_edge_type, bond_adj,torch.from_numpy(masked_z).long(),torch.from_numpy(mask_idc).long()




