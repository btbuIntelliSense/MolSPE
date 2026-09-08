import numpy as np
from scipy.spatial import distance_matrix
import torch


def coords2unimol(
    atoms,mask_atom,coordinates,mask_coord,targets,bond_adj,dictionary, max_atoms=256, **params
):

    z = atoms.numpy()#z
    coord = coordinates.numpy()#pos_target
    mask_z=mask_atom.numpy()#masked_z
    mask_coord=mask_coord.numpy()#mask_pos
    targets=targets.numpy()#targets

    bond_adj=bond_adj.numpy()
    # atoms = [str(num) for num in atoms]
    # mask_atom = [str(num) for num in mask_atom]
    # targets = [str(num) for num in targets]

    # cropping atoms and coordinates 剪裁原子和坐标
    if len(z) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        z = z[idx]
        coord = coord[idx]
        mask_z = mask_z[idx]
        mask_coord = mask_coord[idx]
        targets = targets[idx]
        # pos = pos[idx]
        # pos_noise = pos_noise[idx]
        bond_adj = bond_adj[idx]
    # tokens padding
    # z = np.array(
    #     [dictionary.bos()]
    #     + [dictionary.atomic_numbers_to_indices(atom) for atom in atoms]
    #     + [dictionary.eos()]
    # )
    # mask_z = np.array(
    #     [dictionary.bos()]
    #     + [dictionary.index(atom) for atom in mask_atom]
    #     + [dictionary.eos()]
    # )
    #
    # targets = np.array([dictionary.index(atom) for atom in targets] )
    # z = np.array([dictionary.atomic_numbers_to_indices(atom) for atom in atoms])
    # mask_z = np.array([dictionary.atomic_numbers_to_indices(atom) for atom in mask_atom])
    # targets = np.array([dictionary.atomic_numbers_to_indices(atom) for atom in targets])




    # coord = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)
    # mask_coord = np.concatenate([np.zeros((1, 3)), mask_coord, np.zeros((1, 3))], axis=0)
    # Calculate distance matrix BEFORE padding
    dist = distance_matrix(coord, coord).astype(np.float32)
    mask_dist = distance_matrix(mask_coord, mask_coord).astype(np.float32)
    # padding to max_atoms
    padding_length = max_atoms - len(atoms)
    if padding_length > 0:
        z = np.pad(z, (0, padding_length), mode='constant', constant_values=dictionary.pad())
        mask_z = np.pad(mask_z, (0, padding_length), mode='constant', constant_values=dictionary.pad())
        targets = np.pad(targets, (0, padding_length), mode='constant', constant_values=dictionary.pad())
        coord = np.pad(coord, ((0, padding_length), (0, 0)), mode='constant', constant_values=0.0)
        mask_coord = np.pad(mask_coord, ((0, padding_length), (0, 0)), mode='constant', constant_values=0.0)
        # pos = np.pad(pos, ((0, padding_length), (0, 0)), mode='constant', constant_values=0.0)
        # pos_noise = np.pad(pos_noise, ((0, padding_length), (0, 0)), mode='constant', constant_values=0.0)
        dist = np.pad(dist, ((0, padding_length), (0,padding_length)), constant_values=0.0)
        mask_dist = np.pad(mask_dist, ((0, padding_length), (0, padding_length)), constant_values=0.0)
        bond_adj=np.pad(bond_adj, ((0, padding_length), (0, padding_length)), constant_values=0.0)
    # distance matrix
    # dist = distance_matrix(coord, coord).astype(np.float32)
    # mask_dist = distance_matrix(mask_coord, mask_coord).astype(np.float32)
    # --- Step 5: Edge type with padding mask ---
    # valid_mask = (z != dictionary.pad())
    # src_edge_type = np.where(valid_mask[:, None] & valid_mask[None, :],
    #                          z.reshape(-1, 1) * len(dictionary) + z.reshape(1, -1),
    #                          dictionary.pad())
    # edge type
    src_edge_type = z.reshape(-1, 1) * len(dictionary) + z.reshape( 1, -1)

    z = torch.tensor(z, dtype=torch.long)
    mask_z = torch.tensor(mask_z, dtype=torch.long)
    coord = torch.tensor(coord, dtype=torch.float32)
    mask_coord = torch.tensor(mask_coord, dtype=torch.float32)
    dist = torch.tensor(dist, dtype=torch.float32)
    mask_dist = torch.tensor(mask_dist, dtype=torch.float32)
    targets=torch.tensor(targets, dtype=torch.long)
    src_edge_type = torch.tensor(src_edge_type, dtype=torch.long)
    # pos=torch.tensor(pos, dtype=torch.float32)
    # pos_noise = torch.tensor(pos_noise, dtype=torch.float32)
    bond_adj = torch.tensor(bond_adj, dtype=torch.long)
    return z,mask_z,coord,mask_coord,dist,mask_dist,targets,src_edge_type,bond_adj
    # return {
    #     'z': z,
    #     'mask_z': mask_z,
    #     'coord': coord,
    #     'mask_coord': mask_coord,
    #     'dist': dist,
    #     'mask_dist':mask_dist,
    #     'src_edge_type': src_edge_type,
    #     "targets":targets
    # }

