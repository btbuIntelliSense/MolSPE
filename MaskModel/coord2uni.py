import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix

def coords2unimol(
    atoms,mask_atom,coordinates,mask_coord,pos,pos_noise, targets,dictionary, max_atoms=256, **params
):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param atoms: (list) List of atom symbols.
    :param coordinates: (ndarray) Array of atomic coordinates.
    :param dictionary: (Dictionary) An object that maps atom symbols to unique integers.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation.
    :param params: Additional parameters.

    :return: A dictionary containing the molecular representation with tokens, distances, coordinates, and edge types.
    """
    # atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=remove_hs)
    # atoms = np.array(atoms)
    # coordinates = np.array(pos).astype(np.float32)
    z = np.array(atoms)  # z
    coord =np.array(coordinates).astype(np.float32)  # pos_target
    mask_z = np.array(mask_atom) # masked_z
    mask_coord = np.array(mask_coord).astype(np.float32)  # mask_pos
    targets = np.array(targets)  # targets
    pos = np.array(pos).astype(np.float32)
    pos_noise = np.array(pos_noise).astype(np.float32)

    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    # tokens padding
    src_tokens = np.array(
        [dictionary.bos()]
        + [dictionary.index(atom) for atom in mask_z]
        + [dictionary.eos()]
    )
    # src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    # coordinates normalize & padding
    src_coord = mask_coord - mask_coord.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1, 3)), src_coord, np.zeros((1, 3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(
        1, -1
    )
    return src_tokens,src_distance,src_coord,src_edge_type

    return {
        'src_tokens': src_tokens.astype(int),
        'src_distance': src_distance.astype(np.float32),
        'src_coord': src_coord.astype(np.float32),
        'src_edge_type': src_edge_type.astype(int),
    }

    # return {
    #     'src_tokens': src_tokens.astype(int),
    #     'src_distance': src_distance.astype(np.float32),
    #     'src_coord': src_coord.astype(np.float32),
    #     'src_edge_type': src_edge_type.astype(int),
    # }


