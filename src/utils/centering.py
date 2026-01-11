"""Molecular centering utilities for pharmacophore scoring."""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from typing import List, Tuple, Optional
import copy


def get_molecule_centroid(mol: Mol) -> np.ndarray:
    """Get the centroid of a molecule's coordinates."""
    if mol is None:
        return None
    conf = mol.GetConformer(0)
    coords = conf.GetPositions()
    return coords.mean(axis=0)


def center_molecule(mol: Mol, centroid: np.ndarray, in_place: bool = False) -> Mol:
    """
    Center a molecule by subtracting a centroid from all atom coordinates.
    
    Args:
        mol: RDKit molecule object
        centroid: The centroid to subtract (numpy array of [x, y, z])
        in_place: If True, modify the molecule in place. If False, work on a copy.
        
    Returns:
        The centered molecule
    """
    if mol is None:
        return None
    
    if not in_place:
        mol = copy.deepcopy(mol)
    
    conf = mol.GetConformer(0)
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (
            pos.x - centroid[0],
            pos.y - centroid[1],
            pos.z - centroid[2]
        ))
    
    return mol


def center_molecules(mols: List[Mol], centroid: np.ndarray, in_place: bool = False) -> List[Mol]:
    """
    Center a list of molecules by subtracting a centroid from all atom coordinates.
    
    Args:
        mols: List of RDKit molecule objects
        centroid: The centroid to subtract (numpy array of [x, y, z])
        in_place: If True, modify molecules in place. If False, work on copies.
        
    Returns:
        List of centered molecules
    """
    return [center_molecule(mol, centroid, in_place) for mol in mols]


def compute_global_centroid(mols: List[Mol]) -> np.ndarray:
    """
    Compute the global centroid across all molecules (mean of all molecule centroids).
    
    Args:
        mols: List of RDKit molecule objects
        
    Returns:
        Global centroid as numpy array [x, y, z]
    """
    centroids = []
    for mol in mols:
        if mol is not None:
            centroid = get_molecule_centroid(mol)
            if centroid is not None:
                centroids.append(centroid)
    
    if not centroids:
        return np.array([0.0, 0.0, 0.0])
    
    return np.mean(centroids, axis=0)

