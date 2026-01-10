"""File I/O utilities for molecular data."""

from pathlib import Path
from rdkit import Chem


def load_molecules(path: Path) -> list:
    """
    Load RDKit molecule objects from SDF/PDB file or directory.
    
    Handles:
    - Single SDF file
    - Single PDB file
    - Directory containing multiple SDF files
    - Directory containing multiple PDB files
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of valid RDKit molecule objects
    """
    path = Path(path)
    mols = []
    
    if path.is_file():
        mols = _load_single_file(path)
            
    elif path.is_dir():
        mols = _load_directory(path)
    else:
        print(f"Error: Path does not exist: {path}")
        
    return mols


def _load_single_file(path: Path) -> list:
    """Load molecules from a single file."""
    mols = []
    suffix = path.suffix.lower()
    
    if suffix == '.sdf':
        print(f"Loading molecules from: {path}")
        suppl = Chem.SDMolSupplier(str(path))
        mols = [mol for mol in suppl if mol is not None]
        print(f"  Loaded {len(mols)} molecules")
        
    elif suffix == '.pdb':
        print(f"Loading molecule from PDB: {path}")
        mol = Chem.MolFromPDBFile(str(path), removeHs=False)
        if mol is not None:
            mols = [mol]
            print(f"  Loaded 1 molecule")
        else:
            print(f"  Warning: Could not parse PDB file")
    else:
        print(f"Warning: Unsupported file type: {suffix}")
    
    return mols


def _load_directory(path: Path) -> list:
    """Load molecules from all SDF/PDB files in a directory."""
    mols = []
    
    # Find all supported files
    sdf_files = list(path.glob("*.sdf")) + list(path.rglob("*.sdf"))
    pdb_files = list(path.glob("*.pdb")) + list(path.rglob("*.pdb"))
    
    # Remove duplicates (from glob + rglob)
    sdf_files = list(set(sdf_files))
    pdb_files = list(set(pdb_files))
    
    all_files = sdf_files + pdb_files
    
    if not all_files:
        print(f"Warning: No SDF or PDB files found in {path}")
        return []
    
    print(f"Loading molecules from {len(all_files)} file(s) in: {path}")
    
    # Load SDF files
    for sdf_file in sorted(sdf_files):
        suppl = Chem.SDMolSupplier(str(sdf_file))
        file_mols = [mol for mol in suppl if mol is not None]
        print(f"  {sdf_file.name}: {len(file_mols)} molecules")
        mols.extend(file_mols)
    
    # Load PDB files
    for pdb_file in sorted(pdb_files):
        mol = Chem.MolFromPDBFile(str(pdb_file), removeHs=False)
        if mol is not None:
            print(f"  {pdb_file.name}: 1 molecule")
            mols.append(mol)
        else:
            print(f"  {pdb_file.name}: failed to parse")
    
    print(f"  Total: {len(mols)} molecules")
    return mols


def validate_path(path: str, must_exist: bool = True) -> Path:
    """
    Validate and convert a path string to a Path object.
    
    Args:
        path: String path to validate
        must_exist: If True, raises error if path doesn't exist
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If must_exist is True and path doesn't exist
    """
    p = Path(path)
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return p


def get_smiles_from_mol(mol) -> str:
    """
    Get the canonical SMILES string from an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Canonical SMILES string or empty string if conversion fails
    """
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""