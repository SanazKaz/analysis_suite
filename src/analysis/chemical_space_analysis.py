"""
Chemical Space Visualization using t-SNE

Visualizes chemical space coverage of different methods using:
1. ECFP4 fingerprints
2. PCA reduction to 50 components
3. t-SNE projection to 2D

Based on Pat Walters' approach for proper chemical space visualization.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Tuple

from src.utils import load_molecules
from src.utils.plotting import set_pub_style, save_figure


def compute_ecfp4_fingerprint_array(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """
    Compute ECFP4 fingerprint as numpy array.
    
    Args:
        mol: RDKit molecule object
        n_bits: Number of bits in fingerprint
        
    Returns:
        Numpy array of fingerprint bits, or None if failed
    """
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None


def sanitize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """Attempt to sanitize a molecule."""
    if mol is None:
        return None
    
    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        return mol_copy
    except Exception:
        pass
    
    try:
        mol_copy = Chem.RWMol(mol)
        for atom in mol_copy.GetAtoms():
            if atom.GetAtomicNum() == 7:
                if atom.GetTotalValence() == 4 and atom.GetFormalCharge() == 0:
                    atom.SetFormalCharge(1)
        Chem.SanitizeMol(mol_copy)
        return mol_copy.GetMol()
    except Exception:
        return None


def load_and_fingerprint(path: Path, n_bits: int = 2048) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load molecules and compute fingerprints.
    
    Returns:
        Tuple of (fingerprint arrays, SMILES strings)
    """
    RDLogger.DisableLog('rdApp.*')
    
    raw_mols = load_molecules(path)
    
    fingerprints = []
    smiles_list = []
    
    for mol in raw_mols:
        sanitized = sanitize_molecule(mol)
        if sanitized is not None:
            fp = compute_ecfp4_fingerprint_array(sanitized, n_bits)
            if fp is not None:
                fingerprints.append(fp)
                try:
                    smiles_list.append(Chem.MolToSmiles(sanitized))
                except:
                    smiles_list.append("N/A")
    
    RDLogger.EnableLog('rdApp.*')
    return fingerprints, smiles_list


def run_chemical_space_analysis(config) -> pd.DataFrame:
    """
    Run chemical space visualization analysis.
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        DataFrame with t-SNE coordinates
    """
    print("\n" + "="*70)
    print("CHEMICAL SPACE ANALYSIS (t-SNE)")
    print("="*70)
    
    output_dir = config.output_dir / 'chemical_space'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_coords = []
    
    for dataset in config.datasets:
        coords_df = process_dataset_tsne(
            dataset,
            output_dir,
            formats=config.plotting.figure_formats,
            dpi=config.plotting.dpi
        )
        if coords_df is not None:
            all_coords.append(coords_df)
    
    print("\nChemical space analysis complete!")
    
    if all_coords:
        combined = pd.concat(all_coords, ignore_index=True)
        coords_path = output_dir / 'tsne_coordinates.csv'
        combined.to_csv(coords_path, index=False)
        print(f"\nSaved coordinates: {coords_path}")
        return combined
    
    return pd.DataFrame()


def process_dataset_tsne(
    dataset,
    output_dir: Path,
    formats: List[str] = None,
    dpi: int = 300,
    n_pca_components: int = 50,
    perplexity: int = 30,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Process a single dataset for t-SNE chemical space visualization.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset.name}")
    print('='*60)
    
    # Collect all fingerprints and labels
    all_fps = []
    all_labels = []
    all_smiles = []
    
    # Load reference molecules
    print(f"\nLoading Reference molecules...")
    ref_fps, ref_smiles = load_and_fingerprint(dataset.reference)
    print(f"  Loaded {len(ref_fps)} molecules")
    
    all_fps.extend(ref_fps)
    all_labels.extend(['Reference'] * len(ref_fps))
    all_smiles.extend(ref_smiles)
    
    # Load each method
    for method_name, sdf_path in dataset.methods.items():
        print(f"\nLoading {method_name}...")
        method_fps, method_smiles = load_and_fingerprint(sdf_path)
        print(f"  Loaded {len(method_fps)} molecules")
        
        all_fps.extend(method_fps)
        all_labels.extend([method_name] * len(method_fps))
        all_smiles.extend(method_smiles)
    
    if len(all_fps) < 100:
        print("  WARNING: Not enough molecules for t-SNE, skipping")
        return None
    
    # Convert to numpy array
    print(f"\nTotal molecules: {len(all_fps)}")
    fp_matrix = np.array(all_fps)
    
    # Step 1: PCA to reduce dimensions
    print(f"\nRunning PCA (2048 → {n_pca_components} dimensions)...")
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    pca_coords = pca.fit_transform(fp_matrix)
    
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  Explained variance: {explained_var:.1f}%")
    
    # Step 2: t-SNE to 2D
    print(f"\nRunning t-SNE ({n_pca_components} → 2 dimensions)...")
    print(f"  This may take a minute...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        init='pca'
    )
    tsne_coords = tsne.fit_transform(pca_coords)
    print(f"  Done!")
    
    # Create DataFrame with results
    coords_df = pd.DataFrame({
        'dataset': dataset.name,
        'method': all_labels,
        'tsne_1': tsne_coords[:, 0],
        'tsne_2': tsne_coords[:, 1],
        'smiles': all_smiles
    })
    
    # Save per-dataset coordinates
    dataset_coords_path = output_dir / f'{dataset.name}_tsne_coords.csv'
    coords_df.to_csv(dataset_coords_path, index=False)
    print(f"\n  Saved coordinates: {dataset_coords_path}")
    
    # Create visualization
    plot_path = output_dir / f'{dataset.name}_chemical_space'
    plot_chemical_space(
        coords_df,
        dataset.name,
        explained_var,
        plot_path,
        formats=formats,
        dpi=dpi
    )
    
    # Print summary
    print_summary(coords_df)
    
    return coords_df


def plot_chemical_space(
    coords_df: pd.DataFrame,
    dataset_name: str,
    explained_var: float,
    output_path: Path,
    formats: List[str] = None,
    dpi: int = 300
):
    """
    Create t-SNE chemical space visualization.
    """
    set_pub_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color and style settings - expanded for multiple PRISM variants
    style_config = {
        'DiffSBDD': {'color': '#E94F37', 'alpha': 0.5, 'size': 20, 'marker': 'o', 'zorder': 2},
        'PRISM': {'color': '#2E86AB', 'alpha': 0.5, 'size': 20, 'marker': 'o', 'zorder': 3},
        'PRISM PoseBusters': {'color': '#2E86AB', 'alpha': 0.5, 'size': 20, 'marker': 'o', 'zorder': 3},
        'PRISM 80-20 HBond Aromatic': {'color': '#9B59B6', 'alpha': 0.5, 'size': 20, 'marker': 's', 'zorder': 4},
        'PRISM 50-50 HBond Aromatic': {'color': '#F39C12', 'alpha': 0.5, 'size': 20, 'marker': '^', 'zorder': 5},
        'PRISM Aromatic Bonus Feature': {'color': '#1ABC9C', 'alpha': 0.5, 'size': 20, 'marker': 'D', 'zorder': 6},
        'PRISM Aromatic Bonus DBSCAN': {'color': '#E74C3C', 'alpha': 0.5, 'size': 20, 'marker': 'v', 'zorder': 7},
        'Reference': {'color': '#2ECC71', 'alpha': 0.9, 'size': 60, 'marker': '*', 'zorder': 10},
    }
    default_style = {'color': '#7D8491', 'alpha': 0.5, 'size': 20, 'marker': 'o', 'zorder': 1}
    
    # Plot each method (reference last so it's on top)
    methods = coords_df['method'].unique()
    plot_order = [m for m in methods if m != 'Reference'] + ['Reference'] if 'Reference' in methods else list(methods)
    
    for method in plot_order:
        mask = coords_df['method'] == method
        style = style_config.get(method, default_style)
        
        n_mols = mask.sum()
        ax.scatter(
            coords_df.loc[mask, 'tsne_1'],
            coords_df.loc[mask, 'tsne_2'],
            c=style['color'],
            alpha=style['alpha'],
            s=style['size'],
            marker=style['marker'],
            label=f"{method} (n={n_mols})",
            zorder=style['zorder'],
            edgecolors='none'
        )
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'Chemical Space: {dataset_name}\n(PCA explained variance: {explained_var:.1f}%)')
    
    # Legend
    legend = ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=10
    )
    
    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, output_path, formats=formats, dpi=dpi)
    plt.close(fig)


def print_summary(coords_df: pd.DataFrame):
    """Print summary statistics about the chemical space."""
    print("\n" + "-"*50)
    print("CHEMICAL SPACE SUMMARY")
    print("-"*50)
    
    for method in coords_df['method'].unique():
        mask = coords_df['method'] == method
        subset = coords_df[mask]
        
        print(f"\n{method}:")
        print(f"  N molecules: {len(subset)}")
        print(f"  t-SNE 1 range: [{subset['tsne_1'].min():.1f}, {subset['tsne_1'].max():.1f}]")
        print(f"  t-SNE 2 range: [{subset['tsne_2'].min():.1f}, {subset['tsne_2'].max():.1f}]")
    
    print("-"*50)

