"""
Tanimoto Similarity Analysis using ECFP4 Fingerprints

Compares chemical space coverage between methods by computing
Tanimoto similarity of ECFP4 fingerprints to reference molecules.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs
from typing import List, Tuple
import matplotlib.pyplot as plt

from src.utils import load_molecules
from src.utils.plotting import set_pub_style, save_figure


def compute_ecfp4_fingerprint(mol: Chem.Mol, n_bits: int = 2048):
    """
    Compute ECFP4 fingerprint for a molecule.
    
    Args:
        mol: RDKit molecule object
        n_bits: Number of bits in fingerprint (default 2048)
        
    Returns:
        Morgan fingerprint bit vector or None if failed
    """
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    except Exception:
        return None


def compute_fingerprints(molecules: List[Chem.Mol], n_bits: int = 2048) -> List:
    """
    Compute ECFP4 fingerprints for a list of molecules.
    
    Args:
        molecules: List of RDKit molecule objects
        n_bits: Number of bits in fingerprint
        
    Returns:
        List of fingerprints (None entries filtered out)
    """
    fingerprints = []
    for mol in molecules:
        if mol is not None:
            fp = compute_ecfp4_fingerprint(mol, n_bits)
            if fp is not None:
                fingerprints.append(fp)
    return fingerprints


def compute_nearest_neighbor_similarities(
    query_fps: List, 
    reference_fps: List
) -> np.ndarray:
    """
    For each query fingerprint, compute max Tanimoto similarity to any reference.
    
    This gives the "nearest neighbor" similarity - how close is each
    generated molecule to the closest known binder.
    
    Args:
        query_fps: List of query fingerprints
        reference_fps: List of reference fingerprints
        
    Returns:
        Array of max similarities for each query
    """
    if not query_fps or not reference_fps:
        return np.array([])
    
    similarities = []
    for query_fp in query_fps:
        # Compute Tanimoto to all references and take max
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps)
        similarities.append(max(sims))
    
    return np.array(similarities)


def compute_pairwise_diversity(fingerprints: List) -> float:
    """
    Compute average pairwise Tanimoto diversity (1 - similarity).
    
    Args:
        fingerprints: List of fingerprints
        
    Returns:
        Average diversity score
    """
    if len(fingerprints) < 2:
        return 0.0
    
    total_div = 0.0
    count = 0
    
    for i in range(len(fingerprints)):
        # Compute similarities to all subsequent fingerprints
        sims = DataStructs.BulkTanimotoSimilarity(
            fingerprints[i], 
            fingerprints[i+1:]
        )
        total_div += sum(1 - s for s in sims)
        count += len(sims)
    
    return total_div / count if count > 0 else 0.0


def sanitize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Attempt to sanitize a molecule, handling common issues.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Sanitized molecule or None if failed
    """
    if mol is None:
        return None
    
    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        return mol_copy
    except Exception:
        pass
    
    # Try fixing nitrogen charges
    try:
        mol_copy = Chem.RWMol(mol)
        for atom in mol_copy.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                if atom.GetTotalValence() == 4 and atom.GetFormalCharge() == 0:
                    atom.SetFormalCharge(1)
        Chem.SanitizeMol(mol_copy)
        return mol_copy.GetMol()
    except Exception:
        return None


def load_and_process_molecules(path: Path) -> Tuple[List[Chem.Mol], List]:
    """
    Load molecules and compute their ECFP4 fingerprints.
    
    Args:
        path: Path to SDF file or directory
        
    Returns:
        Tuple of (sanitized molecules, fingerprints)
    """
    RDLogger.DisableLog('rdApp.*')
    
    raw_mols = load_molecules(path)
    
    valid_mols = []
    fingerprints = []
    
    for mol in raw_mols:
        sanitized = sanitize_molecule(mol)
        if sanitized is not None:
            fp = compute_ecfp4_fingerprint(sanitized)
            if fp is not None:
                valid_mols.append(sanitized)
                fingerprints.append(fp)
    
    RDLogger.EnableLog('rdApp.*')
    return valid_mols, fingerprints


def plot_similarity_histograms(
    method_similarities: dict,
    dataset_name: str,
    output_path: Path,
    formats: List[str] = None,
    dpi: int = 300
):
    """
    Create overlaid histograms of nearest-neighbor similarities.
    
    Args:
        method_similarities: Dict of {method_name: similarity_array}
        dataset_name: Name of the dataset for title
        output_path: Base path for saving (without extension)
        formats: List of output formats
        dpi: Resolution for raster formats
    """
    set_pub_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette - expanded for multiple PRISM variants
    colors = {
        'PRISM': '#2E86AB',                    # Steel blue
        'PRISM PoseBusters': '#2E86AB',        # Steel blue
        'DiffSBDD': '#E94F37',                 # Coral red
        'PRISM 80-20 HBond Aromatic': '#9B59B6',  # Purple
        'PRISM 50-50 HBond Aromatic': '#F39C12',  # Orange
        'PRISM Aromatic Bonus Feature': '#1ABC9C', # Teal
        'PRISM Aromatic Bonus DBSCAN': '#E74C3C',  # Red
        'Reference': '#7D8491',                # Gray
    }
    default_colors = ['#2E86AB', '#E94F37', '#44AF69', '#F18F01', '#C73E1D', '#9B59B6', '#1ABC9C']
    
    bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
    
    for idx, (method_name, similarities) in enumerate(method_similarities.items()):
        if len(similarities) == 0:
            continue
            
        color = colors.get(method_name, default_colors[idx % len(default_colors)])
        
        ax.hist(
            similarities, 
            bins=bins, 
            alpha=0.6, 
            label=f'{method_name} (n={len(similarities)}, μ={np.mean(similarities):.3f})',
            color=color,
            edgecolor='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('Tanimoto Similarity to Nearest Reference')
    ax.set_ylabel('Count')
    ax.set_title(f'ECFP4 Tanimoto Similarity Distribution\n{dataset_name}')
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.set_xlim(0, 1)
    
    # Add vertical lines for common thresholds
    ax.axvline(x=0.4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.41, ax.get_ylim()[1] * 0.95, 'Tc=0.4', fontsize=9, color='gray')
    ax.text(0.71, ax.get_ylim()[1] * 0.95, 'Tc=0.7', fontsize=9, color='gray')
    
    plt.tight_layout()
    save_figure(fig, output_path, formats=formats, dpi=dpi)
    plt.close(fig)


def plot_cross_method_histograms(
    cross_similarities: dict,
    dataset_name: str,
    output_path: Path,
    formats: List[str] = None,
    dpi: int = 300
):
    """
    Create overlaid histograms of cross-method nearest-neighbor similarities.
    
    Shows how similar molecules from one method are to molecules from another.
    
    Args:
        cross_similarities: Dict of {"MethodA → MethodB": similarity_array}
        dataset_name: Name of the dataset for title
        output_path: Base path for saving (without extension)
        formats: List of output formats
        dpi: Resolution for raster formats
    """
    set_pub_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette for cross comparisons
    colors = ['#2E86AB', '#E94F37', '#44AF69', '#F18F01']
    
    bins = np.linspace(0, 1, 41)  # 40 bins from 0 to 1
    
    for idx, (comparison, similarities) in enumerate(cross_similarities.items()):
        if len(similarities) == 0:
            continue
        
        color = colors[idx % len(colors)]
        
        ax.hist(
            similarities, 
            bins=bins, 
            alpha=0.5, 
            label=f'{comparison} (μ={np.mean(similarities):.3f})',
            color=color,
            edgecolor='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('Tanimoto Similarity (Nearest Neighbor in Other Method)')
    ax.set_ylabel('Count')
    ax.set_title(f'Cross-Method Chemical Space Overlap (ECFP4)\n{dataset_name}')
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.set_xlim(0, 1)
    
    # Add interpretation guide
    ax.axvline(x=0.4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.41, ax.get_ylim()[1] * 0.95, 'Tc=0.4\n(similar)', fontsize=8, color='gray')
    
    # Add annotation explaining the plot
    textstr = 'Higher values = more overlap in chemical space'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    save_figure(fig, output_path, formats=formats, dpi=dpi)
    plt.close(fig)


def run_tanimoto_analysis(config) -> pd.DataFrame:
    """
    Run Tanimoto similarity analysis from config.
    
    Compares chemical space overlap between methods by computing
    cross-method nearest-neighbor similarities.
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        Combined results DataFrame
    """
    print("\n" + "="*70)
    print("TANIMOTO SIMILARITY ANALYSIS (ECFP4)")
    print("Cross-method chemical space comparison")
    print("="*70)
    
    output_dir = config.output_dir / 'tanimoto'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for dataset in config.datasets:
        result_df = process_dataset_tanimoto(
            dataset, 
            output_dir,
            formats=config.plotting.figure_formats,
            dpi=config.plotting.dpi
        )
        if result_df is not None:
            all_results.append(result_df)
    
    print("\nTanimoto similarity analysis complete!")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        summary_path = output_dir / 'tanimoto_summary.csv'
        combined.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")
        return combined
    
    return pd.DataFrame()


def process_dataset_tanimoto(
    dataset, 
    output_dir: Path,
    formats: List[str] = None,
    dpi: int = 300
) -> pd.DataFrame:
    """
    Process a single dataset for cross-method Tanimoto similarity analysis.
    
    Compares each method's molecules to the other method's molecules
    to see if they're exploring similar or different chemical space.
    
    Args:
        dataset: Dataset object with name, reference, and methods
        output_dir: Output directory for results
        formats: List of output formats for figures
        dpi: Resolution for raster formats
        
    Returns:
        DataFrame with similarity statistics
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset.name}")
    print('='*60)
    
    # Load all methods' molecules and fingerprints
    method_data = {}
    
    for method_name, sdf_path in dataset.methods.items():
        print(f"\nLoading {method_name}...")
        mols, fps = load_and_process_molecules(sdf_path)
        print(f"  Loaded {len(mols)} molecules with valid fingerprints")
        
        if fps:
            method_data[method_name] = {'mols': mols, 'fps': fps}
        else:
            print(f"  WARNING: No valid fingerprints for {method_name}, skipping")
    
    if len(method_data) < 2:
        print("  WARNING: Need at least 2 methods for comparison, skipping dataset")
        return None
    
    method_names = list(method_data.keys())
    stats_rows = []
    cross_similarities = {}
    
    # Compute cross-method similarities (each method vs the other)
    for i, method_a in enumerate(method_names):
        for method_b in method_names:
            if method_a == method_b:
                continue
            
            print(f"\nComputing {method_a} → {method_b} nearest-neighbor similarities...")
            
            fps_a = method_data[method_a]['fps']
            fps_b = method_data[method_b]['fps']
            
            # For each molecule in A, find nearest neighbor in B
            nn_sims = compute_nearest_neighbor_similarities(fps_a, fps_b)
            
            label = f"{method_a} → {method_b}"
            cross_similarities[label] = nn_sims
            
            # Compute intra-method diversity for method A
            if i == 0:  # Only compute once per method
                diversity_a = compute_pairwise_diversity(fps_a[:500])  # Sample for speed
            else:
                diversity_a = compute_pairwise_diversity(method_data[method_a]['fps'][:500])
            
            stats = {
                'dataset': dataset.name,
                'comparison': label,
                'n_query': len(nn_sims),
                'n_target': len(fps_b),
                'mean_nn_similarity': np.mean(nn_sims),
                'median_nn_similarity': np.median(nn_sims),
                'std_nn_similarity': np.std(nn_sims),
                'min_nn_similarity': np.min(nn_sims),
                'max_nn_similarity': np.max(nn_sims),
                'pct_above_0.3': (nn_sims >= 0.3).mean() * 100,
                'pct_above_0.4': (nn_sims >= 0.4).mean() * 100,
                'pct_above_0.5': (nn_sims >= 0.5).mean() * 100,
                'pct_above_0.7': (nn_sims >= 0.7).mean() * 100,
            }
            stats_rows.append(stats)
            
            print(f"  Mean NN similarity: {stats['mean_nn_similarity']:.3f} ± {stats['std_nn_similarity']:.3f}")
            print(f"  Median: {stats['median_nn_similarity']:.3f}")
            print(f"  Range: [{stats['min_nn_similarity']:.3f}, {stats['max_nn_similarity']:.3f}]")
            print(f"  % with match ≥ 0.4: {stats['pct_above_0.4']:.1f}%")
            print(f"  % with match ≥ 0.7: {stats['pct_above_0.7']:.1f}%")
    
    # Also compute intra-method diversity for context
    print(f"\nComputing intra-method diversity (sampled)...")
    for method_name in method_names:
        fps = method_data[method_name]['fps']
        # Sample for speed if large
        sample_fps = fps[:min(500, len(fps))]
        diversity = compute_pairwise_diversity(sample_fps)
        print(f"  {method_name} internal diversity: {diversity:.3f}")
    
    # Create histogram comparing cross-method similarities
    plot_path = output_dir / f'{dataset.name}_tanimoto_cross_method'
    plot_cross_method_histograms(
        cross_similarities, 
        dataset.name, 
        plot_path,
        formats=formats,
        dpi=dpi
    )
    
    # Print comparison table
    print_cross_method_table(dataset.name, stats_rows)
    
    return pd.DataFrame(stats_rows)


def print_comparison_table(dataset_name: str, stats_rows: List[dict]):
    """Print a formatted comparison table."""
    print("\n" + "-"*70)
    print(f"TANIMOTO SIMILARITY SUMMARY - {dataset_name}")
    print("-"*70)
    
    header = f"{'Method':<15} {'N':>6} {'Mean':>8} {'Median':>8} {'≥0.4%':>8} {'≥0.7%':>8} {'Diversity':>10}"
    print(header)
    print("-"*70)
    
    for stats in stats_rows:
        row = (
            f"{stats['method']:<15} "
            f"{stats['n_molecules']:>6} "
            f"{stats['mean_similarity']:>8.3f} "
            f"{stats['median_similarity']:>8.3f} "
            f"{stats['pct_above_0.4']:>7.1f}% "
            f"{stats['pct_above_0.7']:>7.1f}% "
            f"{stats['intra_diversity']:>10.3f}"
        )
        print(row)
    
    print("-"*70)


def print_cross_method_table(dataset_name: str, stats_rows: List[dict]):
    """Print a formatted cross-method comparison table."""
    print("\n" + "-"*80)
    print(f"CROSS-METHOD TANIMOTO SIMILARITY - {dataset_name}")
    print("-"*80)
    
    header = f"{'Comparison':<20} {'N Query':>8} {'N Target':>8} {'Mean':>8} {'Median':>8} {'≥0.4%':>8} {'≥0.7%':>8}"
    print(header)
    print("-"*80)
    
    for stats in stats_rows:
        row = (
            f"{stats['comparison']:<20} "
            f"{stats['n_query']:>8} "
            f"{stats['n_target']:>8} "
            f"{stats['mean_nn_similarity']:>8.3f} "
            f"{stats['median_nn_similarity']:>8.3f} "
            f"{stats['pct_above_0.4']:>7.1f}% "
            f"{stats['pct_above_0.7']:>7.1f}%"
        )
        print(row)
    
    print("-"*80)
    print("\nInterpretation:")
    print("  - Higher similarity = methods generating in similar chemical space")
    print("  - Lower similarity = methods exploring different regions")
    print("  - Tc ≥ 0.4 typically indicates structurally similar compounds")

