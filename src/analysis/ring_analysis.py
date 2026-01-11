"""Ring and molecular property analysis with violin plot comparisons."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

from src.utils import (
    load_molecules,
    set_pub_style,
    save_figure,
    cliffs_delta,
    pvalue_to_asterisks,
    get_method_color
)


# Property definitions for analysis
RING_PROPERTIES = [
    ('n_rings', 'Number of Rings'),
    ('n_aromatic_rings', 'Aromatic Rings'),
    ('n_aliphatic_rings', 'Aliphatic Rings'),
    ('max_ring_size', 'Max Ring Size'),
]

MOLECULAR_PROPERTIES = [
    ('mw', 'Molecular Weight'),
    ('hba', 'H-Bond Acceptors'),
    ('hbd', 'H-Bond Donors'),
    ('tpsa', 'TPSA'),
    ('rotatable_bonds', 'Rotatable Bonds'),
    ('n_heavy_atoms', 'Heavy Atoms'),
]

# Consistent color palette for all methods
METHOD_COLORS = {
    'Reference': '#37474F',                    # Blue-grey
    'DiffSBDD': '#5C6BC0',                     # Indigo
    'PRISM': '#26A69A',                        # Teal
    'PRISM PoseBusters': '#26A69A',            # Teal
    'PRISM 80/20 HBond Aromatic': '#9B59B6',   # Purple
    'PRISM 50/50 HBond Aromatic': '#F39C12',   # Orange
    'PRISM Aromatic Bonus Feature': '#E91E63', # Pink
    'PRISM Aromatic Bonus DBSCAN': '#E74C3C',  # Red
}


def analyse_molecule(mol) -> dict:
    """
    Extract ring and molecular properties from a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of molecular properties, or None if molecule is invalid
    """
    if mol is None:
        return None
    
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    
    n_aromatic = Lipinski.NumAromaticRings(mol)
    n_aliphatic = Lipinski.NumAliphaticRings(mol)
    
    try:
        mw = Descriptors.MolWt(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        n_atoms = mol.GetNumHeavyAtoms()
    except Exception:
        mw = hba = hbd = rotatable = tpsa = n_atoms = 0
    
    return {
        'n_rings': ring_info.NumRings(),
        'ring_sizes': ring_sizes,
        'max_ring_size': max(ring_sizes) if ring_sizes else 0,
        'n_large_rings': sum(1 for s in ring_sizes if s > 6),
        'n_aromatic_rings': n_aromatic,
        'n_aliphatic_rings': n_aliphatic,
        'mw': mw,
        'hba': hba,
        'hbd': hbd,
        'rotatable_bonds': rotatable,
        'tpsa': tpsa,
        'n_heavy_atoms': n_atoms
    }


def analyse_molecules(molecules: list, source_name: str) -> pd.DataFrame:
    """
    Analyse a list of molecules and return a DataFrame.
    
    Args:
        molecules: List of RDKit molecule objects
        source_name: Name identifier for this set of molecules
        
    Returns:
        DataFrame with molecular properties
    """
    results = []
    for i, mol in enumerate(molecules):
        props = analyse_molecule(mol)
        if props:
            props['mol_idx'] = i
            props['source'] = source_name
            results.append(props)
    
    return pd.DataFrame(results)


def plot_property_violins(
    combined_df: pd.DataFrame,
    dataset_name: str,
    properties: list[tuple[str, str]],
    output_path: Path,
    figure_formats: list[str] = None,
    dpi: int = 300
):
    """
    Create violin plots comparing reference vs methods for multiple properties.
    
    Args:
        combined_df: DataFrame with 'source' column and property columns
        dataset_name: Name of the dataset for the title
        properties: List of (column_name, display_name) tuples
        output_path: Base path for output figure
        figure_formats: List of formats to save
        dpi: Resolution for raster formats
    """
    set_pub_style()
    
    if figure_formats is None:
        figure_formats = ['png', 'svg']
    
    sources = combined_df['source'].unique().tolist()
    n_properties = len(properties)
    
    # Determine layout
    if n_properties <= 3:
        n_cols = n_properties
        n_rows = 1
    elif n_properties <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_properties + 2) // 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_properties == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Use consistent color palette
    colors = METHOD_COLORS.copy()
    fallback_colors = ['#7E57C2', '#EF5350', '#FFA726', '#66BB6A', '#AB47BC', '#29B6F6']
    method_sources = [s for s in sources if s != 'Reference']
    for i, method in enumerate(method_sources):
        if method not in colors:
            colors[method] = fallback_colors[i % len(fallback_colors)]
    
    for idx, (prop_col, prop_name) in enumerate(properties):
        ax = axes[idx]
        
        sns.violinplot(
            data=combined_df,
            x='source',
            y=prop_col,
            order=sources,
            palette=colors,
            cut=0,
            inner='box',
            ax=ax
        )
        
        # Style adjustments
        for collection in ax.collections:
            collection.set_alpha(0.7)
        
        ax.set_xlabel('')
        ax.set_ylabel(prop_name, fontsize=12, fontweight='semibold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add statistical comparisons between methods (not just vs reference)
        stats_text = []
        
        # Compare between the two methods if both exist
        if len(method_sources) >= 2:
            method1, method2 = method_sources[0], method_sources[1]
            data1 = combined_df[combined_df['source'] == method1][prop_col].values
            data2 = combined_df[combined_df['source'] == method2][prop_col].values
            
            if len(data1) > 0 and len(data2) > 0:
                _, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                delta = cliffs_delta(data2, data1)
                asterisks = pvalue_to_asterisks(p_val)
                stats_text.append(f"{method1} vs {method2}:")
                stats_text.append(f"p={p_val:.2e}{asterisks}")
                stats_text.append(f"δ={delta:.2f}")
        
        if stats_text:
            ax.text(
                0.98, 0.98,
                '\n'.join(stats_text),
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
            )
    
    # Hide empty subplots
    for idx in range(len(properties), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'{dataset_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


def generate_summary_statistics(
    combined_df: pd.DataFrame,
    dataset_name: str,
    properties: list[tuple[str, str]]
) -> pd.DataFrame:
    """
    Generate summary statistics comparing methods to reference.
    
    Args:
        combined_df: DataFrame with 'source' column and property columns
        dataset_name: Name of the dataset
        properties: List of (column_name, display_name) tuples
        
    Returns:
        DataFrame with summary statistics
    """
    sources = combined_df['source'].unique().tolist()
    ref_data = combined_df[combined_df['source'] == 'Reference']
    method_sources = [s for s in sources if s != 'Reference']
    
    rows = []
    for prop_col, prop_name in properties:
        ref_vals = ref_data[prop_col].values
        
        row = {
            'Dataset': dataset_name,
            'Property': prop_name,
            'Ref_Mean': np.mean(ref_vals),
            'Ref_Median': np.median(ref_vals),
            'Ref_Std': np.std(ref_vals),
        }
        
        for method in method_sources:
            method_vals = combined_df[combined_df['source'] == method][prop_col].values
            
            if len(method_vals) > 0:
                _, p_val = mannwhitneyu(ref_vals, method_vals, alternative='two-sided')
                delta = cliffs_delta(method_vals, ref_vals)
                
                row[f'{method}_Mean'] = np.mean(method_vals)
                row[f'{method}_Median'] = np.median(method_vals)
                row[f'{method}_Std'] = np.std(method_vals)
                row[f'{method}_p'] = p_val
                row[f'{method}_delta'] = delta
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def process_dataset(dataset, output_dir: Path, config) -> pd.DataFrame:
    """
    Process a single dataset: load molecules, analyse, and generate plots.
    
    Args:
        dataset: Dataset object with reference and methods
        output_dir: Directory to save outputs
        config: AnalysisConfig for plotting settings
        
    Returns:
        Combined DataFrame with all molecular properties
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset.name}")
    print('='*60)
    
    # Load reference molecules
    print(f"\n  Loading reference: {dataset.reference}")
    ref_mols = load_molecules(dataset.reference)
    if not ref_mols:
        print(f"  WARNING: Could not load reference molecules, skipping dataset")
        return None
    
    ref_df = analyse_molecules(ref_mols, 'Reference')
    print(f"  Analysed {len(ref_df)} reference molecules")
    
    # Load method molecules
    method_dfs = [ref_df]
    for method_name, sdf_path in dataset.methods.items():
        print(f"\n  Loading {method_name}: {sdf_path}")
        mols = load_molecules(sdf_path)
        if not mols:
            print(f"  WARNING: Could not load {method_name} molecules, skipping")
            continue
        
        method_df = analyse_molecules(mols, method_name)
        print(f"  Analysed {len(method_df)} molecules")
        method_dfs.append(method_df)
    
    if len(method_dfs) < 2:
        print(f"  WARNING: Not enough data for comparison, skipping dataset")
        return None
    
    # Combine all data
    combined_df = pd.concat(method_dfs, ignore_index=True)
    
    # Save raw data
    csv_path = output_dir / f'{dataset.name}_molecular_properties.csv'
    export_df = combined_df.drop(columns=['ring_sizes'], errors='ignore')
    export_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    
    # Generate ring property violin plots
    print("\n  Generating ring property plots...")
    plot_property_violins(
        combined_df,
        dataset.name,
        RING_PROPERTIES,
        output_dir / f'{dataset.name}_ring_properties',
        figure_formats=config.plotting.figure_formats,
        dpi=config.plotting.dpi
    )
    
    # Generate molecular property violin plots
    print("  Generating molecular property plots...")
    plot_property_violins(
        combined_df,
        dataset.name,
        MOLECULAR_PROPERTIES,
        output_dir / f'{dataset.name}_molecular_properties',
        figure_formats=config.plotting.figure_formats,
        dpi=config.plotting.dpi
    )
    
    return combined_df


def run_ring_analysis(config) -> pd.DataFrame:
    """
    Run complete ring and molecular property analysis.
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        Summary DataFrame with statistics for all datasets
    """
    print("\n" + "="*70)
    print("RING AND MOLECULAR PROPERTY ANALYSIS")
    print("="*70)
    
    output_dir = config.output_dir / 'molecular_properties'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_summaries = []
    all_data = []
    
    for dataset in config.datasets:
        combined_df = process_dataset(dataset, output_dir, config)
        
        if combined_df is not None:
            all_data.append(combined_df)
            
            # Generate summary statistics
            ring_summary = generate_summary_statistics(
                combined_df, dataset.name, RING_PROPERTIES
            )
            mol_summary = generate_summary_statistics(
                combined_df, dataset.name, MOLECULAR_PROPERTIES
            )
            all_summaries.extend([ring_summary, mol_summary])
    
    # Save combined summary
    if all_summaries:
        summary_df = pd.concat(all_summaries, ignore_index=True)
        summary_path = output_dir / 'property_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")
    else:
        summary_df = pd.DataFrame()
    
    print("\nRing and molecular property analysis complete!")
    return summary_df