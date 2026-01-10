"""Radar plot analysis for chemical property distributions.

adapted from 
https://github.com/jianingli-purdue/Benchmarking_gene_model/blob/main/Picture_drawing.ipynb
Yang et al https://pubs.acs.org/doi/10.1021/acs.jmedchem.5c01706

"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

from src.utils import load_molecules, save_figure


# Property denominators from the paper for normalization
PROPERTY_DENOMINATORS = {
    'MW': 500,
    'AliR_C': 4,
    'AroR_C': 3,
    'ChiA_C': 6,
    'SA': 6,
    'NHOH_C': 6,
    'HetA_C': 10,
    'RotB_C': 8,
    'BriA_C': 2,
}

PROPERTY_ORDER = ['MW', 'AliR_C', 'AroR_C', 'ChiA_C', 'SA', 'NHOH_C', 'HetA_C', 'RotB_C', 'BriA_C']


def calculate_sa_score(mol):
    """
    Calculate synthetic accessibility score.
    
    Tries to import sascorer, falls back to a simple heuristic if unavailable.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        SA score (1-10 scale, lower is more accessible)
    """
    try:
        from sascorer import calculateScore
        return calculateScore(mol)
    except ImportError:
        # Fallback: use a simple heuristic based on complexity
        # This is a rough approximation
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        # Simple heuristic: more complex = higher SA score
        score = 2.0 + (n_rings * 0.5) + (n_stereo * 0.3) + (n_bridgehead * 0.5)
        return min(score, 10.0)


def compute_properties_for_mol(mol) -> dict:
    """
    Compute the 9 normalized properties for a single molecule.
    
    Each property is divided by its denominator as per the paper's methodology.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of property_name -> normalized value (0-1 scale)
    """
    if mol is None:
        return None
    
    try:
        # 1) Molecular weight
        mol_wt = Descriptors.MolWt(mol)
        
        # 2) Aliphatic ring count
        aliph_ring_count = Lipinski.NumAliphaticRings(mol)
        
        # 3) Aromatic ring count
        arom_ring_count = Lipinski.NumAromaticRings(mol)
        
        # 4) Chiral atom count
        chiral_count = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        
        # 5) Synthetic accessibility score
        sa_score = calculate_sa_score(mol)
        
        # 6) NH/OH count (hydrogen bond donors)
        nh_oh_count = Lipinski.NHOHCount(mol)
        
        # 7) Heteroatom count (atoms that are not C or H)
        heteroatom_count = sum(1 for atom in mol.GetAtoms() 
                               if atom.GetAtomicNum() not in (1, 6))
        
        # 8) Rotatable bonds
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        
        # 9) Bridgehead atom count
        bridgehead_count = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        
        # Return normalized values (divided by denominators)
        return {
            'MW': mol_wt / PROPERTY_DENOMINATORS['MW'],
            'AliR_C': aliph_ring_count / PROPERTY_DENOMINATORS['AliR_C'],
            'AroR_C': arom_ring_count / PROPERTY_DENOMINATORS['AroR_C'],
            'ChiA_C': chiral_count / PROPERTY_DENOMINATORS['ChiA_C'],
            'SA': sa_score / PROPERTY_DENOMINATORS['SA'],
            'NHOH_C': nh_oh_count / PROPERTY_DENOMINATORS['NHOH_C'],
            'HetA_C': heteroatom_count / PROPERTY_DENOMINATORS['HetA_C'],
            'RotB_C': rot_bonds / PROPERTY_DENOMINATORS['RotB_C'],
            'BriA_C': bridgehead_count / PROPERTY_DENOMINATORS['BriA_C'],
        }
    except Exception as e:
        print(f"  Warning: Could not calculate properties: {e}")
        return None


def compute_mean_properties(molecules: list) -> dict:
    """
    Compute mean normalized properties across a list of molecules.
    
    Args:
        molecules: List of RDKit molecule objects
        
    Returns:
        Dictionary of property_name -> mean normalized value
    """
    properties_list = {prop: [] for prop in PROPERTY_ORDER}
    
    for mol in molecules:
        props = compute_properties_for_mol(mol)
        if props is not None:
            for k, v in props.items():
                properties_list[k].append(v)
    
    # Compute means
    mean_properties = {}
    for k, v_list in properties_list.items():
        if len(v_list) > 0:
            mean_properties[k] = np.mean(v_list)
        else:
            mean_properties[k] = 0.0
    
    return mean_properties


def plot_radar_single(
    property_dict: dict,
    title: str,
    output_path: Path,
    color: str = '#408EC6',
    figure_formats: list[str] = None,
    dpi: int = 300
):
    """
    Plot a single radar chart matching the paper's style.
    
    Args:
        property_dict: Dictionary of property_name -> normalized value
        title: Plot title
        output_path: Path for output figure
        color: Fill/line color
        figure_formats: List of formats to save
        dpi: Resolution for raster formats
    """
    if figure_formats is None:
        figure_formats = ['png', 'svg']
    
    # Extract values in correct order
    values = [property_dict[cat] for cat in PROPERTY_ORDER]
    
    # Number of variables
    N = len(PROPERTY_ORDER)
    
    # Compute angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the plot
    values += values[:1]  # Close the plot
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    
    # Transparent background
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Start from top, go clockwise
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PROPERTY_ORDER, fontsize=10, fontweight='bold')
    
    # Radial axis settings
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=8)
    
    # Style the boundary and grid
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2)
    ax.xaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.yaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    
    # Plot data
    ax.plot(angles, values, color=color, linewidth=3, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.3)
    
    # Title
    ax.set_title(title, size=14, y=1.08, fontweight='bold', color='black')
    
    plt.tight_layout()
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


def plot_radar_grid(
    all_properties: dict,
    output_path: Path,
    figure_formats: list[str] = None,
    dpi: int = 300
):
    """
    Create a grid of radar plots like the paper figure.
    
    Args:
        all_properties: Dict mapping source names to their property dicts
        output_path: Path for output figure
        figure_formats: List of formats to save
        dpi: Resolution for raster formats
    """
    if figure_formats is None:
        figure_formats = ['png', 'svg']
    
    sources = list(all_properties.keys())
    n_sources = len(sources)
    
    # Determine grid layout
    n_cols = min(3, n_sources)
    n_rows = (n_sources + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        subplot_kw={'polar': True}
    )
    
    if n_sources == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Refined color scheme - expanded for multiple PRISM variants
    colors = {
        'Reference': '#37474F',                    # Blue-grey
        'DiffSBDD': '#5C6BC0',                     # Indigo
        'PRISM': '#26A69A',                        # Teal
        'PRISM PoseBusters': '#26A69A',            # Teal
        'PRISM 80/20 HBond Aromatic': '#9B59B6',   # Purple
        'PRISM 50/50 HBond Aromatic': '#F39C12',   # Orange
        'PRISM Aromatic Bonus Feature': '#1ABC9C', # Turquoise
        'PRISM Aromatic Bonus DBSCAN': '#E74C3C',  # Red
    }
    default_color = '#7E57C2'
    
    N = len(PROPERTY_ORDER)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    for idx, source in enumerate(sources):
        ax = axes[idx]
        property_dict = all_properties[source]
        values = [property_dict[cat] for cat in PROPERTY_ORDER]
        values += values[:1]
        
        color = colors.get(source, default_color)
        
        # Transparent background
        ax.set_facecolor('none')
        
        # Start from top, go clockwise
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(PROPERTY_ORDER, fontsize=9, fontweight='bold')
        
        # Radial settings
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=7)
        
        # Grid styling
        ax.spines['polar'].set_color('black')
        ax.spines['polar'].set_linewidth(2)
        ax.xaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
        ax.yaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
        
        # Plot
        ax.plot(angles, values, color=color, linewidth=3, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.3)
        
        ax.set_title(source, size=12, y=1.08, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_sources, len(axes)):
        axes[idx].set_visible(False)
    
    fig.patch.set_facecolor('none')
    plt.tight_layout()
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


def plot_radar_overlay(
    all_properties: dict,
    title: str,
    output_path: Path,
    figure_formats: list[str] = None,
    dpi: int = 300
):
    """
    Create a single radar plot with all sources overlaid for comparison.
    
    Args:
        all_properties: Dict mapping source names to their property dicts
        title: Plot title
        output_path: Path for output figure
        figure_formats: List of formats to save
        dpi: Resolution for raster formats
    """
    if figure_formats is None:
        figure_formats = ['png', 'svg']
    
    N = len(PROPERTY_ORDER)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    
    # Refined colors for overlay - expanded for multiple PRISM variants
    colors = {
        'Reference': '#37474F',                    # Blue-grey
        'DiffSBDD': '#5C6BC0',                     # Indigo
        'PRISM': '#26A69A',                        # Teal
        'PRISM PoseBusters': '#26A69A',            # Teal
        'PRISM 80/20 HBond Aromatic': '#9B59B6',   # Purple
        'PRISM 50/50 HBond Aromatic': '#F39C12',   # Orange
        'PRISM Aromatic Bonus Feature': '#1ABC9C', # Turquoise
        'PRISM Aromatic Bonus DBSCAN': '#E74C3C',  # Red
    }
    default_colors = ['#7E57C2', '#EF5350', '#FFA726', '#66BB6A', '#AB47BC', '#29B6F6']
    
    # Transparent background
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Start from top, go clockwise
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Plot each source
    color_idx = 0
    for source, property_dict in all_properties.items():
        values = [property_dict[cat] for cat in PROPERTY_ORDER]
        values += values[:1]
        
        if source in colors:
            color = colors[source]
        else:
            color = default_colors[color_idx % len(default_colors)]
            color_idx += 1
        
        ax.plot(angles, values, color=color, linewidth=2.5, linestyle='solid', label=source)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PROPERTY_ORDER, fontsize=11, fontweight='bold')
    
    # Radial settings
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=9)
    
    # Grid styling
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2)
    ax.xaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.yaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    
    ax.set_title(title, size=14, y=1.08, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(-0.3, 1.0), frameon=True)
    
    plt.tight_layout()
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


def process_dataset_radar(dataset, output_dir: Path, config) -> dict:
    """
    Process a single dataset for radar analysis.
    
    Args:
        dataset: Dataset object with reference and methods
        output_dir: Directory to save outputs
        config: AnalysisConfig for plotting settings
        
    Returns:
        Dict mapping source names to their mean property dicts
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset.name}")
    print('='*60)
    
    all_properties = {}
    all_raw_data = []
    
    # Process reference
    print(f"\n  Loading reference: {dataset.reference}")
    ref_mols = load_molecules(dataset.reference)
    if ref_mols:
        mean_props = compute_mean_properties(ref_mols)
        all_properties['Reference'] = mean_props
        print(f"  Computed properties for {len(ref_mols)} reference molecules")
        
        # Store raw data
        for mol in ref_mols:
            props = compute_properties_for_mol(mol)
            if props:
                props['source'] = 'Reference'
                all_raw_data.append(props)
    
    # Process methods
    for method_name, sdf_path in dataset.methods.items():
        print(f"\n  Loading {method_name}: {sdf_path}")
        mols = load_molecules(sdf_path)
        if not mols:
            print(f"  WARNING: Could not load molecules, skipping")
            continue
        
        mean_props = compute_mean_properties(mols)
        all_properties[method_name] = mean_props
        print(f"  Computed properties for {len(mols)} molecules")
        
        # Store raw data
        for mol in mols:
            props = compute_properties_for_mol(mol)
            if props:
                props['source'] = method_name
                all_raw_data.append(props)
    
    if not all_properties:
        return {}
    
    # Save raw data
    if all_raw_data:
        raw_df = pd.DataFrame(all_raw_data)
        csv_path = output_dir / f'{dataset.name}_radar_raw.csv'
        raw_df.to_csv(csv_path, index=False)
        print(f"\n  Saved raw data: {csv_path}")
    
    # Save mean properties
    mean_df = pd.DataFrame(all_properties).T
    mean_df.index.name = 'Source'
    mean_csv = output_dir / f'{dataset.name}_radar_means.csv'
    mean_df.to_csv(mean_csv)
    print(f"  Saved means: {mean_csv}")
    
    # Generate plots
    print("\n  Generating radar plots...")
    
    # Individual plots for each source
    source_colors = {
        'Reference': '#37474F', 
        'DiffSBDD': '#5C6BC0', 
        'PRISM': '#26A69A',
        'PRISM PoseBusters': '#26A69A',
        'PRISM 80/20 HBond Aromatic': '#9B59B6',
        'PRISM 50/50 HBond Aromatic': '#F39C12',
        'PRISM Aromatic Bonus Feature': '#1ABC9C',
        'PRISM Aromatic Bonus DBSCAN': '#E74C3C',
    }
    for source, props in all_properties.items():
        color = source_colors.get(source, '#7E57C2')
        plot_radar_single(
            props,
            f"{dataset.name} - {source}",
            output_dir / f'{dataset.name}_{source}_radar',
            color=color,
            figure_formats=config.plotting.figure_formats,
            dpi=config.plotting.dpi
        )
    
    # Grid of all sources
    plot_radar_grid(
        all_properties,
        output_dir / f'{dataset.name}_radar_grid',
        figure_formats=config.plotting.figure_formats,
        dpi=config.plotting.dpi
    )
    
    # Overlay comparison
    plot_radar_overlay(
        all_properties,
        dataset.name,
        output_dir / f'{dataset.name}_radar_overlay',
        figure_formats=config.plotting.figure_formats,
        dpi=config.plotting.dpi
    )
    
    return all_properties


def run_radar_analysis(config) -> pd.DataFrame:
    """
    Run complete radar plot analysis.
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        Summary DataFrame with mean properties for all datasets/sources
    """
    print("\n" + "="*70)
    print("RADAR PLOT ANALYSIS")
    print("="*70)
    
    output_dir = config.output_dir / 'radar'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    summary_rows = []
    
    for dataset in config.datasets:
        properties = process_dataset_radar(dataset, output_dir, config)
        if properties:
            all_results[dataset.name] = properties
            
            # Add to summary
            for source, props in properties.items():
                row = {'Dataset': dataset.name, 'Source': source}
                row.update(props)
                summary_rows.append(row)
    
    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / 'radar_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")
    else:
        summary_df = pd.DataFrame()
    
    print("\nRadar analysis complete!")
    return summary_df