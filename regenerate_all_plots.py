#!/usr/bin/env python
"""Regenerate all plots from existing CSVs with consistent styling."""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from pathlib import Path

# Import plotting utilities
import sys
sys.path.insert(0, '/Users/sanazkazeminia/Documents/analysis_suite')
from src.utils.plotting import save_figure

# Consistent color palette across all analyses
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

def set_style():
    """Set consistent plot style."""
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelweight": "semibold",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.family": "sans-serif",
    })

# ============================================================================
# RADAR PLOT
# ============================================================================

PROPERTY_ORDER = ['MW', 'AliR_C', 'AroR_C', 'ChiA_C', 'SA', 'NHOH_C', 'HetA_C', 'RotB_C', 'BriA_C']

def regenerate_radar_overlay():
    """Regenerate the radar overlay plot from CSV."""
    print("\n" + "="*60)
    print("Regenerating Radar Overlay Plot")
    print("="*60)
    
    csv_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/radar/AMPC_beta_lactamase_radar_means.csv')
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path, index_col=0)
    print(f"Loaded data for {len(df)} methods: {list(df.index)}")
    
    all_properties = {}
    for source in df.index:
        all_properties[source] = {prop: df.loc[source, prop] for prop in PROPERTY_ORDER}
    
    N = len(PROPERTY_ORDER)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'polar': True})
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    for source, property_dict in all_properties.items():
        values = [property_dict[cat] for cat in PROPERTY_ORDER]
        values += values[:1]
        
        color = METHOD_COLORS.get(source, '#7D8491')
        ax.plot(angles, values, color=color, linewidth=2.5, linestyle='solid', label=source)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PROPERTY_ORDER, fontsize=11, fontweight='bold')
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=9)
    
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2)
    ax.xaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.yaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    
    ax.set_title('AMPC Beta-Lactamase - Property Comparison', size=14, y=1.08, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), frameon=True, fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    output_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/radar/AMPC_beta_lactamase_radar_overlay')
    save_figure(fig, output_path, formats=['png', 'svg'], dpi=300)
    plt.close()
    
    print("✓ Radar overlay regenerated!")


# ============================================================================
# T-SNE PLOT
# ============================================================================

def regenerate_tsne_plot():
    """Regenerate the t-SNE chemical space plot from CSV."""
    print("\n" + "="*60)
    print("Regenerating t-SNE Chemical Space Plot")
    print("="*60)
    
    csv_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/chemical_space/AMPC_beta_lactamase_tsne_coords.csv')
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    coords_df = pd.read_csv(csv_path)
    print(f"Loaded {len(coords_df)} data points")
    
    set_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Style config with markers
    style_config = {
        'Reference': {'marker': '*', 'size': 80, 'alpha': 0.9, 'zorder': 10},
        'DiffSBDD': {'marker': 'o', 'size': 20, 'alpha': 0.5, 'zorder': 2},
        'PRISM PoseBusters': {'marker': 'o', 'size': 20, 'alpha': 0.5, 'zorder': 3},
        'PRISM 80/20 HBond Aromatic': {'marker': 's', 'size': 20, 'alpha': 0.5, 'zorder': 4},
        'PRISM 50/50 HBond Aromatic': {'marker': '^', 'size': 20, 'alpha': 0.5, 'zorder': 5},
        'PRISM Aromatic Bonus Feature': {'marker': 'D', 'size': 20, 'alpha': 0.5, 'zorder': 6},
        'PRISM Aromatic Bonus DBSCAN': {'marker': 'v', 'size': 20, 'alpha': 0.5, 'zorder': 7},
    }
    default_style = {'marker': 'o', 'size': 20, 'alpha': 0.5, 'zorder': 1}
    
    methods = coords_df['method'].unique()
    plot_order = [m for m in methods if m != 'Reference'] + ['Reference'] if 'Reference' in methods else list(methods)
    
    for method in plot_order:
        mask = coords_df['method'] == method
        style = style_config.get(method, default_style)
        color = METHOD_COLORS.get(method, '#7D8491')
        
        n_mols = mask.sum()
        ax.scatter(
            coords_df.loc[mask, 'tsne_1'],
            coords_df.loc[mask, 'tsne_2'],
            c=color,
            alpha=style['alpha'],
            s=style['size'],
            marker=style['marker'],
            label=f"{method} (n={n_mols})",
            zorder=style['zorder'],
            edgecolors='none'
        )
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('Chemical Space: AMPC Beta-Lactamase\n(PCA + t-SNE)', fontsize=14, fontweight='bold')
    
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=10
    )
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    
    output_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/chemical_space/AMPC_beta_lactamase_chemical_space')
    save_figure(fig, output_path, formats=['png', 'svg'], dpi=300)
    plt.close()
    
    print("✓ t-SNE plot regenerated!")


# ============================================================================
# MOLECULAR PROPERTIES PLOTS
# ============================================================================

def regenerate_molecular_properties_plots():
    """Regenerate molecular properties plots with better organization."""
    print("\n" + "="*60)
    print("Regenerating Molecular Properties Plots")
    print("="*60)
    
    csv_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/molecular_properties/AMPC_beta_lactamase_molecular_properties.csv')
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} molecules")
    print(f"Methods: {df['source'].unique()}")
    
    set_style()
    
    # Get method order (Reference first, then others)
    sources = df['source'].unique().tolist()
    if 'Reference' in sources:
        sources.remove('Reference')
        sources = ['Reference'] + sorted(sources)
    
    # Create color palette for violin plots
    palette = {s: METHOD_COLORS.get(s, '#7D8491') for s in sources}
    
    # =========================================================================
    # PLOT 1: Drug-likeness Metrics (QED, SA - need to compute if not present)
    # We'll use what's available in the ring analysis CSV
    # =========================================================================
    
    # Plot ring properties
    ring_props = [
        ('n_rings', 'Number of Rings'),
        ('n_aromatic_rings', 'Aromatic Rings'),
        ('n_aliphatic_rings', 'Aliphatic Rings'),
        ('max_ring_size', 'Max Ring Size'),
    ]
    
    available_ring_props = [(col, name) for col, name in ring_props if col in df.columns]
    
    if available_ring_props:
        n_props = len(available_ring_props)
        fig, axes = plt.subplots(1, n_props, figsize=(4 * n_props, 5))
        if n_props == 1:
            axes = [axes]
        
        for idx, (col, name) in enumerate(available_ring_props):
            ax = axes[idx]
            sns.violinplot(
                data=df, x='source', y=col, order=sources,
                palette=palette, cut=0, inner='box', ax=ax
            )
            for collection in ax.collections:
                collection.set_alpha(0.7)
            ax.set_xlabel('')
            ax.set_ylabel(name, fontsize=11, fontweight='semibold')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        fig.suptitle('Ring Properties - AMPC Beta-Lactamase', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/molecular_properties/AMPC_beta_lactamase_ring_properties')
        save_figure(fig, output_path, formats=['png', 'svg'], dpi=300)
        plt.close()
        print("✓ Ring properties plot regenerated!")
    
    # =========================================================================
    # PLOT 2: Structural Properties
    # =========================================================================
    
    struct_props = [
        ('mw', 'Mol. Weight'),
        ('hba', 'H-Bond Acc.'),
        ('hbd', 'H-Bond Don.'),
        ('tpsa', 'TPSA'),
        ('rotatable_bonds', 'Rot. Bonds'),
        ('n_heavy_atoms', 'Heavy Atoms'),
    ]
    
    available_struct_props = [(col, name) for col, name in struct_props if col in df.columns]
    
    if available_struct_props:
        n_props = len(available_struct_props)
        n_cols = 3
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        
        for idx, (col, name) in enumerate(available_struct_props):
            ax = axes[idx]
            sns.violinplot(
                data=df, x='source', y=col, order=sources,
                palette=palette, cut=0, inner='box', ax=ax
            )
            for collection in ax.collections:
                collection.set_alpha(0.7)
            ax.set_xlabel('')
            ax.set_ylabel(name, fontsize=11, fontweight='semibold')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Hide empty subplots
        for idx in range(len(available_struct_props), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('Structural Properties - AMPC Beta-Lactamase', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/molecular_properties/AMPC_beta_lactamase_molecular_properties')
        save_figure(fig, output_path, formats=['png', 'svg'], dpi=300)
        plt.close()
        print("✓ Structural properties plot regenerated!")


# ============================================================================
# MOLECULAR PROPERTIES COMPARISON TABLE (from molecular_props CSV)
# ============================================================================

def regenerate_properties_table_plot():
    """Create a visual table/heatmap of the properties comparison."""
    print("\n" + "="*60)
    print("Regenerating Properties Comparison Visualization")
    print("="*60)
    
    csv_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/molecular_props/AMPC_beta_lactamase_properties_comparison.csv')
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} properties")
    
    # Extract just mean values for visualization
    properties = df['Property'].tolist()
    
    # Get method columns (those without _p or _sig suffix)
    value_cols = [col for col in df.columns if col not in ['Dataset', 'Property'] 
                  and not col.endswith('_p') and not col.endswith('_sig')]
    
    # Create a cleaner visualization - bar chart comparison
    set_style()
    
    # Select key properties for visualization
    key_properties = ['QED', 'SA Score', 'LogP', 'Molecular Weight', 'H-Bond Donors', 
                      'H-Bond Acceptors', 'N Aromatic Rings', 'N Rings']
    
    key_df = df[df['Property'].isin(key_properties)].copy()
    
    if len(key_df) == 0:
        print("No key properties found in data")
        return
    
    # Parse the "mean ± std" format to get just means
    def parse_mean(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str) and '±' in val:
            return float(val.split('±')[0].strip())
        try:
            return float(val)
        except:
            return np.nan
    
    # Create figure with subplots for each property
    n_props = len(key_df)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(key_df.iterrows()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        prop_name = row['Property']
        
        # Get values for each method
        methods = []
        values = []
        colors = []
        
        for col in value_cols:
            val = parse_mean(row[col])
            if not np.isnan(val):
                methods.append(col.replace('PRISM ', 'P.').replace('HBond Aromatic', 'HB-Aro')
                              .replace('Aromatic Bonus ', 'Aro.').replace('PoseBusters', 'PB'))
                values.append(val)
                colors.append(METHOD_COLORS.get(col, '#7D8491'))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(methods))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=8)
        ax.set_xlabel(prop_name, fontsize=10, fontweight='semibold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_width() + 0.02 * max(values), bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=7)
    
    # Hide empty subplots
    for idx in range(len(key_df), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Key Molecular Properties Comparison - AMPC Beta-Lactamase', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path('/Users/sanazkazeminia/Documents/analysis_suite/results/ampc_full_comparison/molecular_props/AMPC_beta_lactamase_properties_barplot')
    save_figure(fig, output_path, formats=['png', 'svg'], dpi=300)
    plt.close()
    
    print("✓ Properties comparison barplot regenerated!")


if __name__ == '__main__':
    regenerate_radar_overlay()
    regenerate_tsne_plot()
    regenerate_molecular_properties_plots()
    regenerate_properties_table_plot()
    
    print("\n" + "="*60)
    print("ALL PLOTS REGENERATED!")
    print("="*60)

