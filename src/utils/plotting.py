"""Plotting utilities and style configuration."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - must be before pyplot import

import matplotlib.pyplot as plt
import seaborn as sns


# Styling constants
COLOR_PRISM = "steelblue"
COLOR_DIFFSBDD = "orange"
PASS_LABEL = "PB-Valid"
FAIL_LABEL = "PB-Invalid"


def set_pub_style():
    """Set a clean, publication-ready style for plots."""
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelweight": "semibold",
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def get_method_color(method_name: str, is_first: bool = True) -> str:
    """
    Assign color based on method name.
    
    Args:
        method_name: Name of the method
        is_first: Whether this is the first method being compared
        
    Returns:
        Color string for plotting
    """
    if 'prism' in method_name.lower():
        return COLOR_PRISM
    elif 'diff' in method_name.lower():
        return COLOR_DIFFSBDD
    else:
        return COLOR_PRISM if is_first else COLOR_DIFFSBDD


def save_figure(fig, output_path, formats=None, dpi=300):
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure object
        output_path: Base path for output (without extension)
        formats: List of formats to save (default: ['png', 'svg'])
        dpi: Resolution for raster formats
    """
    from pathlib import Path
    
    if formats is None:
        formats = ['png', 'svg']
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {save_path}")