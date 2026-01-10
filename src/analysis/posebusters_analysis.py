"""PoseBusters validity analysis with grouped comparisons."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from posebusters import PoseBusters

from src.utils import (
    load_molecules, 
    set_pub_style, 
    save_figure,
    pvalue_to_asterisks,
    PASS_LABEL, 
    FAIL_LABEL
)


def analyze_molecules(mols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run PoseBusters validation on a list of molecules.
    
    Args:
        mols: List of RDKit molecule objects
        
    Returns:
        Tuple of (results_df, failure_counts_df)
    """
    if not mols:
        print("  Error: No molecules provided for analysis.")
        return None, None

    print(f"  Running PoseBusters on {len(mols)} molecules...")
    buster = PoseBusters(config="mol_fast")
    results_df = buster.bust(mols)

    check_columns = [col for col in results_df.columns if results_df[col].dtype == 'bool']

    if not check_columns:
        print("  Warning: No boolean check columns found in PoseBusters output.")
        return results_df, pd.DataFrame()

    results_df['passed_all_checks'] = results_df[check_columns].all(axis=1)
    
    failed_df = results_df[~results_df['passed_all_checks']]
    failure_counts = {}
    
    if not failed_df.empty:
        print(f"  Found {len(failed_df)} molecules that failed one or more checks.")
        for col in check_columns:
            num_failures = (~failed_df[col]).sum()
            if num_failures > 0:
                failure_counts[col] = num_failures
    else:
        print("  All molecules passed every check!")

    failure_counts_df = pd.DataFrame(
        list(failure_counts.items()), 
        columns=['Check', 'Failure Count']
    ).sort_values(by='Failure Count', ascending=False).reset_index(drop=True)

    return results_df, failure_counts_df


def process_dataset(dataset, output_dir: Path) -> dict:
    """
    Process a single dataset with reference and all methods.
    
    Args:
        dataset: Dataset object containing name, reference, and method paths
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping source names to their results DataFrames
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset.name}")
    print('='*60)
    
    dataset_results = {}
    
    # Process reference first
    print(f"\n  Reference: {dataset.reference}")
    ref_mols = load_molecules(dataset.reference)
    if ref_mols:
        results_df, failure_counts_df = analyze_molecules(ref_mols)
        if results_df is not None:
            csv_path = output_dir / f"{dataset.name}_Reference_PB_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")
            
            if failure_counts_df is not None and not failure_counts_df.empty:
                failure_path = output_dir / f"{dataset.name}_Reference_PB_failures.csv"
                failure_counts_df.to_csv(failure_path, index=False)
            
            dataset_results['Reference'] = results_df
    else:
        print("  WARNING: Could not load reference molecules")
    
    # Process each method
    for method_name, sdf_path in dataset.methods.items():
        print(f"\n  Method: {method_name}")
        print(f"  Path: {sdf_path}")
        
        mols = load_molecules(sdf_path)
        if not mols:
            print(f"  WARNING: Could not load molecules, skipping...")
            continue
        
        results_df, failure_counts_df = analyze_molecules(mols)
        
        if results_df is not None:
            csv_path = output_dir / f"{dataset.name}_{method_name}_PB_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")
            
            if failure_counts_df is not None and not failure_counts_df.empty:
                failure_path = output_dir / f"{dataset.name}_{method_name}_PB_failures.csv"
                failure_counts_df.to_csv(failure_path, index=False)
            
            dataset_results[method_name] = results_df
    
    return dataset_results


def compute_pass_rates(all_results: dict) -> pd.DataFrame:
    """
    Compute pass rates for all datasets and sources.
    
    Args:
        all_results: Nested dict {dataset_name: {source_name: results_df}}
        
    Returns:
        DataFrame with columns: Dataset, Source, Total, Passed, Failed, Pass_Rate
    """
    rows = []
    for dataset_name, sources in all_results.items():
        for source_name, df in sources.items():
            total = len(df)
            passed = df['passed_all_checks'].sum()
            rows.append({
                'Dataset': dataset_name,
                'Source': source_name,
                'Total': total,
                'Passed': passed,
                'Failed': total - passed,
                'Pass_Rate': (passed / total * 100) if total > 0 else 0
            })
    
    return pd.DataFrame(rows)


def compute_method_statistics(all_results: dict, method1: str, method2: str) -> dict:
    """
    Compute chi-square statistics comparing two methods across all datasets.
    
    Args:
        all_results: Nested dict {dataset_name: {source_name: results_df}}
        method1: First method name
        method2: Second method name
        
    Returns:
        Dict mapping dataset names to (p_value, asterisks) tuples
    """
    stats = {}
    for dataset_name, sources in all_results.items():
        if method1 in sources and method2 in sources:
            df1 = sources[method1]
            df2 = sources[method2]
            
            passed1 = df1['passed_all_checks'].sum()
            failed1 = (~df1['passed_all_checks']).sum()
            passed2 = df2['passed_all_checks'].sum()
            failed2 = (~df2['passed_all_checks']).sum()
            
            contingency = [
                [passed1, failed1],
                [passed2, failed2]
            ]
            _, p_val, _, _ = chi2_contingency(contingency, correction=True)
            stats[dataset_name] = (p_val, pvalue_to_asterisks(p_val))
    
    return stats


def plot_grouped_validity(
    summary_df: pd.DataFrame, 
    output_path: Path,
    figure_formats: list[str] = None,
    dpi: int = 300,
    all_results: dict = None
):
    """
    Create grouped bar chart with datasets on x-axis and sources side-by-side.
    Includes statistical tests between methods (not against reference).
    
    Args:
        summary_df: DataFrame with Dataset, Source, Pass_Rate columns
        output_path: Path for output figure (without extension)
        figure_formats: List of formats to save
        dpi: Resolution for raster formats
        all_results: Optional nested dict for computing statistics
    """
    set_pub_style()
    
    if figure_formats is None:
        figure_formats = ['png', 'svg']
    
    datasets = summary_df['Dataset'].unique()
    
    # Order sources: Reference first, then alphabetical for methods
    all_sources = summary_df['Source'].unique().tolist()
    sources = ['Reference'] if 'Reference' in all_sources else []
    method_sources = sorted([s for s in all_sources if s != 'Reference'])
    sources += method_sources
    
    n_datasets = len(datasets)
    n_sources = len(sources)
    
    # Wider figure for more space between groups
    fig_width = max(14, 3.0 * n_datasets)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    x = np.arange(n_datasets)
    total_width = 0.80
    bar_width = total_width / n_sources
    
    # Refined color palette - more visually appealing
    colors = {
        'Reference': '#37474F',  # Blue-grey (sophisticated dark)
        'DiffSBDD': '#5C6BC0',   # Indigo (distinct but harmonious)
        'PRISM': '#26A69A'       # Teal (complementary, fresh)
    }
    # Fallback colors for other methods
    fallback_colors = ['#7E57C2', '#EF5350', '#FFA726', '#66BB6A']
    for i, method in enumerate(method_sources):
        if method not in colors:
            colors[method] = fallback_colors[i % len(fallback_colors)]
    
    # Store bar positions for statistical annotations
    bar_positions = {source: {} for source in sources}
    
    for i, source in enumerate(sources):
        source_data = summary_df[summary_df['Source'] == source]
        
        pass_rates = []
        totals = []
        for dataset in datasets:
            row = source_data[source_data['Dataset'] == dataset]
            if not row.empty:
                pass_rates.append(row['Pass_Rate'].values[0])
                totals.append(row['Total'].values[0])
            else:
                pass_rates.append(0)
                totals.append(0)
        
        offset = (i - n_sources/2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, 
            pass_rates, 
            bar_width * 0.85,
            label=source,
            color=colors.get(source, '#9E9E9E'),
            edgecolor='white',
            linewidth=1.0,
            zorder=3
        )
        
        # Store positions and add labels
        for j, (bar, rate, total) in enumerate(zip(bars, pass_rates, totals)):
            bar_positions[source][datasets[j]] = (bar.get_x() + bar.get_width()/2, bar.get_height())
            if rate > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1.5,
                    f'{rate:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='semibold',
                    color='#263238'
                )
    
    # Add statistical comparisons between methods (not reference)
    if all_results and len(method_sources) >= 2:
        method1, method2 = method_sources[0], method_sources[1]
        method_stats = compute_method_statistics(all_results, method1, method2)
        
        # Draw significance brackets between methods
        max_height = summary_df['Pass_Rate'].max()
        bracket_height = max_height + 12
        
        for j, dataset in enumerate(datasets):
            if dataset in method_stats:
                p_val, asterisks = method_stats[dataset]
                if asterisks:  # Only show if significant
                    # Get positions of the two method bars
                    x1 = bar_positions[method1][dataset][0]
                    x2 = bar_positions[method2][dataset][0]
                    y1 = bar_positions[method1][dataset][1]
                    y2 = bar_positions[method2][dataset][1]
                    
                    # Draw bracket
                    bracket_y = max(y1, y2) + 8
                    ax.plot([x1, x1, x2, x2], 
                           [bracket_y - 2, bracket_y, bracket_y, bracket_y - 2],
                           color='#455A64', linewidth=1.2, zorder=4)
                    
                    # Add asterisks
                    ax.text((x1 + x2) / 2, bracket_y + 1, asterisks,
                           ha='center', va='bottom', fontsize=12, 
                           fontweight='bold', color='#D32F2F')
    
    # Styling
    ax.set_ylabel('PB-Valid (%)', fontsize=12, fontweight='semibold')
    ax.set_xlabel('')
    ax.set_xticks(x)
    
    # Cleaner x-tick labels
    clean_labels = [d.replace('_', '\n') for d in datasets]
    ax.set_xticklabels(clean_labels, rotation=0, ha='center', fontsize=10)
    
    ax.set_ylim(0, 130)  # More headroom for annotations
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    
    # Grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(title='Method', loc='upper right', frameon=True, 
                       fontsize=10, title_fontsize=11, framealpha=0.95)
    legend.get_frame().set_edgecolor('#B0BEC5')
    
    ax.set_title('PoseBusters Validity Comparison by Target', 
                 pad=20, fontsize=14, fontweight='bold', color='#263238')
    
    # Add significance legend
    if all_results and len(method_sources) >= 2:
        sig_text = "χ² test: * p<0.05, ** p<0.01, *** p<0.001"
        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
               fontsize=9, va='top', ha='left', style='italic',
               color='#546E7A')
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#B0BEC5')
    ax.spines['bottom'].set_color('#B0BEC5')
    
    plt.tight_layout()
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


def plot_dataset_comparison(
    sources_dict: dict,
    dataset_name: str,
    output_path: Path,
    figure_formats: list[str] = None,
    dpi: int = 300
):
    """
    Create a single dataset comparison plot with Reference and methods.
    Includes statistical comparison between methods (not against reference).
    
    Args:
        sources_dict: Dict mapping source names to results DataFrames
        dataset_name: Name of the dataset for title
        output_path: Path for output figure (without extension)
        figure_formats: List of formats to save
        dpi: Resolution for raster formats
    """
    set_pub_style()
    
    if figure_formats is None:
        figure_formats = ['png', 'svg']
    
    # Order: Reference first, then methods alphabetically
    source_names = ['Reference'] if 'Reference' in sources_dict else []
    method_names = sorted([s for s in sources_dict.keys() if s != 'Reference'])
    source_names += method_names
    
    n_sources = len(source_names)
    
    # Refined color palette
    colors = {
        'Reference': '#37474F',  # Blue-grey
        'DiffSBDD': '#5C6BC0',   # Indigo
        'PRISM': '#26A69A'       # Teal
    }
    fallback_colors = ['#7E57C2', '#EF5350', '#FFA726', '#66BB6A']
    for i, name in enumerate(method_names):
        if name not in colors:
            colors[name] = fallback_colors[i % len(fallback_colors)]
    
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_sources), 7))
    
    x = np.arange(n_sources)
    bar_width = 0.65
    
    pass_rates = []
    totals = []
    passed_counts = []
    failed_counts = []
    
    for source in source_names:
        df = sources_dict[source]
        total = len(df)
        passed = df['passed_all_checks'].sum()
        pass_rates.append(passed / total * 100 if total > 0 else 0)
        totals.append(total)
        passed_counts.append(passed)
        failed_counts.append(total - passed)
    
    # Create bars with source-specific colors
    bars = []
    for i, (source, rate) in enumerate(zip(source_names, pass_rates)):
        bar = ax.bar(
            i, 
            rate, 
            bar_width, 
            color=colors.get(source, '#9E9E9E'),
            edgecolor='white',
            linewidth=1.5,
            zorder=3
        )
        bars.append(bar[0])
        
        # Hatched portion for failures (lighter shade)
        ax.bar(
            i,
            100 - rate,
            bar_width,
            bottom=rate,
            color='#ECEFF1',
            hatch='///',
            edgecolor=colors.get(source, '#9E9E9E'),
            linewidth=0.5,
            alpha=0.7,
            zorder=2
        )
    
    # Add percentage labels inside bars
    for i, (rate, total) in enumerate(zip(pass_rates, totals)):
        # Pass rate inside the bar
        label_color = 'white' if rate > 25 else '#263238'
        ax.text(i, rate/2, f'{rate:.1f}%', ha='center', va='center', 
               fontweight='bold', fontsize=14, color=label_color, zorder=4)
    
    # Statistical comparison BETWEEN methods (not against reference)
    if len(method_names) >= 2:
        # Find indices of the two methods
        method_indices = [source_names.index(m) for m in method_names[:2]]
        idx1, idx2 = method_indices[0], method_indices[1]
        
        # Chi-square test between the two methods
        contingency = [
            [passed_counts[idx1], failed_counts[idx1]],
            [passed_counts[idx2], failed_counts[idx2]]
        ]
        _, p_val, _, _ = chi2_contingency(contingency, correction=True)
        asterisks = pvalue_to_asterisks(p_val)
        
        # Draw bracket between the two method bars
        y_max = max(pass_rates[idx1], pass_rates[idx2])
        bracket_y = y_max + 8
        
        ax.plot([idx1, idx1, idx2, idx2], 
               [bracket_y - 3, bracket_y, bracket_y, bracket_y - 3],
               color='#455A64', linewidth=1.5, zorder=5)
        
        # Add asterisks or ns
        sig_label = asterisks if asterisks else 'ns'
        sig_color = '#D32F2F' if asterisks else '#78909C'
        ax.text((idx1 + idx2) / 2, bracket_y + 2, sig_label,
               ha='center', va='bottom', fontsize=14, 
               fontweight='bold', color=sig_color, zorder=5)
        
        # Add p-value annotation
        p_text = f'p = {p_val:.2e}' if p_val < 0.001 else f'p = {p_val:.4f}'
        ax.text((idx1 + idx2) / 2, bracket_y + 10, p_text,
               ha='center', va='bottom', fontsize=9, 
               color='#546E7A', style='italic', zorder=5)
    
    # Styling
    ax.set_ylabel('Percentage', fontsize=12, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n(n={t:,})' for s, t in zip(source_names, totals)],
                      fontsize=11)
    ax.set_ylim(0, 130)  # More headroom
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    
    # Clean title
    clean_name = dataset_name.replace('_', ' ')
    ax.set_title(f'{clean_name}', pad=20, fontsize=14, fontweight='bold', color='#263238')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#78909C', label=PASS_LABEL),
        Patch(facecolor='#ECEFF1', hatch='///', edgecolor='#78909C', label=FAIL_LABEL)
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                      fontsize=10, framealpha=0.95)
    legend.get_frame().set_edgecolor('#B0BEC5')
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#B0BEC5')
    ax.spines['bottom'].set_color('#B0BEC5')
    
    plt.tight_layout()
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


def generate_failure_breakdown(
    all_results: dict,
    output_dir: Path
) -> pd.DataFrame:
    """
    Generate a detailed breakdown of which PoseBusters checks failed.
    
    Args:
        all_results: Nested dict {dataset_name: {source_name: results_df}}
        output_dir: Directory to save the breakdown
        
    Returns:
        DataFrame with failure counts per check per source
    """
    rows = []
    
    for dataset_name, sources in all_results.items():
        for source_name, df in sources.items():
            check_columns = [col for col in df.columns 
                          if df[col].dtype == 'bool' and col != 'passed_all_checks']
            
            for col in check_columns:
                failures = (~df[col]).sum()
                total = len(df)
                rows.append({
                    'Dataset': dataset_name,
                    'Source': source_name,
                    'Check': col,
                    'Failures': failures,
                    'Total': total,
                    'Failure_Rate': (failures / total * 100) if total > 0 else 0
                })
    
    breakdown_df = pd.DataFrame(rows)
    
    if not breakdown_df.empty:
        breakdown_path = output_dir / 'posebusters_failure_breakdown.csv'
        breakdown_df.to_csv(breakdown_path, index=False)
        print(f"Saved failure breakdown: {breakdown_path}")
    
    return breakdown_df


def run_posebusters_analysis(config) -> pd.DataFrame:
    """
    Run complete PoseBusters analysis pipeline.
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        Summary DataFrame with pass rates for all datasets/sources
    """
    print("\n" + "="*70)
    print("POSEBUSTERS VALIDITY ANALYSIS")
    print("="*70)
    
    output_dir = config.output_dir / 'posebusters'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all datasets
    all_results = {}
    for dataset in config.datasets:
        dataset_results = process_dataset(dataset, output_dir)
        if dataset_results:
            all_results[dataset.name] = dataset_results
    
    if not all_results:
        print("ERROR: No valid results to analyze!")
        return None
    
    # Compute summary statistics
    summary_df = compute_pass_rates(all_results)
    summary_path = output_dir / 'posebusters_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    
    # Generate failure breakdown
    print("\nGenerating failure breakdown...")
    generate_failure_breakdown(all_results, output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Main grouped comparison plot (all datasets) with statistical tests
    plot_grouped_validity(
        summary_df,
        output_dir / 'posebusters_grouped_comparison',
        figure_formats=config.plotting.figure_formats,
        dpi=config.plotting.dpi,
        all_results=all_results
    )
    
    # Per-dataset detailed plots
    for dataset_name, sources_dict in all_results.items():
        plot_dataset_comparison(
            sources_dict,
            dataset_name,
            output_dir / f'{dataset_name}_posebusters',
            figure_formats=config.plotting.figure_formats,
            dpi=config.plotting.dpi
        )
    
    print("\nPoseBusters analysis complete!")
    return summary_df