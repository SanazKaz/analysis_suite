"""
Vina Scoring Analysis

Score ligands against a protein receptor using AutoDock Vina (score_only mode).
Converts files using OpenBabel and runs Vina scoring.
"""

import subprocess
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import re


def check_dependencies() -> Tuple[bool, list]:
    """Check if required tools (obabel, vina) are available."""
    missing = []
    
    try:
        subprocess.run(['obabel', '-V'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append('obabel (OpenBabel)')
    
    try:
        subprocess.run(['vina', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append('vina (AutoDock Vina)')
    
    return len(missing) == 0, missing


def prepare_protein(pdb_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Clean protein PDB and convert to PDBQT format.
    
    Args:
        pdb_path: Path to input PDB file
        output_dir: Directory for output files
        
    Returns:
        Path to PDBQT file or None if failed
    """
    protein_name = pdb_path.stem
    clean_pdb = output_dir / f"{protein_name}_clean.pdb"
    protein_pdbqt = output_dir / f"{protein_name}.pdbqt"
    
    # Clean protein - keep only ATOM records
    print(f"  Cleaning protein: {pdb_path.name}")
    with open(pdb_path, 'r') as f_in, open(clean_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                f_out.write(line)
    
    # Convert to PDBQT
    print(f"  Converting to PDBQT...")
    cmd = [
        'obabel', '-ipdb', str(clean_pdb),
        '-opdbqt', '-O', str(protein_pdbqt),
        '-xr', '-xh', '--partialcharge', 'gasteiger'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if protein_pdbqt.exists():
        print(f"  ✓ Protein prepared: {protein_pdbqt.name}")
        return protein_pdbqt
    else:
        print(f"  ✗ Failed to prepare protein: {result.stderr}")
        return None


def prepare_ligands(sdf_path: Path, output_dir: Path) -> list:
    """
    Split SDF file and convert ligands to PDBQT format.
    
    Args:
        sdf_path: Path to SDF file with ligands
        output_dir: Directory for output files
        
    Returns:
        List of paths to PDBQT files
    """
    ligands_dir = output_dir / 'ligands'
    ligands_dir.mkdir(exist_ok=True)
    
    ligand_base = sdf_path.stem
    
    # Split SDF into individual files
    print(f"  Splitting SDF file...")
    cmd = [
        'obabel', str(sdf_path),
        '-O', str(ligands_dir / f"{ligand_base}_.sdf"),
        '-m'
    ]
    subprocess.run(cmd, capture_output=True)
    
    # Convert each SDF to PDBQT
    pdbqt_files = []
    sdf_files = sorted(ligands_dir.glob("*.sdf"))
    
    print(f"  Converting {len(sdf_files)} ligands to PDBQT...")
    for sdf_file in sdf_files:
        pdbqt_file = sdf_file.with_suffix('.pdbqt')
        cmd = [
            'obabel', str(sdf_file),
            '-O', str(pdbqt_file),
            '-xh', '--partialcharge', 'gasteiger'
        ]
        result = subprocess.run(cmd, capture_output=True)
        if pdbqt_file.exists():
            pdbqt_files.append(pdbqt_file)
    
    print(f"  ✓ Prepared {len(pdbqt_files)} ligands")
    return pdbqt_files


def get_ligand_center(pdbqt_path: Path) -> Tuple[float, float, float]:
    """Calculate the geometric center of a ligand from PDBQT file."""
    coords = []
    with open(pdbqt_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    
    if coords:
        coords = np.array(coords)
        return tuple(coords.mean(axis=0))
    return (0.0, 0.0, 0.0)


def run_vina_score(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    output_dir: Path,
    center: Tuple[float, float, float] = None,
    box_size: Tuple[float, float, float] = (25, 25, 25)
) -> Optional[float]:
    """
    Run Vina in score_only mode.
    
    Args:
        receptor_pdbqt: Path to receptor PDBQT
        ligand_pdbqt: Path to ligand PDBQT
        output_dir: Directory for output files
        center: Binding site center (x, y, z). If None, uses ligand center.
        box_size: Box dimensions (x, y, z)
        
    Returns:
        Vina score (kcal/mol) or None if failed
    """
    ligand_name = ligand_pdbqt.stem
    log_file = output_dir / f"{ligand_name}.log"
    
    # If no center specified, use ligand center
    if center is None:
        center = get_ligand_center(ligand_pdbqt)
    
    cmd = [
        'vina',
        '--receptor', str(receptor_pdbqt),
        '--ligand', str(ligand_pdbqt),
        '--score_only',
        '--center_x', str(center[0]),
        '--center_y', str(center[1]),
        '--center_z', str(center[2]),
        '--size_x', str(box_size[0]),
        '--size_y', str(box_size[1]),
        '--size_z', str(box_size[2]),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse score from output
    score = None
    for line in result.stdout.split('\n'):
        if 'Affinity:' in line:
            match = re.search(r'Affinity:\s+([-\d.]+)', line)
            if match:
                score = float(match.group(1))
                break
    
    # Save log
    with open(log_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    
    return score


def score_ligands(
    receptor_pdbqt: Path,
    ligand_pdbqts: list,
    output_dir: Path,
    method_name: str,
    center: Tuple[float, float, float] = None,
    box_size: Tuple[float, float, float] = (25, 25, 25)
) -> pd.DataFrame:
    """
    Score all ligands against a receptor.
    
    Returns:
        DataFrame with ligand names and scores
    """
    results = []
    total = len(ligand_pdbqts)
    
    for i, ligand_pdbqt in enumerate(ligand_pdbqts):
        ligand_name = ligand_pdbqt.stem
        
        # Use ligand center if no global center specified
        lig_center = center if center else get_ligand_center(ligand_pdbqt)
        
        score = run_vina_score(
            receptor_pdbqt, ligand_pdbqt, output_dir,
            center=lig_center, box_size=box_size
        )
        
        results.append({
            'ligand_idx': i,
            'ligand_name': ligand_name,
            'method': method_name,
            'vina_score': score if score is not None else np.nan
        })
        
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    Scored {i + 1}/{total} ligands")
    
    return pd.DataFrame(results)


def run_vina_analysis(config) -> pd.DataFrame:
    """
    Run Vina scoring analysis from config.
    
    Config should have:
    - vina_protein: path to protein PDB
    - vina_center: (x, y, z) binding site center (optional)
    - vina_box_size: (x, y, z) box dimensions (optional)
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        Combined DataFrame with all scores
    """
    print("\n" + "="*70)
    print("VINA SCORING ANALYSIS")
    print("="*70)
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"ERROR: Missing required tools: {', '.join(missing)}")
        print("Please install OpenBabel and AutoDock Vina.")
        return pd.DataFrame()
    
    # Get Vina-specific config
    protein_path = getattr(config, 'vina_protein', None)
    if protein_path is None:
        print("ERROR: No protein specified for Vina analysis (vina_protein in config)")
        return pd.DataFrame()
    
    protein_path = Path(protein_path)
    if not protein_path.exists():
        print(f"ERROR: Protein file not found: {protein_path}")
        return pd.DataFrame()
    
    center = getattr(config, 'vina_center', None)
    box_size = getattr(config, 'vina_box_size', (25, 25, 25))
    
    print(f"\nProtein: {protein_path}")
    if center:
        print(f"Binding site center: {center}")
    else:
        print("Binding site center: Using ligand positions")
    print(f"Box size: {box_size}")
    
    # Create output directory
    output_dir = config.output_dir / 'vina'
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = output_dir / 'prepared'
    prep_dir.mkdir(exist_ok=True)
    
    # Prepare protein
    print("\nPreparing protein...")
    receptor_pdbqt = prepare_protein(protein_path, prep_dir)
    if receptor_pdbqt is None:
        print("ERROR: Failed to prepare protein")
        return pd.DataFrame()
    
    all_results = []
    
    for dataset in config.datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset.name}")
        print('='*60)
        
        # Process reference
        print(f"\nScoring Reference ligands...")
        ref_prep_dir = prep_dir / f"{dataset.name}_Reference"
        ref_prep_dir.mkdir(exist_ok=True)
        
        ref_pdbqts = prepare_ligands(dataset.reference, ref_prep_dir)
        if ref_pdbqts:
            ref_scores = score_ligands(
                receptor_pdbqt, ref_pdbqts, output_dir,
                'Reference', center, box_size
            )
            ref_scores['dataset'] = dataset.name
            all_results.append(ref_scores)
        
        # Process each method
        for method_name, sdf_path in dataset.methods.items():
            print(f"\nScoring {method_name} ligands...")
            method_prep_dir = prep_dir / f"{dataset.name}_{method_name}"
            method_prep_dir.mkdir(exist_ok=True)
            
            method_pdbqts = prepare_ligands(sdf_path, method_prep_dir)
            if method_pdbqts:
                method_scores = score_ligands(
                    receptor_pdbqt, method_pdbqts, output_dir,
                    method_name, center, box_size
                )
                method_scores['dataset'] = dataset.name
                all_results.append(method_scores)
    
    if not all_results:
        print("ERROR: No results generated")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save results
    results_csv = output_dir / 'vina_scores.csv'
    combined_df.to_csv(results_csv, index=False)
    print(f"\n✓ Saved scores: {results_csv}")
    
    # Generate summary
    summary = create_vina_summary(combined_df, output_dir)
    
    print("\nVina scoring analysis complete!")
    return combined_df


def create_vina_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Create summary statistics for Vina scores."""
    from scipy.stats import mannwhitneyu
    
    summary_rows = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        methods = dataset_df['method'].unique()
        
        # Get reference data
        ref_scores = dataset_df[dataset_df['method'] == 'Reference']['vina_score'].dropna()
        
        for method in methods:
            method_scores = dataset_df[dataset_df['method'] == method]['vina_score'].dropna()
            
            if len(method_scores) == 0:
                continue
            
            row = {
                'dataset': dataset,
                'method': method,
                'n_scored': len(method_scores),
                'mean_score': method_scores.mean(),
                'std_score': method_scores.std(),
                'median_score': method_scores.median(),
                'min_score': method_scores.min(),
                'max_score': method_scores.max(),
            }
            
            # Statistical comparison to reference
            if method != 'Reference' and len(ref_scores) > 0:
                try:
                    _, p_val = mannwhitneyu(ref_scores, method_scores, alternative='two-sided')
                    row['p_value_vs_ref'] = p_val
                    if p_val < 0.0001:
                        row['significance'] = '****'
                    elif p_val < 0.001:
                        row['significance'] = '***'
                    elif p_val < 0.01:
                        row['significance'] = '**'
                    elif p_val < 0.05:
                        row['significance'] = '*'
                    else:
                        row['significance'] = 'ns'
                except:
                    row['p_value_vs_ref'] = None
                    row['significance'] = '-'
            
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'vina_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary: {summary_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("VINA SCORING SUMMARY")
    print("="*80)
    print(f"\n{'Method':<20} {'N':<6} {'Mean±Std':<18} {'Median':<10} {'Min/Max':<15} {'Sig':<5}")
    print("-"*80)
    
    for _, row in summary_df.iterrows():
        mean_std = f"{row['mean_score']:.2f} ± {row['std_score']:.2f}"
        min_max = f"{row['min_score']:.2f}/{row['max_score']:.2f}"
        sig = row.get('significance', '-')
        print(f"{row['method']:<20} {row['n_scored']:<6} {mean_std:<18} {row['median_score']:<10.2f} {min_max:<15} {sig:<5}")
    
    print("-"*80)
    print("Lower scores = better binding affinity (kcal/mol)")
    print("Significance: *p<0.05, **p<0.01, ***p<0.001, ****p<0.0001, ns=not significant")
    
    return summary_df

