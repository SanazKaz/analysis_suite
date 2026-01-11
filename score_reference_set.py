#!/usr/bin/env python
"""Score the reference set with FeatureDensityReward to see mean and max scores."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '/Users/sanazkazeminia/Documents/analysis_suite')

import numpy as np
from rdkit import Chem, RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

from src.analysis.rewards.feature_density import FeatureDensityReward
from src.utils.io import load_molecules

# Paths
PKL_PATH = "/Users/sanazkazeminia/Documents/analysis_suite/data/temp_runs/ampc_hotspot_global_centered_eps0.5_min10_aromatic5.pkl"
REFERENCE_PATH = "/Users/sanazkazeminia/Documents/analysis_suite/data/data/AMPC_beta_lactamase/02_preprocessed/sdf_files"

def main():
    print("="*70)
    print("SCORING REFERENCE SET WITH FEATURE DENSITY REWARD")
    print("="*70)
    
    # Load the reward scorer
    print(f"\nLoading reward scorer from: {PKL_PATH}")
    scorer = FeatureDensityReward(
        pkl_path=PKL_PATH,
        sigma=1.0,
        cutoff=5.0,
        verbose=False  # Set to True to see per-molecule details
    )
    
    # Load reference molecules
    print(f"\nLoading reference molecules from: {REFERENCE_PATH}")
    ref_mols = load_molecules(REFERENCE_PATH)
    print(f"Loaded {len(ref_mols)} reference molecules")
    
    # Score all molecules
    print("\nScoring molecules...")
    scores = []
    failed = 0
    
    for i, mol in enumerate(ref_mols):
        if mol is None:
            failed += 1
            continue
        
        try:
            score = scorer.score_mol(mol)
            scores.append(score)
            
            if (i + 1) % 50 == 0:
                print(f"  Scored {i+1}/{len(ref_mols)} molecules...")
        except Exception as e:
            failed += 1
    
    scores = np.array(scores)
    
    # Print statistics
    print("\n" + "="*70)
    print("REFERENCE SET SCORING RESULTS")
    print("="*70)
    
    print(f"\nTotal molecules: {len(ref_mols)}")
    print(f"Successfully scored: {len(scores)}")
    print(f"Failed: {failed}")
    
    print(f"\n{'Statistic':<20} {'Value':>12}")
    print("-"*35)
    print(f"{'Mean score:':<20} {np.mean(scores):>12.4f}")
    print(f"{'Std dev:':<20} {np.std(scores):>12.4f}")
    print(f"{'Median:':<20} {np.median(scores):>12.4f}")
    print(f"{'Min score:':<20} {np.min(scores):>12.4f}")
    print(f"{'Max score:':<20} {np.max(scores):>12.4f}")
    print(f"{'25th percentile:':<20} {np.percentile(scores, 25):>12.4f}")
    print(f"{'75th percentile:':<20} {np.percentile(scores, 75):>12.4f}")
    print(f"{'90th percentile:':<20} {np.percentile(scores, 90):>12.4f}")
    
    # Score distribution
    print("\nScore distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(scores, bins=bins)
    for i in range(len(bins)-1):
        pct = hist[i] / len(scores) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:>4} ({pct:>5.1f}%) {bar}")
    
    # Top 10 scores
    print("\nTop 10 scores:")
    top_indices = np.argsort(scores)[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:>2}. Score: {scores[idx]:.4f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

