"""
Feature Reward Analysis

Analyzes molecules against pharmacophore hotspots using aromatic and 
polar feature matching. Adapted from PRISM reward scoring functions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu

from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from src.utils import load_molecules
from src.utils.statistics import pvalue_to_asterisks


class AromaticFeatureScorer:
    """Score molecules based on aromatic hotspot matching."""
    
    def __init__(self, pkl_path: str, sigma: float = 0.8, cutoff: float = 2.5, n_targets: int = 3):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.global_centroid = np.array(data.get('global_centroid', [0, 0, 0]))
        self.sigma, self.cutoff = sigma, cutoff
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        
        # Get top N aromatic clusters
        aromatic_idx = [i for i, f in enumerate(data['cluster_features']) if f == 'Aromatic']
        sorted_idx = sorted(aromatic_idx, key=lambda i: data['cluster_counts'][i], reverse=True)[:n_targets]
        
        self.target_centers = data['cluster_centers'][sorted_idx]
        self.target_counts = [data['cluster_counts'][i] for i in sorted_idx]
        self.norm_factor = max(self.target_counts) if self.target_counts else 1.0
    
    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0
        try:
            feats = [f for f in self.fdef.GetFeaturesForMol(mol) if f.GetFamily() == 'Aromatic']
        except:
            return 0.0
        
        if not feats:
            return 0.0
        
        cost = np.zeros((len(feats), len(self.target_centers)))
        for r, feat in enumerate(feats):
            pos = np.array([feat.GetPos().x, feat.GetPos().y, feat.GetPos().z]) - self.global_centroid
            for c in range(len(self.target_centers)):
                dist = np.linalg.norm(pos - self.target_centers[c])
                if dist <= self.cutoff:
                    cost[r, c] = -(np.exp(-0.5 * (dist / self.sigma) ** 2) * self.target_counts[c])
        
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(min(1.0, -cost[row_ind, col_ind].sum() / self.norm_factor))


class FeatureDensityScorer:
    """Score molecules based on polar interaction hotspots."""
    
    WEIGHTS = {'Acceptor': 0.45, 'Donor': 0.45, 'NegIonizable': 0.10}
    
    def __init__(self, pkl_path: str, sigma: float = 0.8, cutoff: float = 3.2):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.global_centroid = np.array(data.get('global_centroid', [0, 0, 0]))
        self.target_profile = data.get('target_profile', {})
        self.sigma, self.cutoff = sigma, cutoff
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        
        # Group clusters by type
        self.clusters = {}
        for i, feat in enumerate(data['cluster_features']):
            if feat in self.WEIGHTS:
                self.clusters.setdefault(feat, []).append((data['cluster_centers'][i], data['cluster_counts'][i]))
        for ft in self.clusters:
            self.clusters[ft] = sorted(self.clusters[ft], key=lambda x: x[1], reverse=True)
    
    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0
        try:
            raw_feats = self.fdef.GetFeaturesForMol(mol)
        except:
            return 0.0
        
        score = 0.0
        for feat_type, weight in self.WEIGHTS.items():
            mol_feats = [f for f in raw_feats if f.GetFamily() == feat_type]
            if not mol_feats or feat_type not in self.clusters:
                continue
            
            ideal = self.target_profile.get(feat_type, (len(self.clusters[feat_type]), 0))[0]
            targets = self.clusters[feat_type][:ideal]
            if not targets:
                continue
            
            centers, counts = zip(*targets)
            norm = sum(counts)
            
            cost = np.zeros((len(mol_feats), len(targets)))
            for r, feat in enumerate(mol_feats):
                pos = np.array([feat.GetPos().x, feat.GetPos().y, feat.GetPos().z]) - self.global_centroid
                for c, center in enumerate(centers):
                    dist = np.linalg.norm(pos - center)
                    if dist <= self.cutoff:
                        cost[r, c] = -(np.exp(-0.5 * (dist / self.sigma) ** 2) * counts[c])
            
            row_ind, col_ind = linear_sum_assignment(cost)
            score += weight * min(1.0, -cost[row_ind, col_ind].sum() / norm)
        
        return float(np.clip(score, 0.0, 1.0))


def run_feature_reward_analysis(config) -> pd.DataFrame:
    """Run feature reward analysis from config."""
    print("\n" + "="*70)
    print("FEATURE REWARD ANALYSIS")
    print("="*70)
    
    pkl_path = getattr(config, 'feature_reward_pkl', None)
    if not pkl_path or not Path(pkl_path).exists():
        print(f"ERROR: Hotspot pickle not found: {pkl_path}")
        return pd.DataFrame()
    
    print(f"Hotspot data: {pkl_path}")
    
    # Initialize scorers
    try:
        aromatic = AromaticFeatureScorer(str(pkl_path))
        print(f"✓ Aromatic scorer (norm: {aromatic.norm_factor:.1f})")
    except Exception as e:
        aromatic = None
    
    try:
        density = FeatureDensityScorer(str(pkl_path))
        print(f"✓ Density scorer")
    except Exception as e:
        density = None
    
    if not aromatic and not density:
        print("ERROR: No scorers initialized")
        return pd.DataFrame()
    
    output_dir = config.output_dir / 'feature_rewards'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_scores = []
    
    for dataset in config.datasets:
        print(f"\n{'='*60}\nProcessing: {dataset.name}\n{'='*60}")
        
        # Score reference
        print(f"\nScoring Reference...")
        ref_mols = load_molecules(dataset.reference)
        if ref_mols:
            all_scores.append(_score_set(ref_mols, 'Reference', dataset.name, aromatic, density))
        
        # Score methods
        for name, path in dataset.methods.items():
            print(f"Scoring {name}...")
            mols = load_molecules(path)
            if mols:
                all_scores.append(_score_set(mols, name, dataset.name, aromatic, density))
    
    if not all_scores:
        return pd.DataFrame()
    
    # Save and print results
    return _save_results(all_scores, output_dir)


def _score_set(mols: List[Mol], method: str, dataset: str, aromatic, density) -> dict:
    """Score a set of molecules."""
    result = {'method': method, 'dataset': dataset, 'n': len(mols)}
    
    if aromatic:
        scores = np.array([aromatic.score_mol(m) for m in mols])
        result.update({'aro_scores': scores, 'aro_mean': scores.mean(), 
                       'aro_std': scores.std(), 'aro_max': scores.max()})
    
    if density:
        scores = np.array([density.score_mol(m) for m in mols])
        result.update({'den_scores': scores, 'den_mean': scores.mean(),
                       'den_std': scores.std(), 'den_max': scores.max()})
    
    print(f"  {method}: {len(mols)} molecules scored")
    return result


def _save_results(results: List[dict], output_dir: Path) -> pd.DataFrame:
    """Save results to CSV and print summary."""
    # Summary
    summary = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith('_scores')} for r in results])
    summary.to_csv(output_dir / 'feature_reward_summary.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'feature_reward_summary.csv'}")
    
    # Detailed scores
    rows = []
    for r in results:
        for i in range(r['n']):
            row = {'dataset': r['dataset'], 'method': r['method'], 'mol_idx': i}
            if 'aro_scores' in r:
                row['aromatic_score'] = r['aro_scores'][i]
            if 'den_scores' in r:
                row['density_score'] = r['den_scores'][i]
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / 'feature_reward_scores.csv', index=False)
    print(f"✓ Saved: {output_dir / 'feature_reward_scores.csv'}")
    
    # Statistical comparison
    _save_comparison(results, output_dir)
    
    # Print table
    print("\n" + "="*80)
    print("FEATURE REWARD SUMMARY")
    print("="*80)
    has_aro = 'aro_mean' in results[0]
    has_den = 'den_mean' in results[0]
    
    header = f"{'Method':<20} {'N':<6}"
    if has_aro: header += f" {'Aromatic (Mean±Std)':<22} {'Max':<8}"
    if has_den: header += f" {'Density (Mean±Std)':<22} {'Max':<8}"
    print(header + "\n" + "-"*80)
    
    for r in results:
        line = f"{r['method']:<20} {r['n']:<6}"
        if has_aro: line += f" {r['aro_mean']:.3f} ± {r['aro_std']:.3f}          {r['aro_max']:<8.3f}"
        if has_den: line += f" {r['den_mean']:.3f} ± {r['den_std']:.3f}          {r['den_max']:<8.3f}"
        print(line)
    
    print("-"*80 + "\nScores: 0.0 to 1.0 (higher = better)")
    print("\nFeature reward analysis complete!")
    return summary


def _save_comparison(results: List[dict], output_dir: Path):
    """Save statistical comparisons vs reference."""
    by_dataset = {}
    for r in results:
        by_dataset.setdefault(r['dataset'], []).append(r)
    
    rows = []
    for dataset, methods in by_dataset.items():
        ref = next((m for m in methods if m['method'] == 'Reference'), None)
        if not ref:
            continue
        for m in methods:
            if m['method'] == 'Reference':
                continue
            row = {'dataset': dataset, 'method': m['method']}
            for key, name in [('aro_scores', 'aromatic'), ('den_scores', 'density')]:
                if key in ref and key in m:
                    try:
                        _, p = mannwhitneyu(ref[key], m[key], alternative='two-sided')
                        row[f'{name}_p'] = p
                        row[f'{name}_sig'] = pvalue_to_asterisks(p)
                    except:
                        pass
            rows.append(row)
    
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / 'feature_reward_comparison.csv', index=False)
        print(f"✓ Saved: {output_dir / 'feature_reward_comparison.csv'}")
