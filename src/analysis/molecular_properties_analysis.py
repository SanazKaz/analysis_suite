"""
Molecular Properties Analysis

Analyzes molecular properties from SDF files with nitrogen valence corrections.
Calculates comprehensive drug-likeness metrics and structural properties.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
from rdkit import DataStructs
from typing import List, Tuple
from scipy.stats import mannwhitneyu

from src.utils import load_molecules


def calculate_sa_score(mol):
    """
    Calculate synthetic accessibility score.
    
    Tries to import sascorer, falls back to a simple heuristic if unavailable.
    """
    try:
        from sascorer import calculateScore
        return calculateScore(mol)
    except ImportError:
        # Fallback: use a simple heuristic based on complexity
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        score = 2.0 + (n_rings * 0.5) + (n_stereo * 0.3) + (n_bridgehead * 0.5)
        return min(score, 10.0)


class MoleculeNitrogenFixer:
    """Fix common nitrogen valence issues in molecules."""
    
    def add_nitrogen_charges(self, mol: Chem.Mol) -> Chem.Mol:
        """Add formal charges to quaternary nitrogens."""
        if mol is None:
            return None
        
        mol_copy = Chem.RWMol(mol)
        
        for atom in mol_copy.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                valence = atom.GetTotalValence()
                if valence == 4 and atom.GetFormalCharge() == 0:
                    atom.SetFormalCharge(1)
        
        try:
            Chem.SanitizeMol(mol_copy)
            return mol_copy.GetMol()
        except:
            return None
    
    def strip_radicals(self, mol: Chem.Mol):
        """Remove radical electrons from atoms."""
        for atom in mol.GetAtoms():
            atom.SetNumRadicalElectrons(0)
    
    def fix_kekulisation(self, mol: Chem.Mol) -> Chem.Mol:
        """Attempt to fix kekulization issues."""
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                            Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return mol
        except:
            return None


class MolecularPropertiesCalculator:
    """Calculate comprehensive molecular properties."""
    
    def __init__(self):
        self.nitrogen_fixer = MoleculeNitrogenFixer()
    
    def sanitize_molecule(self, mol: Chem.Mol) -> Tuple[Chem.Mol, str]:
        """
        Attempt to sanitize molecule with nitrogen corrections.
        
        Returns:
            Tuple of (sanitized_mol, status) where status is 'valid', 'fixed', or 'unfixable'
        """
        if mol is None:
            return None, 'unfixable'
        
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            return mol_copy, 'valid'
        except Exception:
            pass
        
        try:
            fixed_mol = self.nitrogen_fixer.add_nitrogen_charges(mol)
            if fixed_mol is not None:
                return fixed_mol, 'fixed'
        except Exception:
            pass
        
        try:
            mol_copy = Chem.Mol(mol)
            self.nitrogen_fixer.strip_radicals(mol_copy)
            fixed_mol = self.nitrogen_fixer.fix_kekulisation(mol_copy)
            return fixed_mol, 'fixed'
        except Exception:
            pass
        
        return None, 'unfixable'
    
    def load_and_sanitize_molecules(self, path: Path) -> List[Chem.Mol]:
        """
        Load molecules from SDF file or directory and sanitize them.
        """
        RDLogger.DisableLog('rdApp.*')
        
        molecules = load_molecules(path)
        
        sanitized_mols = []
        for mol in molecules:
            sanitized_mol, _ = self.sanitize_molecule(mol)
            if sanitized_mol is not None:
                sanitized_mols.append(sanitized_mol)
        
        RDLogger.EnableLog('rdApp.*')
        return sanitized_mols
    
    def count_cyclopropanes(self, mol: Chem.Mol) -> int:
        """Count number of cyclopropane rings."""
        try:
            ring_info = mol.GetRingInfo()
            return sum(1 for ring in ring_info.AtomRings() if len(ring) == 3)
        except:
            return 0
    
    def count_chiral_centers(self, mol: Chem.Mol) -> int:
        """Count number of chiral centers."""
        try:
            return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        except:
            return 0
    
    def count_heteroatoms(self, mol: Chem.Mol) -> int:
        """Count number of heteroatoms (non-C, non-H)."""
        try:
            return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        except:
            return 0
    
    def has_macrocycle(self, mol: Chem.Mol) -> int:
        """Check if molecule contains a macrocycle (ring > 12 atoms)."""
        try:
            ring_info = mol.GetRingInfo()
            return int(any(len(ring) > 12 for ring in ring_info.AtomRings()))
        except:
            return 0
    
    def calculate_properties(self, mol: Chem.Mol) -> dict:
        """Calculate comprehensive molecular properties."""
        if mol is None:
            return None
        
        try:
            sa_raw = calculate_sa_score(mol)
            props = {
                # Drug-likeness metrics
                'qed': QED.qed(mol),
                'sa': round((10 - sa_raw) / 9, 2),
                'logp': Crippen.MolLogP(mol),
                'lipinski_violations': self.calculate_lipinski_violations(mol),
                
                # Basic properties
                'mw': Descriptors.ExactMolWt(mol),
                'n_heavy_atoms': mol.GetNumHeavyAtoms(),
                
                # Hydrogen bonding
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                
                # Flexibility
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                
                # Ring properties
                'n_rings': rdMolDescriptors.CalcNumRings(mol),
                'n_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'n_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                'n_saturated_rings': rdMolDescriptors.CalcNumSaturatedRings(mol),
                'n_cyclopropanes': self.count_cyclopropanes(mol),
                'has_macrocycle': self.has_macrocycle(mol),
                
                # Complexity metrics
                'frac_sp3': Descriptors.FractionCSP3(mol),
                'n_chiral_centers': self.count_chiral_centers(mol),
                'n_stereocenters': rdMolDescriptors.CalcNumAtomStereoCenters(mol),
                'n_heteroatoms': self.count_heteroatoms(mol),
            }
            
            props['n_non_aromatic_rings'] = props['n_rings'] - props['n_aromatic_rings']
            
            return props
            
        except Exception:
            return None
    
    def calculate_lipinski_violations(self, mol: Chem.Mol) -> int:
        """Calculate number of Lipinski rule violations."""
        try:
            rule_1 = Descriptors.ExactMolWt(mol) < 500
            rule_2 = Lipinski.NumHDonors(mol) <= 5
            rule_3 = Lipinski.NumHAcceptors(mol) <= 10
            logp = Crippen.MolLogP(mol)
            rule_4 = (-2 <= logp <= 5)
            rule_5 = rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
            
            violations = sum([not rule for rule in [rule_1, rule_2, rule_3, rule_4, rule_5]])
            return violations
        except:
            return 5
    
    def calculate_diversity(self, molecules: List[Chem.Mol], max_samples: int = 500) -> float:
        """Calculate average pairwise Tanimoto diversity (sampled for speed)."""
        if len(molecules) < 2:
            return 0.0
        
        try:
            # Sample molecules for speed if we have many
            import random
            if len(molecules) > max_samples:
                sampled_mols = random.sample(molecules, max_samples)
            else:
                sampled_mols = molecules
            
            # Pre-compute fingerprints once
            fingerprints = [Chem.RDKFingerprint(mol) for mol in sampled_mols]
            
            div = 0
            total = 0
            for i in range(len(fingerprints)):
                for j in range(i + 1, len(fingerprints)):
                    similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    div += 1 - similarity
                    total += 1
            return div / total if total > 0 else 0.0
        except:
            return 0.0


def run_molecular_properties_analysis(config) -> pd.DataFrame:
    """
    Run molecular properties analysis from config.
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        Combined comparison DataFrame
    """
    print("\n" + "="*70)
    print("MOLECULAR PROPERTIES ANALYSIS")
    print("="*70)
    
    output_dir = config.output_dir / 'molecular_props'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_comparisons = []
    
    for dataset in config.datasets:
        comparison_df = process_dataset_properties(dataset, output_dir)
        if comparison_df is not None:
            all_comparisons.append(comparison_df)
    
    print("\nMolecular properties analysis complete!")
    
    if all_comparisons:
        return pd.concat(all_comparisons, ignore_index=True)
    return pd.DataFrame()


def process_dataset_properties(dataset, output_dir: Path) -> pd.DataFrame:
    """Process a single dataset for molecular properties analysis."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {dataset.name}")
    print('='*60)
    
    calculator = MolecularPropertiesCalculator()
    
    # Load reference molecules
    print(f"\nLoading Reference molecules...")
    ref_mols = calculator.load_and_sanitize_molecules(dataset.reference)
    print(f"  Loaded and sanitized {len(ref_mols)} molecules")
    
    if not ref_mols:
        print("  WARNING: Could not load reference molecules")
        return None
    
    # Calculate reference properties
    print(f"  Calculating reference properties...")
    ref_results = []
    ref_failed = 0
    for i, mol in enumerate(ref_mols):
        props = calculator.calculate_properties(mol)
        if props:
            props['mol_idx'] = i
            props['method'] = 'Reference'
            try:
                props['smiles'] = Chem.MolToSmiles(mol)
            except:
                props['smiles'] = "N/A"
            ref_results.append(props)
        else:
            ref_failed += 1
    
    ref_df = pd.DataFrame(ref_results)
    print(f"  Reference: {len(ref_results)} success, {ref_failed} failed")
    print(f"  Calculating reference diversity...")
    ref_diversity = calculator.calculate_diversity(ref_mols)
    print(f"  Reference diversity: {ref_diversity:.3f}")
    
    # Process each method
    method_dfs = {'Reference': (ref_df, ref_diversity, ref_mols)}
    
    for method_name, sdf_path in dataset.methods.items():
        print(f"\nLoading {method_name} molecules...")
        mols = calculator.load_and_sanitize_molecules(sdf_path)
        print(f"  Loaded and sanitized {len(mols)} molecules")
        
        if not mols:
            print(f"  WARNING: Could not load molecules, skipping")
            continue
        
        results = []
        failed = 0
        total_mols = len(mols)
        for i, mol in enumerate(mols):
            props = calculator.calculate_properties(mol)
            if props:
                props['mol_idx'] = i
                props['method'] = method_name
                try:
                    props['smiles'] = Chem.MolToSmiles(mol)
                except:
                    props['smiles'] = "N/A"
                results.append(props)
            else:
                failed += 1
            
            # Print progress every 1000 molecules
            if (i + 1) % 1000 == 0:
                print(f"    Processed {i+1}/{total_mols} molecules...")
        
        method_df = pd.DataFrame(results)
        print(f"  Calculating diversity (sampled)...")
        diversity = calculator.calculate_diversity(mols)
        method_dfs[method_name] = (method_df, diversity, mols)
        
        print(f"  {method_name}: {len(results)} success, {failed} failed, diversity={diversity:.3f}")
    
    if len(method_dfs) < 2:
        print("  WARNING: Not enough data for comparison")
        return None
    
    # Save combined molecule data
    all_dfs = [df for df, _, _ in method_dfs.values()]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    csv_path = output_dir / f'{dataset.name}_molecular_properties.csv'
    combined_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    
    # Create comparison table
    comparison_df = create_comparison_table(dataset.name, method_dfs, output_dir)
    
    return comparison_df


def create_comparison_table(dataset_name: str, method_dfs: dict, output_dir: Path) -> pd.DataFrame:
    """Create formatted comparison table between methods."""
    
    key_properties = [
        ('QED', 'qed', '{:.3f}'),
        ('SA Score', 'sa', '{:.2f}'),
        ('LogP', 'logp', '{:.2f}'),
        ('Molecular Weight', 'mw', '{:.1f}'),
        ('H-Bond Donors', 'hbd', '{:.1f}'),
        ('H-Bond Acceptors', 'hba', '{:.1f}'),
        ('TPSA', 'tpsa', '{:.1f}'),
        ('Rotatable Bonds', 'rotatable_bonds', '{:.1f}'),
        ('Lipinski Violations', 'lipinski_violations', '{:.2f}'),
        ('Fraction sp3', 'frac_sp3', '{:.3f}'),
        ('N Rings', 'n_rings', '{:.1f}'),
        ('N Aromatic Rings', 'n_aromatic_rings', '{:.2f}'),
        ('N Cyclopropanes', 'n_cyclopropanes', '{:.2f}'),
        ('N Chiral Centers', 'n_chiral_centers', '{:.1f}'),
        ('N Heteroatoms', 'n_heteroatoms', '{:.1f}'),
        ('Diversity', 'diversity', '{:.3f}'),
    ]
    
    methods = list(method_dfs.keys())
    ref_df, ref_diversity, _ = method_dfs.get('Reference', (None, 0, []))
    
    comparison_rows = []
    
    for display_name, col_name, fmt in key_properties:
        row = {'Dataset': dataset_name, 'Property': display_name}
        
        for method_name in methods:
            df, diversity, _ = method_dfs[method_name]
            
            if col_name == 'diversity':
                val = diversity
                std = None
            elif col_name in df.columns:
                val = df[col_name].mean()
                std = df[col_name].std()
            else:
                continue
            
            val_str = fmt.format(val) if val is not None else 'N/A'
            if std is not None:
                std_str = fmt.format(std)
                row[method_name] = f"{val_str} ± {std_str}"
            else:
                row[method_name] = val_str
            
            # Add p-value comparison to reference
            if method_name != 'Reference' and ref_df is not None and col_name != 'diversity':
                if col_name in df.columns and col_name in ref_df.columns:
                    try:
                        _, p_val = mannwhitneyu(ref_df[col_name], df[col_name], alternative='two-sided')
                        if p_val < 0.0001:
                            sig = '****'
                        elif p_val < 0.001:
                            sig = '***'
                        elif p_val < 0.01:
                            sig = '**'
                        elif p_val < 0.05:
                            sig = '*'
                        else:
                            sig = 'ns'
                        row[f'{method_name}_p'] = f"{p_val:.4f}"
                        row[f'{method_name}_sig'] = sig
                    except:
                        row[f'{method_name}_p'] = '-'
                        row[f'{method_name}_sig'] = '-'
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / f'{dataset_name}_properties_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  Saved: {comparison_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"MOLECULAR PROPERTIES COMPARISON - {dataset_name}")
    print("="*80)
    
    # Determine column widths
    print(f"\n{'Property':<22}", end='')
    for method in methods:
        print(f"{method:<22}", end='')
    print()
    print("-"*80)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Property']:<22}", end='')
        for method in methods:
            if method in row:
                print(f"{row[method]:<22}", end='')
        print()
    
    print("-"*80)
    
    # Print molecule counts
    counts = [f"{m}={len(df)}" for m, (df, _, _) in method_dfs.items()]
    print(f"\nN molecules: {', '.join(counts)}")
    print("Significance: *p<0.05, **p<0.01, ***p<0.001, ****p<0.0001, ns=not significant")
    
    return comparison_df

