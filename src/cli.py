"""Command-line interface for the analysis pipeline."""

import argparse
import sys
from pathlib import Path

from src.config import AnalysisConfig
from src.analysis import (
    run_posebusters_analysis, 
    run_ring_analysis, 
    run_radar_analysis, 
    run_molecular_properties_analysis, 
    run_vina_analysis,
    run_feature_reward_analysis,
    run_tanimoto_analysis,
    run_chemical_space_analysis
)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Molecular analysis pipeline for PRISM benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config/my_analysis.yaml
  python main.py --config config/my_analysis.yaml --analyses posebusters rings
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--analyses', '-a',
        nargs='+',
        choices=['posebusters', 'rings', 'radar', 'molecular_props', 'vina', 'feature_rewards', 'tanimoto', 'chemical_space'],
        help='Override which analyses to run (default: use config file)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate config and paths, do not run analyses'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    try:
        config = AnalysisConfig.from_yaml(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Override analyses if specified
    if args.analyses:
        config.analyses = args.analyses
    
    # Validate paths
    print("\nValidating paths...")
    errors = config.validate_paths()
    if errors:
        print("ERROR: The following paths do not exist:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    print("All paths valid!")
    
    if args.validate_only:
        print("\nValidation complete. Exiting (--validate-only flag set).")
        sys.exit(0)
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {config.output_dir}")
    
    # Run requested analyses
    print(f"\nAnalyses to run: {config.analyses}")
    
    if 'posebusters' in config.analyses:
        run_posebusters_analysis(config)
    
    if 'rings' in config.analyses:
        run_ring_analysis(config)
    
    if 'radar' in config.analyses:
        run_radar_analysis(config)
    
    if 'molecular_props' in config.analyses:
        run_molecular_properties_analysis(config)
    
    if 'vina' in config.analyses:
        run_vina_analysis(config)
    
    if 'feature_rewards' in config.analyses:
        run_feature_reward_analysis(config)
    
    if 'tanimoto' in config.analyses:
        run_tanimoto_analysis(config)
    
    if 'chemical_space' in config.analyses:
        run_chemical_space_analysis(config)
    
    if 'properties' in config.analyses:
        print("\nMolecular properties included in ring analysis")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()