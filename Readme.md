# PRISM Analysis Suite

A molecular analysis pipeline for benchmarking structure-based drug design methods. Generates publication-quality figures comparing generated molecules against reference ligands.

## Installation
```bash
cd analysis_suite
pip install -e .
```

## Quick Start
```bash
python main.py --config config/default_config.yaml
```

Validate paths without running analyses:
```bash
python main.py --config config/default_config.yaml --validate-only
```

Run specific analyses only:
```bash
python main.py --config config/default_config.yaml --analyses posebusters radar
```

## Configuration

Create a YAML config file specifying your datasets and analyses:
```yaml
output_dir: ./results

datasets:
  - name: AMPC_beta_lactamase
    reference: /path/to/reference_ligands.sdf
    methods:
      PRISM: /path/to/prism/generated_molecules.sdf
      DiffSBDD: /path/to/diffsbdd/molecules.sdf
  
  - name: Carbonic_Anhydrase
    reference: /path/to/ca/reference_ligands.sdf
    methods:
      PRISM: /path/to/prism/ca/generated_molecules.sdf
      DiffSBDD: /path/to/diffsbdd/ca/molecules.sdf

analyses:
  - posebusters
  - rings
  - radar

plotting:
  figure_format: [png, svg]
  dpi: 300
```

## Available Analyses

| Analysis | Description | Output |
|----------|-------------|--------|
| `posebusters` | PoseBusters validity checks | Pass/fail bar charts, failure breakdown |
| `rings` | Ring and molecular property distributions | Violin plots per dataset |
| `radar` | Chemical property radar plots | Spider charts comparing property profiles |

## Output Structure
```
results/
├── posebusters/
│   ├── posebusters_summary.csv
│   ├── posebusters_grouped_comparison.png
│   ├── {dataset}_posebusters.png
│   └── posebusters_failure_breakdown.csv
├── molecular_properties/
│   ├── property_summary.csv
│   ├── {dataset}_ring_properties.png
│   └── {dataset}_molecular_properties.png
└── radar/
    ├── radar_summary.csv
    ├── {dataset}_radar_grid.png
    ├── {dataset}_radar_overlay.png
    └── {dataset}_{source}_radar.png
```

## Adding a New Analysis

Follow these steps to add a new analysis module:

### Step 1: Create the analysis file

Create `src/analysis/your_analysis.py`:
```python
"""Your analysis description."""

import pandas as pd
from pathlib import Path
from src.utils import load_molecules, set_pub_style, save_figure


def process_dataset(dataset, output_dir: Path, config):
    """Process a single dataset."""
    print(f"\nProcessing: {dataset.name}")
    
    # Load reference
    ref_mols = load_molecules(dataset.reference)
    
    # Load methods
    for method_name, sdf_path in dataset.methods.items():
        mols = load_molecules(sdf_path)
        # Your analysis logic here
    
    return results


def run_your_analysis(config) -> pd.DataFrame:
    """
    Run your analysis pipeline.
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        Summary DataFrame
    """
    print("\n" + "="*70)
    print("YOUR ANALYSIS NAME")
    print("="*70)
    
    output_dir = config.output_dir / 'your_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in config.datasets:
        process_dataset(dataset, output_dir, config)
    
    print("\nYour analysis complete!")
    return pd.DataFrame()
```

### Step 2: Update src/analysis/__init__.py
```python
from .posebusters_analysis import run_posebusters_analysis
from .ring_analysis import run_ring_analysis
from .radar_analysis import run_radar_analysis
from .your_analysis import run_your_analysis  # Add this
```

### Step 3: Update src/config.py

Find `valid_analyses` and add your analysis:
```python
valid_analyses = {'posebusters', 'rings', 'properties', 'radar', 'your_analysis'}
```

### Step 4: Update src/cli.py

Add the import:
```python
from src.analysis import (
    run_posebusters_analysis, 
    run_ring_analysis, 
    run_radar_analysis,
    run_your_analysis  # Add this
)
```

Add the execution block in `main()`:
```python
if 'your_analysis' in config.analyses:
    run_your_analysis(config)
```

### Step 5: Update config YAML
```yaml
analyses:
  - posebusters
  - rings
  - radar
  - your_analysis
```

### Step 6: Run it
```bash
python main.py --config config/default_config.yaml
```

## Project Structure
```
analysis_suite/
├── config/
│   └── default_config.yaml
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── posebusters_analysis.py
│   │   ├── ring_analysis.py
│   │   └── radar_analysis.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── plotting.py
│       └── statistics.py
├── main.py
├── pyproject.toml
└── README.md
```

## Utilities

Import commonly used utilities in your analysis modules:
```python
from src.utils import (
    load_molecules,      # Load RDKit mols from SDF
    set_pub_style,       # Set publication-ready plot style
    save_figure,         # Save figure in multiple formats
    get_method_color,    # Get consistent colors for methods
    cliffs_delta,        # Effect size calculation
    pvalue_to_asterisks, # Convert p-values to significance stars
    PASS_LABEL,          # "PB-Valid"
    FAIL_LABEL,          # "PB-Invalid"
)
```

## License

MIT