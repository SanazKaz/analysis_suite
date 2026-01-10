"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Dataset:
    """A dataset with reference and methods to compare."""
    name: str
    reference: Path
    methods: dict[str, Path]
    
    def __post_init__(self):
        self.reference = Path(self.reference)
        self.methods = {k: Path(v) for k, v in self.methods.items()}


@dataclass 
class PlottingConfig:
    """Plotting preferences."""
    figure_formats: list[str] = field(default_factory=lambda: ['png', 'svg'])
    dpi: int = 300


@dataclass
class AnalysisConfig:
    """Complete analysis configuration."""
    output_dir: Path
    datasets: list[Dataset]
    analyses: list[str]
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    # Vina-specific config
    vina_protein: Path = None
    vina_center: tuple = None  # (x, y, z) binding site center
    vina_box_size: tuple = field(default_factory=lambda: (25, 25, 25))
    # Feature reward config
    feature_reward_pkl: Path = None
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> 'AnalysisConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            AnalysisConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        
        datasets = []
        for ds in raw.get('datasets', []):
            if 'reference' not in ds:
                raise ValueError(f"Dataset '{ds.get('name', 'unknown')}' missing 'reference' path")
            datasets.append(Dataset(
                name=ds['name'],
                reference=ds['reference'],
                methods=ds['methods']
            ))
        
        if not datasets:
            raise ValueError("Config must contain at least one dataset")
        
        plot_raw = raw.get('plotting', {})
        plotting = PlottingConfig(
            figure_formats=plot_raw.get('figure_format', ['png', 'svg']),
            dpi=plot_raw.get('dpi', 300)
        )
        
        analyses = raw.get('analyses', ['posebusters'])
        valid_analyses = {'posebusters', 'rings', 'radar', 'molecular_props', 'vina', 'feature_rewards', 'tanimoto', 'chemical_space'}
        for a in analyses:
            if a not in valid_analyses:
                raise ValueError(f"Unknown analysis type: {a}. Valid options: {valid_analyses}")
        
        # Vina-specific config
        vina_raw = raw.get('vina', {})
        vina_protein = vina_raw.get('protein')
        if vina_protein:
            vina_protein = Path(vina_protein)
        
        vina_center = vina_raw.get('center')
        if vina_center:
            vina_center = tuple(vina_center)
        
        vina_box_size = vina_raw.get('box_size', [25, 25, 25])
        vina_box_size = tuple(vina_box_size)
        
        # Feature reward config
        feature_raw = raw.get('feature_rewards', {})
        feature_reward_pkl = feature_raw.get('hotspot_pkl')
        if feature_reward_pkl:
            feature_reward_pkl = Path(feature_reward_pkl)
        
        return cls(
            output_dir=Path(raw.get('output_dir', './results')),
            datasets=datasets,
            analyses=analyses,
            plotting=plotting,
            vina_protein=vina_protein,
            vina_center=vina_center,
            vina_box_size=vina_box_size,
            feature_reward_pkl=feature_reward_pkl
        )
    
    def validate_paths(self) -> list[str]:
        """
        Check that all paths exist.
        
        Returns:
            List of error messages for missing paths (empty if all valid)
        """
        errors = []
        for dataset in self.datasets:
            if not dataset.reference.exists():
                errors.append(f"[{dataset.name}] reference: {dataset.reference}")
            for method_name, path in dataset.methods.items():
                if not path.exists():
                    errors.append(f"[{dataset.name}] {method_name}: {path}")
        return errors