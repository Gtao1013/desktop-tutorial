"""
Point-E Starship Point Cloud Generation Package

A comprehensive solution for generating Starship point clouds using OpenAI's Point-E model
based on aerodynamic conditions.
"""

__version__ = "1.0.0"
__author__ = "Point-E Starship Team"
__email__ = "support@example.com"

# Import main classes for easy access
try:
    from .setup_point_e import PointESetup, setup_point_e
    from .aero_condition_adapter import AeroConditionAdapter, create_sample_aero_conditions
    from .data_processor import PointCloudProcessor, PointCloudDataset, create_dataloader
    from .point_e_generator import PointEGenerator
    from .point_e_trainer import PointETrainer, train_point_e_model
    from .visualization import PointCloudVisualizer, AerodynamicVisualizer, create_visualization_gallery
    from .evaluation import PointCloudEvaluator, AerodynamicConsistencyEvaluator, comprehensive_evaluation
    
    __all__ = [
        'PointESetup',
        'setup_point_e',
        'AeroConditionAdapter', 
        'create_sample_aero_conditions',
        'PointCloudProcessor',
        'PointCloudDataset',
        'create_dataloader',
        'PointEGenerator',
        'PointETrainer',
        'train_point_e_model',
        'PointCloudVisualizer',
        'AerodynamicVisualizer', 
        'create_visualization_gallery',
        'PointCloudEvaluator',
        'AerodynamicConsistencyEvaluator',
        'comprehensive_evaluation'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some dependencies are missing: {e}. Some functionality may not be available.")
    
    # Import only the modules that don't require external dependencies
    from .aero_condition_adapter import AeroConditionAdapter, create_sample_aero_conditions
    
    __all__ = [
        'AeroConditionAdapter',
        'create_sample_aero_conditions'
    ]


def get_version():
    """Get the package version"""
    return __version__


def list_available_components():
    """List available components in the package"""
    return __all__


def quick_start_guide():
    """Print a quick start guide"""
    guide = """
    Point-E Starship Quick Start Guide
    =================================
    
    1. Install dependencies:
       pip install -r requirements.txt
       pip install git+https://github.com/openai/point-e.git
    
    2. Basic usage:
       from point_e_starship import PointEGenerator, create_sample_aero_conditions
       
       # Create sample aerodynamic conditions
       aero_condition = create_sample_aero_conditions()
       
       # Generate point cloud
       generator = PointEGenerator()
       point_clouds = generator.generate_from_aero_conditions(aero_condition)
    
    3. Run examples:
       python examples/basic_generation.py
       python examples/conditional_generation.py
       python examples/batch_generation.py
    
    4. Run full demo:
       python point_e_starship/demo.py
    
    For more information, see the documentation and examples.
    """
    print(guide)


# Package metadata
PACKAGE_INFO = {
    'name': 'point_e_starship',
    'version': __version__,
    'description': 'Point cloud generation for Starship using Point-E and aerodynamic conditions',
    'features': [
        'Aerodynamic condition parsing and analysis',
        'Point-E model integration',
        'Synthetic point cloud generation',
        'Batch processing capabilities',
        'Comprehensive evaluation metrics',
        'Interactive visualizations',
        'Model training interface'
    ],
    'requirements': [
        'Point-E (OpenAI)',
        'PyTorch',
        'NumPy',
        'SciPy',
        'Matplotlib',
        'Plotly',
        'Open3D (optional)',
        'Trimesh',
        'scikit-learn'
    ]
}


def get_package_info():
    """Get package information"""
    return PACKAGE_INFO