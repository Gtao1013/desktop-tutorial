"""
Point-E Environment Setup
Configures Point-E model environment, downloads models, and sets up CUDA if available.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointESetup:
    """Point-E environment configuration and setup"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize Point-E setup
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/point_e")
        self.device = self._setup_device()
        self._ensure_cache_dir()
        
    def _setup_device(self) -> torch.device:
        """Setup and return the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU.")
        
        return device
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
    
    def load_point_e_models(self) -> Dict[str, Any]:
        """
        Load Point-E models for text-to-point-cloud generation
        
        Returns:
            Dictionary containing loaded models
        """
        try:
            from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
            from point_e.diffusion.sampler import PointCloudSampler
            from point_e.models.download import load_checkpoint
            from point_e.models.configs import MODEL_CONFIGS, model_from_config
            from point_e.util.plotting import plot_point_cloud
            
            logger.info("Loading Point-E models...")
            
            # Load base model
            base_name = 'base40M-textvec'
            base_model = model_from_config(MODEL_CONFIGS[base_name], device=self.device)
            base_model.load_state_dict(load_checkpoint(base_name, self.device))
            
            # Load upsampler model
            upsampler_name = 'upsample'
            upsampler_model = model_from_config(MODEL_CONFIGS[upsampler_name], device=self.device)
            upsampler_model.load_state_dict(load_checkpoint(upsampler_name, self.device))
            
            # Load diffusion configs
            base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
            upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[upsampler_name])
            
            # Create samplers
            base_sampler = PointCloudSampler(
                device=self.device,
                models=[base_model],
                diffusions=[base_diffusion],
                num_points=[1024],
                aux_channels=['R', 'G', 'B'],
            )
            
            upsampler_sampler = PointCloudSampler(
                device=self.device,
                models=[upsampler_model],
                diffusions=[upsampler_diffusion],
                num_points=[4096],
                aux_channels=['R', 'G', 'B'],
            )
            
            models = {
                'base_model': base_model,
                'upsampler_model': upsampler_model,
                'base_sampler': base_sampler,
                'upsampler_sampler': upsampler_sampler,
                'device': self.device
            }
            
            logger.info("Point-E models loaded successfully!")
            return models
            
        except ImportError as e:
            logger.error(f"Failed to import Point-E modules: {e}")
            logger.error("Please install Point-E: pip install git+https://github.com/openai/point-e.git")
            raise
        except Exception as e:
            logger.error(f"Failed to load Point-E models: {e}")
            raise
    
    def test_installation(self) -> bool:
        """
        Test Point-E installation by loading models
        
        Returns:
            True if installation is working correctly
        """
        try:
            models = self.load_point_e_models()
            logger.info("Point-E installation test passed!")
            return True
        except Exception as e:
            logger.error(f"Point-E installation test failed: {e}")
            return False
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get memory information for optimization"""
        if torch.cuda.is_available():
            return {
                'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'allocated_memory_gb': torch.cuda.memory_allocated() / 1e9,
                'cached_memory_gb': torch.cuda.memory_reserved() / 1e9
            }
        else:
            return {'cpu_memory': 'CPU mode - no GPU memory info'}


def setup_point_e(cache_dir: Optional[str] = None, test: bool = True) -> PointESetup:
    """
    Convenience function to setup Point-E environment
    
    Args:
        cache_dir: Cache directory for models
        test: Whether to test installation
        
    Returns:
        Configured PointESetup instance
    """
    setup = PointESetup(cache_dir=cache_dir)
    
    if test:
        if not setup.test_installation():
            raise RuntimeError("Point-E installation test failed")
    
    return setup


if __name__ == "__main__":
    # Test setup when run directly
    print("Setting up Point-E environment...")
    setup = setup_point_e()
    print("Setup complete!")
    
    # Print memory info
    memory_info = setup.get_memory_info()
    print("\nMemory Information:")
    for key, value in memory_info.items():
        print(f"  {key}: {value}")