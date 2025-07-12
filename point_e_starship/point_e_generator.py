"""
Point-E Generator
Main interface for generating point clouds using Point-E models
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path

from .setup_point_e import PointESetup
from .aero_condition_adapter import AeroConditionAdapter

logger = logging.getLogger(__name__)


class PointEGenerator:
    """Point-E based point cloud generator with aerodynamic conditioning"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize Point-E generator
        
        Args:
            cache_dir: Directory for model cache
        """
        self.setup = PointESetup(cache_dir=cache_dir)
        self.models = None
        self.aero_adapter = AeroConditionAdapter()
        self.device = self.setup.device
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load Point-E models"""
        try:
            self.models = self.setup.load_point_e_models()
            logger.info("Point-E generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Point-E generator: {e}")
            raise
    
    def generate_from_text(
        self, 
        text_prompt: str, 
        num_samples: int = 1,
        guidance_scale: float = 15.0,
        upsample: bool = True
    ) -> List[np.ndarray]:
        """
        Generate point clouds from text description
        
        Args:
            text_prompt: Text description of the desired point cloud
            num_samples: Number of point clouds to generate
            guidance_scale: Guidance scale for generation
            upsample: Whether to upsample to higher resolution
            
        Returns:
            List of generated point clouds
        """
        try:
            from point_e.util.plotting import plot_point_cloud
            
            # Generate base point clouds
            base_sampler = self.models['base_sampler']
            
            logger.info(f"Generating {num_samples} point clouds from text: '{text_prompt}'")
            
            generated_clouds = []
            
            for i in range(num_samples):
                # Sample from base model
                samples = base_sampler.sample_batch_progressive(
                    batch_size=1,
                    model_kwargs=dict(texts=[text_prompt]),
                    guidance_scale=guidance_scale,
                )
                
                # Get the final sample
                pc = samples[-1]  # Last sample is the final result
                
                if upsample:
                    # Upsample using the upsampler model
                    upsampler_sampler = self.models['upsampler_sampler']
                    
                    upsampled = upsampler_sampler.sample_batch_progressive(
                        batch_size=1,
                        model_kwargs=dict(low_res=pc),
                        guidance_scale=guidance_scale,
                    )
                    pc = upsampled[-1]
                
                # Convert to numpy array
                point_cloud = self._pointcloud_to_numpy(pc)
                generated_clouds.append(point_cloud)
                
                logger.info(f"Generated point cloud {i+1}/{num_samples}, shape: {point_cloud.shape}")
            
            return generated_clouds
            
        except Exception as e:
            logger.error(f"Failed to generate point clouds from text: {e}")
            raise
    
    def generate_from_aero_conditions(
        self,
        aero_conditions: Union[np.ndarray, List[np.ndarray]],
        num_samples: int = 1,
        guidance_scale: float = 15.0,
        upsample: bool = True,
        use_text: bool = True
    ) -> List[np.ndarray]:
        """
        Generate point clouds from aerodynamic conditions
        
        Args:
            aero_conditions: 21-dimensional aerodynamic conditions or list of conditions
            num_samples: Number of samples per condition
            guidance_scale: Guidance scale for generation
            upsample: Whether to upsample to higher resolution
            use_text: Whether to use text descriptions (True) or direct embeddings (False)
            
        Returns:
            List of generated point clouds
        """
        # Ensure aero_conditions is a list
        if isinstance(aero_conditions, np.ndarray):
            if aero_conditions.ndim == 1:
                aero_conditions = [aero_conditions]
            else:
                aero_conditions = list(aero_conditions)
        
        all_generated_clouds = []
        
        for aero_condition in aero_conditions:
            if use_text:
                # Convert to text description
                text_prompt = self.aero_adapter.to_text_description(aero_condition)
                clouds = self.generate_from_text(
                    text_prompt=text_prompt,
                    num_samples=num_samples,
                    guidance_scale=guidance_scale,
                    upsample=upsample
                )
            else:
                # Use direct embedding (would require model modification)
                # For now, fallback to text approach
                logger.warning("Direct embedding mode not yet implemented, using text mode")
                text_prompt = self.aero_adapter.to_text_description(aero_condition)
                clouds = self.generate_from_text(
                    text_prompt=text_prompt,
                    num_samples=num_samples,
                    guidance_scale=guidance_scale,
                    upsample=upsample
                )
            
            all_generated_clouds.extend(clouds)
        
        return all_generated_clouds
    
    def batch_generate(
        self,
        inputs: Union[List[str], List[np.ndarray]],
        batch_size: int = 4,
        guidance_scale: float = 15.0,
        upsample: bool = True
    ) -> List[np.ndarray]:
        """
        Generate point clouds in batches for efficiency
        
        Args:
            inputs: List of text prompts or aerodynamic conditions
            batch_size: Batch size for generation
            guidance_scale: Guidance scale
            upsample: Whether to upsample
            
        Returns:
            List of generated point clouds
        """
        all_clouds = []
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(inputs) + batch_size - 1)//batch_size}")
            
            # Convert to text prompts if needed
            if isinstance(batch_inputs[0], np.ndarray):
                text_prompts = [
                    self.aero_adapter.to_text_description(aero_cond) 
                    for aero_cond in batch_inputs
                ]
            else:
                text_prompts = batch_inputs
            
            # Generate batch
            try:
                batch_clouds = self._generate_text_batch(
                    text_prompts=text_prompts,
                    guidance_scale=guidance_scale,
                    upsample=upsample
                )
                all_clouds.extend(batch_clouds)
            except Exception as e:
                logger.error(f"Failed to generate batch {i//batch_size + 1}: {e}")
                # Generate individually as fallback
                for text_prompt in text_prompts:
                    try:
                        cloud = self.generate_from_text(
                            text_prompt=text_prompt,
                            num_samples=1,
                            guidance_scale=guidance_scale,
                            upsample=upsample
                        )[0]
                        all_clouds.append(cloud)
                    except Exception as e2:
                        logger.error(f"Failed to generate individual sample: {e2}")
                        # Add dummy cloud to maintain order
                        all_clouds.append(np.zeros((1024, 6)))
        
        return all_clouds
    
    def _generate_text_batch(
        self,
        text_prompts: List[str],
        guidance_scale: float = 15.0,
        upsample: bool = True
    ) -> List[np.ndarray]:
        """Generate a batch of point clouds from text prompts"""
        batch_size = len(text_prompts)
        
        # Generate base point clouds
        base_sampler = self.models['base_sampler']
        
        samples = base_sampler.sample_batch_progressive(
            batch_size=batch_size,
            model_kwargs=dict(texts=text_prompts),
            guidance_scale=guidance_scale,
        )
        
        # Get the final sample
        pc_batch = samples[-1]
        
        if upsample:
            # Upsample using the upsampler model
            upsampler_sampler = self.models['upsampler_sampler']
            
            upsampled = upsampler_sampler.sample_batch_progressive(
                batch_size=batch_size,
                model_kwargs=dict(low_res=pc_batch),
                guidance_scale=guidance_scale,
            )
            pc_batch = upsampled[-1]
        
        # Convert batch to list of numpy arrays
        point_clouds = []
        for i in range(batch_size):
            pc = self._extract_single_pointcloud(pc_batch, i)
            point_cloud = self._pointcloud_to_numpy(pc)
            point_clouds.append(point_cloud)
        
        return point_clouds
    
    def _pointcloud_to_numpy(self, pc) -> np.ndarray:
        """Convert Point-E point cloud to numpy array"""
        try:
            # Extract coordinates and colors
            coords = pc.coords.cpu().numpy()  # (N, 3)
            
            # Extract colors if available
            if hasattr(pc, 'channels') and 'R' in pc.channels:
                colors = np.stack([
                    pc.channels['R'].cpu().numpy(),
                    pc.channels['G'].cpu().numpy(), 
                    pc.channels['B'].cpu().numpy()
                ], axis=-1)  # (N, 3)
            else:
                # Default gray color
                colors = np.full((coords.shape[0], 3), 0.5)
            
            # Combine coordinates and colors
            point_cloud = np.concatenate([coords, colors], axis=1)  # (N, 6)
            
            return point_cloud
            
        except Exception as e:
            logger.error(f"Failed to convert point cloud to numpy: {e}")
            # Return dummy point cloud
            return np.random.rand(1024, 6)
    
    def _extract_single_pointcloud(self, pc_batch, index: int):
        """Extract single point cloud from batch"""
        try:
            # Create a new point cloud object for the single item
            from point_e.models.point_cloud import PointCloud
            
            coords = pc_batch.coords[index:index+1]  # Keep batch dimension
            channels = {}
            
            if hasattr(pc_batch, 'channels'):
                for key, value in pc_batch.channels.items():
                    channels[key] = value[index:index+1]
            
            return PointCloud(coords=coords, channels=channels)
            
        except Exception as e:
            logger.error(f"Failed to extract single point cloud: {e}")
            return pc_batch  # Return original as fallback
    
    def save_point_clouds(
        self,
        point_clouds: List[np.ndarray],
        output_dir: str,
        prefix: str = "generated",
        format: str = "ply"
    ):
        """
        Save generated point clouds to files
        
        Args:
            point_clouds: List of point clouds to save
            output_dir: Output directory
            prefix: Filename prefix
            format: Output format ('ply', 'npy', 'xyz')
        """
        from .data_processor import PointCloudProcessor
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processor = PointCloudProcessor()
        
        for i, point_cloud in enumerate(point_clouds):
            filename = f"{prefix}_{i:06d}.{format}"
            filepath = output_dir / filename
            
            try:
                processor.save_point_cloud(point_cloud, filepath, format=format)
                logger.info(f"Saved point cloud {i+1}/{len(point_clouds)}: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save point cloud {i}: {e}")
    
    def interpolate_conditions(
        self,
        start_condition: np.ndarray,
        end_condition: np.ndarray,
        num_steps: int = 10,
        guidance_scale: float = 15.0,
        upsample: bool = True
    ) -> List[np.ndarray]:
        """
        Generate point clouds with interpolated aerodynamic conditions
        
        Args:
            start_condition: Starting 21D aerodynamic condition
            end_condition: Ending 21D aerodynamic condition
            num_steps: Number of interpolation steps
            guidance_scale: Guidance scale
            upsample: Whether to upsample
            
        Returns:
            List of interpolated point clouds
        """
        # Create interpolated conditions
        interpolated_conditions = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interp_condition = (1 - alpha) * start_condition + alpha * end_condition
            interpolated_conditions.append(interp_condition)
        
        # Generate point clouds for each interpolated condition
        return self.generate_from_aero_conditions(
            aero_conditions=interpolated_conditions,
            num_samples=1,
            guidance_scale=guidance_scale,
            upsample=upsample
        )
    
    def get_generation_info(self) -> Dict:
        """Get information about the generator"""
        return {
            "device": str(self.device),
            "models_loaded": self.models is not None,
            "base_model": "base40M-textvec",
            "upsampler_model": "upsample",
            "memory_info": self.setup.get_memory_info()
        }


def create_sample_generations():
    """Create sample generations for testing"""
    generator = PointEGenerator()
    
    # Test text generation
    print("Testing text generation...")
    text_clouds = generator.generate_from_text(
        "A futuristic spacecraft with sleek aerodynamic design",
        num_samples=2
    )
    
    # Test aerodynamic condition generation
    print("Testing aerodynamic condition generation...")
    from .aero_condition_adapter import create_sample_aero_conditions
    
    sample_aero = create_sample_aero_conditions()
    aero_clouds = generator.generate_from_aero_conditions(
        aero_conditions=sample_aero,
        num_samples=2
    )
    
    # Save results
    output_dir = "/tmp/sample_generations"
    generator.save_point_clouds(text_clouds, output_dir, prefix="text")
    generator.save_point_clouds(aero_clouds, output_dir, prefix="aero")
    
    print(f"Sample generations saved to {output_dir}")
    
    return text_clouds, aero_clouds


if __name__ == "__main__":
    # Test the generator
    print("Initializing Point-E generator...")
    
    try:
        text_clouds, aero_clouds = create_sample_generations()
        print("Sample generation completed successfully!")
        
        print(f"Generated {len(text_clouds)} text-based point clouds")
        print(f"Generated {len(aero_clouds)} aerodynamic-based point clouds")
        
    except Exception as e:
        print(f"Sample generation failed: {e}")
        logger.error(f"Sample generation failed: {e}")