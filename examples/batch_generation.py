"""
Batch Point Cloud Generation Example
Example showing how to efficiently generate multiple point clouds in batches
"""

import numpy as np
import logging
from pathlib import Path
import time
import json
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_batch_aero_conditions(batch_size: int = 20) -> List[np.ndarray]:
    """Create a batch of aerodynamic conditions with systematic variations"""
    from point_e_starship.aero_condition_adapter import create_sample_aero_conditions
    
    conditions = []
    base_condition = create_sample_aero_conditions()
    
    print(f"Creating {batch_size} aerodynamic conditions...")
    
    # Create systematic variations
    for i in range(batch_size):
        # Create different types of variations
        variation_type = i % 4
        
        if variation_type == 0:
            # Lift coefficient variation
            scale_factors = [1.0 + 0.4 * (i / batch_size), 1.0, 1.0]
        elif variation_type == 1:
            # Drag coefficient variation
            scale_factors = [1.0, 1.0 + 0.3 * (i / batch_size), 1.0]
        elif variation_type == 2:
            # Moment coefficient variation
            scale_factors = [1.0, 1.0, 1.0 + 0.2 * (i / batch_size)]
        else:
            # Combined variation
            scale_factors = [
                1.0 + 0.2 * np.sin(2 * np.pi * i / batch_size),
                1.0 + 0.15 * np.cos(2 * np.pi * i / batch_size),
                1.0 + 0.1 * np.sin(4 * np.pi * i / batch_size)
            ]
        
        # Apply variation to each angle of attack
        varied_condition = base_condition.copy()
        for angle_idx in range(7):
            start_idx = angle_idx * 3
            varied_condition[start_idx:start_idx+3] *= scale_factors
        
        # Add small random noise
        noise = np.random.normal(0, 0.02, 21)
        varied_condition += noise
        
        conditions.append(varied_condition)
    
    return conditions


def batch_point_e_generation():
    """Batch generation using Point-E models"""
    try:
        from point_e_starship.point_e_generator import PointEGenerator
        from point_e_starship.aero_condition_adapter import AeroConditionAdapter
        
        print("=== Batch Point-E Generation ===")
        
        # Initialize components
        generator = PointEGenerator()
        adapter = AeroConditionAdapter()
        
        # Create batch of aerodynamic conditions
        batch_size = 8  # Smaller batch for Point-E due to memory constraints
        aero_conditions = create_batch_aero_conditions(batch_size)
        
        output_dir = Path("./batch_generation_output")
        output_dir.mkdir(exist_ok=True)
        
        # Batch generation
        print(f"Starting batch generation of {batch_size} point clouds...")
        start_time = time.time()
        
        # Method 1: Generate all at once (if possible)
        try:
            generated_clouds = generator.batch_generate(
                inputs=aero_conditions,
                batch_size=4,  # Process in smaller sub-batches
                guidance_scale=15.0,
                upsample=True
            )
            
            batch_time = time.time() - start_time
            print(f"Batch generation completed in {batch_time:.2f} seconds")
            print(f"Average time per cloud: {batch_time/len(generated_clouds):.2f} seconds")
            
        except Exception as e:
            print(f"Batch generation failed: {e}")
            print("Falling back to individual generation...")
            
            # Method 2: Generate individually
            generated_clouds = []
            start_time = time.time()
            
            for i, aero_condition in enumerate(aero_conditions):
                print(f"Generating cloud {i+1}/{len(aero_conditions)}")
                
                try:
                    clouds = generator.generate_from_aero_conditions(
                        aero_conditions=aero_condition,
                        num_samples=1,
                        guidance_scale=15.0,
                        upsample=True
                    )
                    
                    if clouds:
                        generated_clouds.extend(clouds)
                    else:
                        print(f"Failed to generate cloud {i+1}")
                        
                except Exception as gen_e:
                    print(f"Generation failed for cloud {i+1}: {gen_e}")
            
            individual_time = time.time() - start_time
            print(f"Individual generation completed in {individual_time:.2f} seconds")
            if generated_clouds:
                print(f"Average time per cloud: {individual_time/len(generated_clouds):.2f} seconds")
        
        # Save generated clouds
        if generated_clouds:
            generator.save_point_clouds(
                generated_clouds,
                output_dir=str(output_dir),
                prefix="batch_pointe",
                format="ply"
            )
            
            # Create batch analysis
            create_batch_analysis(generated_clouds, aero_conditions[:len(generated_clouds)], output_dir)
            
            print(f"Successfully generated {len(generated_clouds)} point clouds")
        else:
            print("No point clouds were generated")
        
    except ImportError:
        print("Point-E not available. Please install with:")
        print("pip install git+https://github.com/openai/point-e.git")
    except Exception as e:
        print(f"Batch Point-E generation failed: {e}")


def batch_synthetic_generation():
    """Batch generation using synthetic point cloud generation"""
    try:
        from point_e_starship.data_processor import PointCloudProcessor
        from point_e_starship.aero_condition_adapter import AeroConditionAdapter
        
        print("\n=== Batch Synthetic Generation ===")
        
        # Initialize components
        processor = PointCloudProcessor()
        adapter = AeroConditionAdapter()
        
        # Create large batch of aerodynamic conditions
        batch_size = 50
        aero_conditions = create_batch_aero_conditions(batch_size)
        
        output_dir = Path("./batch_generation_output")
        output_dir.mkdir(exist_ok=True)
        
        # Batch synthetic generation
        print(f"Starting batch synthetic generation of {batch_size} point clouds...")
        start_time = time.time()
        
        generated_clouds = []
        
        for i, aero_condition in enumerate(aero_conditions):
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{batch_size} clouds...")
            
            # Generate synthetic point cloud
            point_cloud = processor.generate_starship_pointcloud(
                aero_conditions=aero_condition,
                num_points=1024
            )
            
            generated_clouds.append(point_cloud)
            
            # Save individual cloud
            processor.save_point_cloud(
                point_cloud,
                str(output_dir / f"batch_synthetic_{i:03d}.ply"),
                format="ply"
            )
        
        generation_time = time.time() - start_time
        print(f"Batch synthetic generation completed in {generation_time:.2f} seconds")
        print(f"Average time per cloud: {generation_time/batch_size:.3f} seconds")
        
        # Create batch analysis
        create_batch_analysis(generated_clouds, aero_conditions, output_dir)
        
        # Create dataset for training
        create_training_dataset(generated_clouds, aero_conditions, output_dir)
        
        print(f"Successfully generated {len(generated_clouds)} synthetic point clouds")
        
    except Exception as e:
        print(f"Batch synthetic generation failed: {e}")


def create_batch_analysis(point_clouds: List[np.ndarray], aero_conditions: List[np.ndarray], output_dir: Path):
    """Create analysis of the batch generation results"""
    try:
        from point_e_starship.evaluation import comprehensive_evaluation
        from point_e_starship.aero_condition_adapter import AeroConditionAdapter
        
        print("Creating batch analysis...")
        
        # Comprehensive evaluation
        results = comprehensive_evaluation(
            generated_clouds=point_clouds,
            aero_conditions=aero_conditions,
            output_path=str(output_dir / "batch_evaluation.json")
        )
        
        # Extract key metrics for summary
        adapter = AeroConditionAdapter()
        
        # Analyze aerodynamic condition diversity
        aero_metrics = []
        for aero_condition in aero_conditions:
            aero_data = adapter.parse_aero_vector(aero_condition)
            analysis = adapter.analyze_aerodynamic_characteristics(aero_data)
            aero_metrics.append({
                'max_cl': analysis['max_lift_coefficient'],
                'min_cd': analysis['min_drag_coefficient'],
                'max_ld': analysis['max_ld_ratio'],
                'stall_angle': analysis['stall_angle'] or 30
            })
        
        # Create summary statistics
        summary_stats = {
            'num_clouds': len(point_clouds),
            'aerodynamic_diversity': {
                'cl_range': [
                    min(m['max_cl'] for m in aero_metrics),
                    max(m['max_cl'] for m in aero_metrics)
                ],
                'cd_range': [
                    min(m['min_cd'] for m in aero_metrics),
                    max(m['min_cd'] for m in aero_metrics)
                ],
                'ld_range': [
                    min(m['max_ld'] for m in aero_metrics),
                    max(m['max_ld'] for m in aero_metrics)
                ]
            },
            'quality_metrics': results['summary']
        }
        
        # Save summary
        with open(output_dir / "batch_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create plots
        create_batch_plots(aero_metrics, output_dir)
        
        print(f"Batch analysis saved to {output_dir}")
        
    except Exception as e:
        print(f"Failed to create batch analysis: {e}")


def create_batch_plots(aero_metrics: List[Dict], output_dir: Path):
    """Create plots for batch analysis"""
    try:
        import matplotlib.pyplot as plt
        
        # Extract metrics
        cl_values = [m['max_cl'] for m in aero_metrics]
        cd_values = [m['min_cd'] for m in aero_metrics]
        ld_values = [m['max_ld'] for m in aero_metrics]
        stall_angles = [m['stall_angle'] for m in aero_metrics]
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(cl_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Maximum CL')
        axes[0, 0].set_xlabel('CL_max')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(cd_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Distribution of Minimum CD')
        axes[0, 1].set_xlabel('CD_min')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].hist(ld_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Distribution of Maximum L/D')
        axes[1, 0].set_xlabel('(L/D)_max')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(stall_angles, bins=20, alpha=0.7, color='wheat', edgecolor='black')
        axes[1, 1].set_title('Distribution of Stall Angles')
        axes[1, 1].set_xlabel('Stall Angle (degrees)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.suptitle('Batch Generation - Aerodynamic Parameter Distributions')
        plt.tight_layout()
        plt.savefig(output_dir / "batch_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # CL vs CD
        axes[0].scatter(cd_values, cl_values, alpha=0.6, s=30)
        axes[0].set_xlabel('CD_min')
        axes[0].set_ylabel('CL_max')
        axes[0].set_title('CL vs CD Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # L/D vs Stall Angle
        axes[1].scatter(stall_angles, ld_values, alpha=0.6, s=30, c=cl_values, cmap='viridis')
        axes[1].set_xlabel('Stall Angle (degrees)')
        axes[1].set_ylabel('(L/D)_max')
        axes[1].set_title('L/D vs Stall Angle (colored by CL_max)')
        axes[1].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
        cbar.set_label('CL_max')
        
        plt.tight_layout()
        plt.savefig(output_dir / "batch_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Batch analysis plots created")
        
    except Exception as e:
        print(f"Failed to create batch plots: {e}")


def create_training_dataset(point_clouds: List[np.ndarray], aero_conditions: List[np.ndarray], output_dir: Path):
    """Create a formatted dataset suitable for training"""
    try:
        import h5py
        
        print("Creating training dataset...")
        
        # Convert to numpy arrays
        point_clouds_array = np.array(point_clouds)
        aero_conditions_array = np.array(aero_conditions)
        
        # Save as HDF5 for efficient loading
        dataset_path = output_dir / "training_dataset.h5"
        
        with h5py.File(dataset_path, 'w') as f:
            f.create_dataset("point_clouds", data=point_clouds_array)
            f.create_dataset("aero_conditions", data=aero_conditions_array)
            
            # Add metadata
            f.attrs['num_samples'] = len(point_clouds)
            f.attrs['num_points'] = point_clouds[0].shape[0]
            f.attrs['point_cloud_shape'] = point_clouds_array.shape
            f.attrs['aero_conditions_shape'] = aero_conditions_array.shape
            f.attrs['description'] = "Batch generated dataset for Point-E training"
        
        # Create metadata file
        metadata = {
            "dataset_info": {
                "num_samples": len(point_clouds),
                "num_points_per_cloud": point_clouds[0].shape[0],
                "point_cloud_dimensions": point_clouds[0].shape[1],
                "aero_condition_dimensions": aero_conditions[0].shape[0]
            },
            "data_format": {
                "point_clouds": "Array of shape (N, num_points, 6) with XYZ + RGB",
                "aero_conditions": "Array of shape (N, 21) with 7 angles × 3 coefficients"
            },
            "usage": {
                "training": "Use for Point-E model fine-tuning",
                "evaluation": "Use for model evaluation and comparison"
            }
        }
        
        with open(output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Training dataset saved to {dataset_path}")
        print(f"Dataset contains {len(point_clouds)} samples")
        
    except Exception as e:
        print(f"Failed to create training dataset: {e}")


def performance_comparison():
    """Compare performance of different generation methods"""
    print("\n=== Performance Comparison ===")
    
    # Test different batch sizes for synthetic generation
    batch_sizes = [10, 25, 50]
    performance_results = {}
    
    try:
        from point_e_starship.data_processor import PointCloudProcessor
        
        processor = PointCloudProcessor()
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create conditions
            aero_conditions = create_batch_aero_conditions(batch_size)
            
            # Time the generation
            start_time = time.time()
            
            for aero_condition in aero_conditions:
                point_cloud = processor.generate_starship_pointcloud(
                    aero_conditions=aero_condition,
                    num_points=1024
                )
            
            total_time = time.time() - start_time
            avg_time = total_time / batch_size
            
            performance_results[batch_size] = {
                'total_time': total_time,
                'avg_time_per_cloud': avg_time,
                'clouds_per_second': 1.0 / avg_time
            }
            
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average per cloud: {avg_time:.3f}s")
            print(f"  Clouds per second: {1.0/avg_time:.2f}")
        
        # Save performance results
        output_dir = Path("./batch_generation_output")
        with open(output_dir / "performance_comparison.json", 'w') as f:
            json.dump(performance_results, f, indent=2)
        
        print("Performance comparison completed")
        
    except Exception as e:
        print(f"Performance comparison failed: {e}")


def main():
    """Run batch generation examples"""
    print("Point-E Starship Batch Generation Examples")
    print("=" * 45)
    
    # Run synthetic batch generation (should always work)
    batch_synthetic_generation()
    
    # Run performance comparison
    performance_comparison()
    
    # Try Point-E batch generation (may fail if not installed)
    batch_point_e_generation()
    
    print("\n" + "=" * 45)
    print("Batch generation examples completed!")
    print("Check './batch_generation_output' for generated files")
    print("Key outputs:")
    print("  - Individual PLY files for each generated point cloud")
    print("  - training_dataset.h5 for model training")
    print("  - batch_evaluation.json for quality metrics")
    print("  - Visualization plots and analysis")


if __name__ == "__main__":
    main()