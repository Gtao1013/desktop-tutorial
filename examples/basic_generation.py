"""
Basic Point Cloud Generation Example
Simple example showing how to generate point clouds from text descriptions
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from point_e_starship.aero_condition_adapter import AeroConditionAdapter, create_sample_aero_conditions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def basic_text_generation():
    """Basic example of text-to-point-cloud generation"""
    try:
        from point_e_starship.point_e_generator import PointEGenerator
        
        print("=== Basic Text-to-Point-Cloud Generation ===")
        
        # Initialize generator
        print("Initializing Point-E generator...")
        generator = PointEGenerator()
        
        # Define text prompts
        text_prompts = [
            "A sleek futuristic spacecraft with aerodynamic design",
            "A rocket ship with fins and smooth curves",
            "An aerospace vehicle with low drag configuration"
        ]
        
        # Generate point clouds
        output_dir = Path("./basic_generation_output")
        output_dir.mkdir(exist_ok=True)
        
        for i, prompt in enumerate(text_prompts):
            print(f"\nGenerating point cloud {i+1}: '{prompt}'")
            
            # Generate single point cloud
            point_clouds = generator.generate_from_text(
                text_prompt=prompt,
                num_samples=1,
                guidance_scale=15.0,
                upsample=True
            )
            
            # Save result
            if point_clouds:
                output_file = output_dir / f"basic_generation_{i+1}.ply"
                generator.save_point_clouds(
                    point_clouds,
                    output_dir=str(output_dir),
                    prefix=f"basic_{i+1}",
                    format="ply"
                )
                print(f"Saved point cloud to {output_file}")
            else:
                print("Generation failed")
        
        print(f"\nBasic generation completed! Check {output_dir} for results.")
        
    except ImportError:
        print("Point-E not available. Please install with:")
        print("pip install git+https://github.com/openai/point-e.git")
    except Exception as e:
        print(f"Generation failed: {e}")
        print("This might be due to missing Point-E installation or GPU issues")


def synthetic_generation_example():
    """Example using synthetic point cloud generation without Point-E"""
    try:
        from point_e_starship.data_processor import PointCloudProcessor
        from point_e_starship.visualization import PointCloudVisualizer
        
        print("\n=== Synthetic Point Cloud Generation ===")
        
        # Initialize components
        processor = PointCloudProcessor()
        visualizer = PointCloudVisualizer()
        
        # Create sample aerodynamic conditions
        aero_condition = create_sample_aero_conditions()
        print(f"Using aerodynamic condition with shape: {aero_condition.shape}")
        
        # Generate synthetic point cloud
        print("Generating synthetic Starship point cloud...")
        point_cloud = processor.generate_starship_pointcloud(
            aero_conditions=aero_condition,
            num_points=2048
        )
        
        print(f"Generated point cloud with shape: {point_cloud.shape}")
        
        # Save point cloud
        output_dir = Path("./basic_generation_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save in multiple formats
        processor.save_point_cloud(
            point_cloud,
            str(output_dir / "synthetic_starship.ply"),
            format="ply"
        )
        
        processor.save_point_cloud(
            point_cloud,
            str(output_dir / "synthetic_starship.npy"),
            format="npy"
        )
        
        # Create visualization
        fig = visualizer.plot_point_cloud_matplotlib(
            point_cloud,
            title="Synthetic Starship Point Cloud",
            color_by="rgb",
            save_path=str(output_dir / "synthetic_starship_plot.png")
        )
        
        # Create interactive visualization
        fig_interactive = visualizer.plot_point_cloud_plotly(
            point_cloud,
            title="Interactive Synthetic Starship",
            color_by="rgb",
            save_path=str(output_dir / "synthetic_starship_interactive.html")
        )
        
        print(f"Synthetic generation completed! Check {output_dir} for results.")
        
    except Exception as e:
        print(f"Synthetic generation failed: {e}")


def text_description_example():
    """Example showing aerodynamic condition to text conversion"""
    try:
        print("\n=== Aerodynamic Condition to Text Example ===")
        
        # Initialize adapter
        adapter = AeroConditionAdapter()
        
        # Create several sample conditions
        print("Creating sample aerodynamic conditions...")
        
        for i in range(3):
            # Create base condition with variation
            base_condition = create_sample_aero_conditions()
            
            # Add variation for different examples
            if i == 1:
                # High performance variant - apply to each triplet
                multipliers = np.tile([1.2, 0.8, 1.0], 7)  # Higher CL, lower CD
                base_condition = base_condition * multipliers
            elif i == 2:
                # High drag variant - apply to each triplet
                multipliers = np.tile([0.8, 1.5, 1.2], 7)  # Lower CL, higher CD
                base_condition = base_condition * multipliers
            
            print(f"\n--- Condition {i+1} ---")
            
            # Parse condition
            aero_data = adapter.parse_aero_vector(base_condition)
            analysis = adapter.analyze_aerodynamic_characteristics(aero_data)
            
            # Show analysis
            print(f"Max lift coefficient: {analysis['max_lift_coefficient']:.3f}")
            print(f"Min drag coefficient: {analysis['min_drag_coefficient']:.3f}")
            print(f"Max L/D ratio: {analysis['max_ld_ratio']:.1f}")
            print(f"Operating envelope: {analysis['operating_envelope']}")
            
            # Generate text description
            text_description = adapter.to_text_description(base_condition)
            print(f"Text description: {text_description}")
        
    except Exception as e:
        print(f"Text description example failed: {e}")


def main():
    """Run all basic examples"""
    print("Point-E Starship Basic Generation Examples")
    print("=" * 45)
    
    # Run text description example (always works)
    text_description_example()
    
    # Run synthetic generation example (should always work)
    synthetic_generation_example()
    
    # Try Point-E generation (may fail if not installed)
    basic_text_generation()
    
    print("\n" + "=" * 45)
    print("Basic examples completed!")
    print("Check './basic_generation_output' for generated files")


if __name__ == "__main__":
    main()