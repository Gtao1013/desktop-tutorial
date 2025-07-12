"""
Conditional Point Cloud Generation Example
Example showing how to generate point clouds conditioned on aerodynamic parameters
"""

import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_varied_aero_conditions(num_conditions: int = 6) -> list:
    """Create a set of varied aerodynamic conditions for demonstration"""
    from point_e_starship.aero_condition_adapter import create_sample_aero_conditions
    
    conditions = []
    
    # Base condition
    base_condition = create_sample_aero_conditions()
    
    # Create variations representing different design points
    variations = [
        {"name": "baseline", "multiplier": [1.0, 1.0, 1.0]},
        {"name": "high_lift", "multiplier": [1.5, 1.1, 1.0]},
        {"name": "low_drag", "multiplier": [0.9, 0.7, 1.0]},
        {"name": "high_performance", "multiplier": [1.3, 0.8, 0.9]},
        {"name": "stable_config", "multiplier": [1.1, 1.0, 0.7]},
        {"name": "aggressive_config", "multiplier": [1.6, 1.3, 1.2]}
    ]
    
    for i, variation in enumerate(variations[:num_conditions]):
        # Apply variation pattern across all angles
        varied_condition = base_condition.copy()
        multiplier = variation["multiplier"]
        
        # Apply to each angle of attack
        for angle_idx in range(7):
            start_idx = angle_idx * 3
            varied_condition[start_idx:start_idx+3] *= multiplier
        
        conditions.append({
            "condition": varied_condition,
            "name": variation["name"],
            "description": f"Configuration {i+1}: {variation['name']}"
        })
    
    return conditions


def conditional_point_e_generation():
    """Generate point clouds using Point-E conditioned on aerodynamic parameters"""
    try:
        from point_e_starship.point_e_generator import PointEGenerator
        from point_e_starship.aero_condition_adapter import AeroConditionAdapter
        
        print("=== Conditional Point-E Generation ===")
        
        # Initialize components
        generator = PointEGenerator()
        adapter = AeroConditionAdapter()
        
        # Create varied aerodynamic conditions
        aero_configs = create_varied_aero_conditions(num_conditions=4)
        
        output_dir = Path("./conditional_generation_output")
        output_dir.mkdir(exist_ok=True)
        
        generated_clouds = []
        condition_names = []
        
        for i, config in enumerate(aero_configs):
            condition = config["condition"]
            name = config["name"]
            
            print(f"\n--- Generating for {name} configuration ---")
            
            # Analyze aerodynamic characteristics
            aero_data = adapter.parse_aero_vector(condition)
            analysis = adapter.analyze_aerodynamic_characteristics(aero_data)
            
            print(f"Max CL: {analysis['max_lift_coefficient']:.3f}")
            print(f"Min CD: {analysis['min_drag_coefficient']:.3f}")
            print(f"L/D ratio: {analysis['max_ld_ratio']:.1f}")
            
            # Generate text description
            text_description = adapter.to_text_description(condition)
            print(f"Description: {text_description}")
            
            # Generate point cloud
            try:
                point_clouds = generator.generate_from_aero_conditions(
                    aero_conditions=condition,
                    num_samples=1,
                    guidance_scale=15.0,
                    upsample=True
                )
                
                if point_clouds:
                    # Save point cloud
                    generator.save_point_clouds(
                        point_clouds,
                        output_dir=str(output_dir),
                        prefix=f"conditional_{name}",
                        format="ply"
                    )
                    
                    generated_clouds.extend(point_clouds)
                    condition_names.append(name)
                    
                    print(f"Successfully generated point cloud for {name}")
                else:
                    print(f"Failed to generate point cloud for {name}")
                    
            except Exception as e:
                print(f"Generation failed for {name}: {e}")
        
        # Create comparison visualization if any clouds were generated
        if generated_clouds:
            create_conditional_comparison(generated_clouds, condition_names, output_dir)
        
        print(f"\nConditional generation completed! Check {output_dir} for results.")
        
    except ImportError:
        print("Point-E not available. Please install with:")
        print("pip install git+https://github.com/openai/point-e.git")
    except Exception as e:
        print(f"Conditional generation failed: {e}")


def conditional_synthetic_generation():
    """Generate synthetic point clouds conditioned on aerodynamic parameters"""
    try:
        from point_e_starship.data_processor import PointCloudProcessor
        from point_e_starship.aero_condition_adapter import AeroConditionAdapter
        from point_e_starship.visualization import PointCloudVisualizer, AerodynamicVisualizer
        
        print("\n=== Conditional Synthetic Generation ===")
        
        # Initialize components
        processor = PointCloudProcessor()
        adapter = AeroConditionAdapter()
        pc_visualizer = PointCloudVisualizer()
        aero_visualizer = AerodynamicVisualizer()
        
        # Create varied aerodynamic conditions
        aero_configs = create_varied_aero_conditions(num_conditions=6)
        
        output_dir = Path("./conditional_generation_output")
        output_dir.mkdir(exist_ok=True)
        
        generated_clouds = []
        aero_conditions = []
        condition_names = []
        
        print("Generating synthetic point clouds for different configurations...")
        
        for i, config in enumerate(aero_configs):
            condition = config["condition"]
            name = config["name"]
            description = config["description"]
            
            print(f"\n{description}")
            
            # Analyze condition
            aero_data = adapter.parse_aero_vector(condition)
            analysis = adapter.analyze_aerodynamic_characteristics(aero_data)
            
            print(f"  Max CL: {analysis['max_lift_coefficient']:.3f} @ {analysis['max_lift_angle']}°")
            print(f"  Min CD: {analysis['min_drag_coefficient']:.3f} @ {analysis['min_drag_angle']}°")
            print(f"  Max L/D: {analysis['max_ld_ratio']:.1f}")
            print(f"  Envelope: {analysis['operating_envelope']}")
            
            # Generate synthetic point cloud
            point_cloud = processor.generate_starship_pointcloud(
                aero_conditions=condition,
                num_points=2048
            )
            
            # Save point cloud
            output_file = output_dir / f"synthetic_{name}.ply"
            processor.save_point_cloud(point_cloud, str(output_file), format="ply")
            
            generated_clouds.append(point_cloud)
            aero_conditions.append(condition)
            condition_names.append(name)
            
            # Create individual visualization
            fig = pc_visualizer.plot_point_cloud_matplotlib(
                point_cloud,
                title=f"{description} - Point Cloud",
                color_by="rgb",
                save_path=str(output_dir / f"pointcloud_{name}.png")
            )
            plt.close(fig)
            
            # Create aerodynamic plot
            aero_fig = aero_visualizer.plot_aero_coefficients(
                condition,
                title=f"{description} - Aerodynamic Coefficients",
                save_path=str(output_dir / f"aero_{name}.png")
            )
            plt.close(aero_fig)
        
        # Create comparison visualizations
        create_conditional_comparison(generated_clouds, condition_names, output_dir)
        create_aerodynamic_comparison(aero_conditions, condition_names, output_dir)
        
        print(f"\nConditional synthetic generation completed! Check {output_dir} for results.")
        
    except Exception as e:
        print(f"Conditional synthetic generation failed: {e}")


def create_conditional_comparison(point_clouds: list, names: list, output_dir: Path):
    """Create comparison visualizations for conditional generation"""
    try:
        from point_e_starship.visualization import PointCloudVisualizer
        
        visualizer = PointCloudVisualizer()
        
        # Create comparison plot
        comparison_fig = visualizer.compare_point_clouds(
            point_clouds[:4],  # Limit to first 4 for clarity
            names[:4],
            title="Conditional Point Cloud Generation Comparison",
            save_path=str(output_dir / "conditional_comparison.html")
        )
        
        print(f"Comparison visualization saved to {output_dir / 'conditional_comparison.html'}")
        
    except Exception as e:
        print(f"Failed to create comparison visualization: {e}")


def create_aerodynamic_comparison(aero_conditions: list, names: list, output_dir: Path):
    """Create aerodynamic comparison plots"""
    try:
        from point_e_starship.aero_condition_adapter import AeroConditionAdapter
        from point_e_starship.visualization import AerodynamicVisualizer
        
        adapter = AeroConditionAdapter()
        visualizer = AerodynamicVisualizer()
        
        # Create surface plot showing variation
        surface_fig = visualizer.plot_aero_surface(
            aero_conditions,
            parameter_name="Configuration",
            save_path=str(output_dir / "aero_comparison_surface.html")
        )
        
        # Create comparison of key metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract metrics for all conditions
        max_cls = []
        min_cds = []
        max_lds = []
        stall_angles = []
        
        for condition in aero_conditions:
            aero_data = adapter.parse_aero_vector(condition)
            analysis = adapter.analyze_aerodynamic_characteristics(aero_data)
            
            max_cls.append(analysis['max_lift_coefficient'])
            min_cds.append(analysis['min_drag_coefficient'])
            max_lds.append(analysis['max_ld_ratio'])
            stall_angles.append(analysis['stall_angle'] or 30)
        
        # Plot comparisons
        x_pos = range(len(names))
        
        axes[0, 0].bar(x_pos, max_cls, color='skyblue')
        axes[0, 0].set_title('Maximum Lift Coefficient')
        axes[0, 0].set_ylabel('CL_max')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(names, rotation=45)
        
        axes[0, 1].bar(x_pos, min_cds, color='lightcoral')
        axes[0, 1].set_title('Minimum Drag Coefficient')
        axes[0, 1].set_ylabel('CD_min')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(names, rotation=45)
        
        axes[1, 0].bar(x_pos, max_lds, color='lightgreen')
        axes[1, 0].set_title('Maximum L/D Ratio')
        axes[1, 0].set_ylabel('(L/D)_max')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(names, rotation=45)
        
        axes[1, 1].bar(x_pos, stall_angles, color='wheat')
        axes[1, 1].set_title('Stall Angle')
        axes[1, 1].set_ylabel('Stall Angle (degrees)')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(names, rotation=45)
        
        plt.suptitle('Aerodynamic Characteristics Comparison')
        plt.tight_layout()
        plt.savefig(output_dir / "aero_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Aerodynamic comparison plots saved to {output_dir}")
        
    except Exception as e:
        print(f"Failed to create aerodynamic comparison: {e}")


def interpolation_example():
    """Demonstrate interpolation between different aerodynamic conditions"""
    try:
        from point_e_starship.data_processor import PointCloudProcessor
        from point_e_starship.aero_condition_adapter import AeroConditionAdapter
        
        print("\n=== Aerodynamic Condition Interpolation ===")
        
        processor = PointCloudProcessor()
        adapter = AeroConditionAdapter()
        
        # Create two different conditions
        aero_configs = create_varied_aero_conditions(num_conditions=6)
        start_config = aero_configs[1]  # high_lift
        end_config = aero_configs[2]    # low_drag
        
        start_condition = start_config["condition"]
        end_condition = end_config["condition"]
        
        print(f"Interpolating from {start_config['name']} to {end_config['name']}")
        
        # Create interpolation sequence
        num_steps = 7
        interpolated_conditions = []
        interpolated_clouds = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interp_condition = (1 - alpha) * start_condition + alpha * end_condition
            
            # Analyze interpolated condition
            aero_data = adapter.parse_aero_vector(interp_condition)
            analysis = adapter.analyze_aerodynamic_characteristics(aero_data)
            
            print(f"Step {i+1}: CL_max={analysis['max_lift_coefficient']:.3f}, "
                  f"CD_min={analysis['min_drag_coefficient']:.3f}")
            
            # Generate point cloud
            point_cloud = processor.generate_starship_pointcloud(
                aero_conditions=interp_condition,
                num_points=1024
            )
            
            interpolated_conditions.append(interp_condition)
            interpolated_clouds.append(point_cloud)
        
        # Save interpolation sequence
        output_dir = Path("./conditional_generation_output")
        interp_dir = output_dir / "interpolation"
        interp_dir.mkdir(exist_ok=True)
        
        for i, (condition, cloud) in enumerate(zip(interpolated_conditions, interpolated_clouds)):
            # Save point cloud
            processor.save_point_cloud(
                cloud,
                str(interp_dir / f"interpolated_{i:02d}.ply"),
                format="ply"
            )
        
        # Create animation visualization
        try:
            from point_e_starship.visualization import PointCloudVisualizer
            
            visualizer = PointCloudVisualizer()
            animation_fig = visualizer.plot_generation_animation(
                interpolated_clouds,
                title="Aerodynamic Condition Interpolation",
                save_path=str(interp_dir / "interpolation_animation.html")
            )
            
            print(f"Interpolation animation saved to {interp_dir / 'interpolation_animation.html'}")
            
        except Exception as e:
            print(f"Failed to create animation: {e}")
        
        print(f"Interpolation sequence saved to {interp_dir}")
        
    except Exception as e:
        print(f"Interpolation example failed: {e}")


def main():
    """Run conditional generation examples"""
    print("Point-E Starship Conditional Generation Examples")
    print("=" * 50)
    
    # Run synthetic conditional generation (should always work)
    conditional_synthetic_generation()
    
    # Run interpolation example
    interpolation_example()
    
    # Try Point-E conditional generation (may fail if not installed)
    conditional_point_e_generation()
    
    print("\n" + "=" * 50)
    print("Conditional generation examples completed!")
    print("Check './conditional_generation_output' for generated files")


if __name__ == "__main__":
    main()