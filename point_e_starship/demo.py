"""
Point-E Starship Demo
Main demonstration script showing the complete Point-E pipeline
"""

import numpy as np
import logging
from pathlib import Path
import argparse
import time
from typing import List, Dict, Optional

# Import our modules
from point_e_starship.setup_point_e import setup_point_e
from point_e_starship.aero_condition_adapter import AeroConditionAdapter, create_sample_aero_conditions
from point_e_starship.data_processor import PointCloudProcessor
from point_e_starship.point_e_generator import PointEGenerator
from point_e_starship.visualization import PointCloudVisualizer, AerodynamicVisualizer, create_visualization_gallery
from point_e_starship.evaluation import comprehensive_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PointEStarshipDemo:
    """Complete demonstration of Point-E Starship point cloud generation"""
    
    def __init__(self, output_dir: str = "./demo_output"):
        """
        Initialize demo
        
        Args:
            output_dir: Directory for demo outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.aero_adapter = AeroConditionAdapter()
        self.processor = PointCloudProcessor()
        self.generator = None
        
        logger.info(f"Demo output directory: {self.output_dir}")
    
    def setup_environment(self):
        """Setup Point-E environment"""
        logger.info("Setting up Point-E environment...")
        
        try:
            # Test Point-E installation
            setup = setup_point_e(test=True)
            logger.info("Point-E environment setup completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to setup Point-E environment: {e}")
            logger.error("Please install Point-E: pip install git+https://github.com/openai/point-e.git")
            return False
    
    def create_sample_data(self, num_samples: int = 10) -> tuple:
        """
        Create sample aerodynamic conditions and synthetic point clouds
        
        Args:
            num_samples: Number of samples to create
            
        Returns:
            Tuple of (point_clouds, aero_conditions)
        """
        logger.info(f"Creating {num_samples} sample aerodynamic conditions...")
        
        # Create sample aerodynamic conditions
        sample_aero_conditions = []
        for i in range(num_samples):
            if i == 0:
                # Use the predefined sample for the first one
                aero_cond = create_sample_aero_conditions()
            else:
                # Generate variations
                base_aero = create_sample_aero_conditions()
                # Add some realistic variation
                variation = 1.0 + 0.2 * np.random.randn(21) * 0.1  # 10% variation
                aero_cond = base_aero * variation
            
            sample_aero_conditions.append(aero_cond)
        
        # Generate corresponding synthetic point clouds
        logger.info("Generating synthetic Starship point clouds...")
        synthetic_clouds = []
        for aero_cond in sample_aero_conditions:
            point_cloud = self.processor.generate_starship_pointcloud(
                aero_cond, num_points=1024
            )
            synthetic_clouds.append(point_cloud)
        
        return synthetic_clouds, sample_aero_conditions
    
    def demonstrate_aero_adaptation(self, aero_conditions: List[np.ndarray]):
        """Demonstrate aerodynamic condition adaptation"""
        logger.info("Demonstrating aerodynamic condition adaptation...")
        
        for i, aero_cond in enumerate(aero_conditions[:3]):  # Show first 3
            logger.info(f"\n--- Aerodynamic Condition {i+1} ---")
            
            # Parse and analyze
            aero_data = self.aero_adapter.parse_aero_vector(aero_cond)
            analysis = self.aero_adapter.analyze_aerodynamic_characteristics(aero_data)
            
            # Show key characteristics
            logger.info(f"Max lift coefficient: {analysis['max_lift_coefficient']:.3f} at {analysis['max_lift_angle']}°")
            logger.info(f"Min drag coefficient: {analysis['min_drag_coefficient']:.3f} at {analysis['min_drag_angle']}°")
            logger.info(f"Max L/D ratio: {analysis['max_ld_ratio']:.1f}")
            logger.info(f"Stall angle: {analysis['stall_angle']}°" if analysis['stall_angle'] else "No stall detected")
            logger.info(f"Moment stability: {analysis['moment_stability']}")
            
            # Convert to text description
            text_description = self.aero_adapter.to_text_description(aero_cond)
            logger.info(f"Text description: {text_description}")
    
    def demonstrate_visualization(self, point_clouds: List[np.ndarray], aero_conditions: List[np.ndarray]):
        """Demonstrate visualization capabilities"""
        logger.info("Creating visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Create visualization gallery
        create_visualization_gallery(
            point_clouds[:5],  # Limit to first 5
            aero_conditions[:5],
            output_dir=str(viz_dir)
        )
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def demonstrate_point_e_generation(self, aero_conditions: List[np.ndarray]) -> List[np.ndarray]:
        """Demonstrate Point-E generation (if available)"""
        logger.info("Attempting Point-E point cloud generation...")
        
        try:
            # Initialize generator
            self.generator = PointEGenerator()
            
            # Generate point clouds from aerodynamic conditions
            generated_clouds = self.generator.generate_from_aero_conditions(
                aero_conditions=aero_conditions[:3],  # Generate for first 3 conditions
                num_samples=1,
                guidance_scale=15.0,
                upsample=True
            )
            
            # Save generated point clouds
            gen_dir = self.output_dir / "generated_clouds"
            gen_dir.mkdir(exist_ok=True)
            
            self.generator.save_point_clouds(
                generated_clouds,
                output_dir=str(gen_dir),
                prefix="pointe_generated",
                format="ply"
            )
            
            logger.info(f"Generated {len(generated_clouds)} point clouds using Point-E")
            logger.info(f"Generated clouds saved to {gen_dir}")
            
            return generated_clouds
            
        except Exception as e:
            logger.warning(f"Point-E generation failed: {e}")
            logger.warning("This is expected if Point-E is not properly installed or configured")
            return []
    
    def demonstrate_evaluation(
        self, 
        synthetic_clouds: List[np.ndarray], 
        aero_conditions: List[np.ndarray],
        generated_clouds: Optional[List[np.ndarray]] = None
    ):
        """Demonstrate evaluation capabilities"""
        logger.info("Demonstrating evaluation metrics...")
        
        # Evaluate synthetic clouds
        results = comprehensive_evaluation(
            generated_clouds=synthetic_clouds,
            aero_conditions=aero_conditions,
            reference_clouds=generated_clouds if generated_clouds else None,
            output_path=str(self.output_dir / "evaluation_results.json")
        )
        
        # Log key results
        summary = results['summary']
        logger.info(f"Overall quality score: {summary['overall_quality_score']:.3f}")
        logger.info(f"Aerodynamic consistency: {summary['aerodynamic_consistency']:.3f}")
        
        if generated_clouds:
            logger.info("Comparison with Point-E generated clouds:")
            comparison = results['point_cloud_quality'].get('comparison_metrics', {})
            if 'chamfer_distance' in comparison:
                chamfer_mean = comparison['chamfer_distance']['mean']
                logger.info(f"Average Chamfer distance: {chamfer_mean:.6f}")
    
    def demonstrate_interpolation(self, aero_conditions: List[np.ndarray]):
        """Demonstrate aerodynamic condition interpolation"""
        logger.info("Demonstrating aerodynamic condition interpolation...")
        
        if len(aero_conditions) < 2:
            logger.warning("Need at least 2 conditions for interpolation")
            return
        
        # Interpolate between first two conditions
        start_condition = aero_conditions[0]
        end_condition = aero_conditions[1]
        
        # Show characteristics of start and end conditions
        logger.info("Start condition characteristics:")
        start_analysis = self.aero_adapter.analyze_aerodynamic_characteristics(
            self.aero_adapter.parse_aero_vector(start_condition)
        )
        logger.info(f"  Max CL: {start_analysis['max_lift_coefficient']:.3f}")
        logger.info(f"  Min CD: {start_analysis['min_drag_coefficient']:.3f}")
        
        logger.info("End condition characteristics:")
        end_analysis = self.aero_adapter.analyze_aerodynamic_characteristics(
            self.aero_adapter.parse_aero_vector(end_condition)
        )
        logger.info(f"  Max CL: {end_analysis['max_lift_coefficient']:.3f}")
        logger.info(f"  Min CD: {end_analysis['min_drag_coefficient']:.3f}")
        
        # Create interpolated conditions
        num_steps = 5
        interpolated_conditions = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interp_condition = (1 - alpha) * start_condition + alpha * end_condition
            interpolated_conditions.append(interp_condition)
        
        # Generate synthetic point clouds for interpolated conditions
        interpolated_clouds = []
        for aero_cond in interpolated_conditions:
            point_cloud = self.processor.generate_starship_pointcloud(
                aero_cond, num_points=1024
            )
            interpolated_clouds.append(point_cloud)
        
        # Save interpolation sequence
        interp_dir = self.output_dir / "interpolation"
        interp_dir.mkdir(exist_ok=True)
        
        for i, (cloud, aero_cond) in enumerate(zip(interpolated_clouds, interpolated_conditions)):
            self.processor.save_point_cloud(
                cloud, 
                str(interp_dir / f"interpolated_{i:02d}.ply"),
                format="ply"
            )
        
        logger.info(f"Interpolation sequence saved to {interp_dir}")
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        logger.info("=== Point-E Starship Point Cloud Generation Demo ===")
        
        start_time = time.time()
        
        # 1. Setup environment
        if not self.setup_environment():
            logger.error("Environment setup failed. Continuing with limited functionality...")
        
        # 2. Create sample data
        synthetic_clouds, aero_conditions = self.create_sample_data(num_samples=8)
        
        # 3. Demonstrate aerodynamic adaptation
        self.demonstrate_aero_adaptation(aero_conditions)
        
        # 4. Create visualizations
        self.demonstrate_visualization(synthetic_clouds, aero_conditions)
        
        # 5. Try Point-E generation
        generated_clouds = self.demonstrate_point_e_generation(aero_conditions)
        
        # 6. Demonstrate evaluation
        self.demonstrate_evaluation(synthetic_clouds, aero_conditions, generated_clouds)
        
        # 7. Demonstrate interpolation
        self.demonstrate_interpolation(aero_conditions)
        
        # Summary
        end_time = time.time()
        logger.info(f"\n=== Demo Completed Successfully ===")
        logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Generated {len(synthetic_clouds)} synthetic point clouds")
        if generated_clouds:
            logger.info(f"Generated {len(generated_clouds)} Point-E point clouds")
        
        # Create summary report
        self._create_summary_report(synthetic_clouds, aero_conditions, generated_clouds)
    
    def _create_summary_report(
        self, 
        synthetic_clouds: List[np.ndarray], 
        aero_conditions: List[np.ndarray],
        generated_clouds: List[np.ndarray]
    ):
        """Create a summary report of the demo"""
        report_path = self.output_dir / "demo_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Point-E Starship Point Cloud Generation Demo Summary\n")
            f.write("=" * 55 + "\n\n")
            
            f.write(f"Generated {len(synthetic_clouds)} synthetic point clouds\n")
            f.write(f"Generated {len(generated_clouds)} Point-E point clouds\n")
            f.write(f"Processed {len(aero_conditions)} aerodynamic conditions\n\n")
            
            f.write("Output Files:\n")
            f.write(f"- Visualizations: {self.output_dir / 'visualizations'}\n")
            f.write(f"- Generated clouds: {self.output_dir / 'generated_clouds'}\n")
            f.write(f"- Interpolation: {self.output_dir / 'interpolation'}\n")
            f.write(f"- Evaluation results: {self.output_dir / 'evaluation_results.json'}\n\n")
            
            f.write("Key Features Demonstrated:\n")
            f.write("- Aerodynamic condition parsing and analysis\n")
            f.write("- Text description generation from aerodynamic data\n")
            f.write("- Synthetic Starship point cloud generation\n")
            f.write("- Point cloud visualization (matplotlib, plotly)\n")
            f.write("- Quality evaluation metrics\n")
            f.write("- Aerodynamic condition interpolation\n")
            if generated_clouds:
                f.write("- Point-E model integration\n")
            
            f.write("\nTo view results:\n")
            f.write("1. Open HTML files in visualizations/ for interactive plots\n")
            f.write("2. Load PLY files in visualization software (MeshLab, CloudCompare)\n")
            f.write("3. Check evaluation_results.json for quantitative metrics\n")
        
        logger.info(f"Demo summary saved to {report_path}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Point-E Starship Point Cloud Generation Demo")
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./demo_output',
        help='Output directory for demo results'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=8,
        help='Number of sample conditions to generate'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run demo
    demo = PointEStarshipDemo(output_dir=args.output_dir)
    demo.run_complete_demo()


if __name__ == "__main__":
    main()