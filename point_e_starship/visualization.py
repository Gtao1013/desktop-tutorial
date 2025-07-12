"""
Visualization Tools
Tools for visualizing point clouds and aerodynamic data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not available. Some visualization features will be limited.")

logger = logging.getLogger(__name__)


class PointCloudVisualizer:
    """Point cloud visualization utilities"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'default': '#1f77b4',
            'generated': '#ff7f0e', 
            'target': '#2ca02c',
            'comparison': ['#d62728', '#9467bd', '#8c564b']
        }
    
    def plot_point_cloud_matplotlib(
        self,
        point_cloud: np.ndarray,
        title: str = "Point Cloud",
        color_by: str = "height",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot point cloud using matplotlib
        
        Args:
            point_cloud: Point cloud array (N, 3) or (N, 6)
            title: Plot title
            color_by: Color scheme ('height', 'distance', 'rgb', 'uniform')
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        if point_cloud.shape[1] >= 3:
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
            
            # Determine colors
            if color_by == "height":
                colors = z
                colormap = 'viridis'
            elif color_by == "distance":
                colors = np.sqrt(x**2 + y**2 + z**2)
                colormap = 'plasma'
            elif color_by == "rgb" and point_cloud.shape[1] >= 6:
                colors = point_cloud[:, 3:6]
                colormap = None
            else:
                colors = self.colors['default']
                colormap = None
            
            # Create scatter plot
            if colormap:
                scatter = ax.scatter(x, y, z, c=colors, cmap=colormap, s=1, alpha=0.6)
                plt.colorbar(scatter, ax=ax, shrink=0.8)
            else:
                ax.scatter(x, y, z, c=colors, s=1, alpha=0.6)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
            
            # Equal aspect ratio
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def plot_point_cloud_plotly(
        self,
        point_cloud: np.ndarray,
        title: str = "Point Cloud",
        color_by: str = "height",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot point cloud using plotly (interactive)
        
        Args:
            point_cloud: Point cloud array (N, 3) or (N, 6)
            title: Plot title
            color_by: Color scheme
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        
        # Determine colors
        if color_by == "height":
            colors = z
            colorscale = 'Viridis'
        elif color_by == "distance":
            colors = np.sqrt(x**2 + y**2 + z**2)
            colorscale = 'Plasma'
        elif color_by == "rgb" and point_cloud.shape[1] >= 6:
            # Convert RGB to plotly format
            rgb_colors = point_cloud[:, 3:6]
            colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in rgb_colors]
            colorscale = None
        else:
            colors = self.colors['default']
            colorscale = None
        
        # Create 3D scatter plot
        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
                colorscale=colorscale,
                showscale=True if colorscale else False,
                opacity=0.7
            ),
            text=[f'Point {i}' for i in range(len(x))],
            name="Point Cloud"
        )
        
        fig = go.Figure(data=[scatter])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive visualization to {save_path}")
        
        return fig
    
    def plot_point_cloud_open3d(
        self,
        point_cloud: np.ndarray,
        title: str = "Point Cloud",
        save_path: Optional[str] = None
    ):
        """
        Visualize point cloud using Open3D
        
        Args:
            point_cloud: Point cloud array (N, 3) or (N, 6)
            title: Window title
            save_path: Path to save screenshot
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D not available, skipping visualization")
            return
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Set colors if available
        if point_cloud.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
        else:
            # Default color
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
        # Estimate normals for better visualization
        pcd.estimate_normals()
        
        # Visualize
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=800,
            height=600
        )
        
        if save_path:
            # Save as PLY file
            o3d.io.write_point_cloud(save_path, pcd)
            logger.info(f"Saved point cloud to {save_path}")
    
    def compare_point_clouds(
        self,
        point_clouds: List[np.ndarray],
        labels: List[str],
        title: str = "Point Cloud Comparison",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Compare multiple point clouds in one plot
        
        Args:
            point_clouds: List of point cloud arrays
            labels: Labels for each point cloud
            title: Plot title
            save_path: Path to save visualization
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for i, (pc, label) in enumerate(zip(point_clouds, labels)):
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
            
            # Use different colors for each point cloud
            color = self.colors['comparison'][i % len(self.colors['comparison'])]
            
            scatter = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.6
                ),
                name=label
            )
            
            fig.add_trace(scatter)
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved comparison visualization to {save_path}")
        
        return fig
    
    def plot_generation_animation(
        self,
        point_clouds: List[np.ndarray],
        title: str = "Point Cloud Generation Animation",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create animation showing point cloud generation progression
        
        Args:
            point_clouds: List of point clouds showing progression
            title: Animation title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure with animation
        """
        frames = []
        
        for i, pc in enumerate(point_clouds):
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
            
            frame = go.Frame(
                data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.7
                    ),
                    name=f"Step {i+1}"
                )],
                name=f"Step {i+1}"
            )
            frames.append(frame)
        
        # Initial frame
        initial_pc = point_clouds[0]
        x, y, z = initial_pc[:, 0], initial_pc[:, 1], initial_pc[:, 2]
        
        fig = go.Figure(
            data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=z,
                    colorscale='Viridis',
                    opacity=0.7
                )
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500}, 'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                    }
                ]
            }]
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved animation to {save_path}")
        
        return fig


class AerodynamicVisualizer:
    """Visualization tools for aerodynamic data"""
    
    def __init__(self):
        """Initialize aerodynamic visualizer"""
        self.coeff_names = ["CL", "CD", "CMy"]
        self.coeff_labels = {
            "CL": "Lift Coefficient",
            "CD": "Drag Coefficient", 
            "CMy": "Moment Coefficient"
        }
    
    def plot_aero_coefficients(
        self,
        aero_conditions: np.ndarray,
        title: str = "Aerodynamic Coefficients",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot aerodynamic coefficients vs angle of attack
        
        Args:
            aero_conditions: 21-dimensional aerodynamic conditions
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from .aero_condition_adapter import AeroConditionAdapter
        
        adapter = AeroConditionAdapter()
        aero_data = adapter.parse_aero_vector(aero_conditions)
        
        angles = [0, 5, 10, 15, 20, 25, 30]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, coeff in enumerate(self.coeff_names):
            values = [aero_data[f"{angle}_degrees"][coeff] for angle in angles]
            
            axes[i].plot(angles, values, 'o-', linewidth=2, markersize=6)
            axes[i].set_xlabel('Angle of Attack (degrees)')
            axes[i].set_ylabel(self.coeff_labels[coeff])
            axes[i].set_title(f'{self.coeff_labels[coeff]} vs AoA')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved aerodynamic plot to {save_path}")
        
        return fig
    
    def plot_polar_diagram(
        self,
        aero_conditions: np.ndarray,
        title: str = "Aerodynamic Polar Diagram",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot CL vs CD polar diagram
        
        Args:
            aero_conditions: 21-dimensional aerodynamic conditions
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from .aero_condition_adapter import AeroConditionAdapter
        
        adapter = AeroConditionAdapter()
        aero_data = adapter.parse_aero_vector(aero_conditions)
        
        angles = [0, 5, 10, 15, 20, 25, 30]
        cl_values = [aero_data[f"{angle}_degrees"]["CL"] for angle in angles]
        cd_values = [aero_data[f"{angle}_degrees"]["CD"] for angle in angles]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot polar curve
        ax.plot(cd_values, cl_values, 'o-', linewidth=2, markersize=8, label='Polar Curve')
        
        # Annotate points with angles
        for i, angle in enumerate(angles):
            ax.annotate(f'{angle}°', (cd_values[i], cl_values[i]), 
                       xytext=(10, 10), textcoords='offset points')
        
        # Add efficiency lines (L/D ratios)
        cd_range = np.linspace(0, max(cd_values), 100)
        for ld_ratio in [5, 10, 15, 20]:
            cl_line = ld_ratio * cd_range
            ax.plot(cd_range, cl_line, '--', alpha=0.5, label=f'L/D = {ld_ratio}')
        
        ax.set_xlabel('Drag Coefficient (CD)')
        ax.set_ylabel('Lift Coefficient (CL)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved polar diagram to {save_path}")
        
        return fig
    
    def plot_aero_surface(
        self,
        aero_conditions_list: List[np.ndarray],
        parameter_name: str = "Design Parameter",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot 3D surface of aerodynamic coefficients
        
        Args:
            aero_conditions_list: List of aerodynamic conditions
            parameter_name: Name of the varied parameter
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        from .aero_condition_adapter import AeroConditionAdapter
        
        adapter = AeroConditionAdapter()
        angles = [0, 5, 10, 15, 20, 25, 30]
        
        # Prepare data
        parameter_values = list(range(len(aero_conditions_list)))
        angle_grid, param_grid = np.meshgrid(angles, parameter_values)
        
        # Extract coefficients
        cl_surface = np.zeros_like(angle_grid, dtype=float)
        cd_surface = np.zeros_like(angle_grid, dtype=float)
        
        for i, aero_conditions in enumerate(aero_conditions_list):
            aero_data = adapter.parse_aero_vector(aero_conditions)
            for j, angle in enumerate(angles):
                cl_surface[i, j] = aero_data[f"{angle}_degrees"]["CL"]
                cd_surface[i, j] = aero_data[f"{angle}_degrees"]["CD"]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Lift Coefficient', 'Drag Coefficient'],
            specs=[[{'type': 'surface'}, {'type': 'surface'}]]
        )
        
        # Add CL surface
        fig.add_trace(
            go.Surface(
                x=angle_grid,
                y=param_grid,
                z=cl_surface,
                colorscale='Viridis',
                name='CL'
            ),
            row=1, col=1
        )
        
        # Add CD surface
        fig.add_trace(
            go.Surface(
                x=angle_grid,
                y=param_grid,
                z=cd_surface,
                colorscale='Plasma',
                name='CD'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Aerodynamic Coefficient Surfaces",
            scene=dict(
                xaxis_title='Angle of Attack (degrees)',
                yaxis_title=parameter_name,
                zaxis_title='CL'
            ),
            scene2=dict(
                xaxis_title='Angle of Attack (degrees)',
                yaxis_title=parameter_name,
                zaxis_title='CD'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved aerodynamic surface plot to {save_path}")
        
        return fig


def create_visualization_gallery(
    point_clouds: List[np.ndarray],
    aero_conditions: List[np.ndarray],
    output_dir: str = "./visualizations"
):
    """
    Create a gallery of visualizations
    
    Args:
        point_clouds: List of point clouds to visualize
        aero_conditions: Corresponding aerodynamic conditions
        output_dir: Output directory for visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pc_viz = PointCloudVisualizer()
    aero_viz = AerodynamicVisualizer()
    
    logger.info(f"Creating visualization gallery with {len(point_clouds)} point clouds")
    
    # Individual point cloud visualizations
    for i, (pc, aero) in enumerate(zip(point_clouds, aero_conditions)):
        # Point cloud plots
        fig_mpl = pc_viz.plot_point_cloud_matplotlib(
            pc, 
            title=f"Point Cloud {i+1}",
            save_path=output_dir / f"pointcloud_{i+1}_matplotlib.png"
        )
        plt.close(fig_mpl)
        
        fig_plotly = pc_viz.plot_point_cloud_plotly(
            pc,
            title=f"Point Cloud {i+1}",
            save_path=output_dir / f"pointcloud_{i+1}_interactive.html"
        )
        
        # Aerodynamic plots
        fig_aero = aero_viz.plot_aero_coefficients(
            aero,
            title=f"Aerodynamic Coefficients {i+1}",
            save_path=output_dir / f"aero_coeffs_{i+1}.png"
        )
        plt.close(fig_aero)
        
        fig_polar = aero_viz.plot_polar_diagram(
            aero,
            title=f"Polar Diagram {i+1}",
            save_path=output_dir / f"polar_diagram_{i+1}.png"
        )
        plt.close(fig_polar)
        
        if i >= 4:  # Limit to first 5 for demo
            break
    
    # Comparison plots
    if len(point_clouds) > 1:
        compare_pcs = point_clouds[:min(3, len(point_clouds))]
        compare_labels = [f"Cloud {i+1}" for i in range(len(compare_pcs))]
        
        fig_compare = pc_viz.compare_point_clouds(
            compare_pcs,
            compare_labels,
            title="Point Cloud Comparison",
            save_path=output_dir / "pointcloud_comparison.html"
        )
        
        # Aerodynamic surface plot
        fig_surface = aero_viz.plot_aero_surface(
            aero_conditions[:min(10, len(aero_conditions))],
            parameter_name="Sample Index",
            save_path=output_dir / "aero_surface.html"
        )
    
    logger.info(f"Visualization gallery created in {output_dir}")


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization tools...")
    
    # Create sample data
    from .data_processor import PointCloudProcessor
    from .aero_condition_adapter import create_sample_aero_conditions
    
    processor = PointCloudProcessor()
    
    # Generate some sample point clouds
    sample_aero = create_sample_aero_conditions()
    point_cloud = processor.generate_starship_pointcloud(sample_aero, num_points=1024)
    
    # Test visualizations
    pc_viz = PointCloudVisualizer()
    aero_viz = AerodynamicVisualizer()
    
    # Test matplotlib visualization
    fig1 = pc_viz.plot_point_cloud_matplotlib(
        point_cloud,
        title="Test Point Cloud",
        save_path="/tmp/test_pointcloud.png"
    )
    plt.close(fig1)
    
    # Test plotly visualization
    fig2 = pc_viz.plot_point_cloud_plotly(
        point_cloud,
        title="Test Point Cloud Interactive",
        save_path="/tmp/test_pointcloud.html"
    )
    
    # Test aerodynamic visualization
    fig3 = aero_viz.plot_aero_coefficients(
        sample_aero,
        title="Test Aerodynamic Coefficients",
        save_path="/tmp/test_aero.png"
    )
    plt.close(fig3)
    
    print("Visualization tests completed!")