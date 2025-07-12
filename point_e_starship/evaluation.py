"""
Evaluation Tools
Metrics and tools for evaluating point cloud generation quality
"""

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from sklearn.metrics import pairwise_distances
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import json

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

logger = logging.getLogger(__name__)


class PointCloudEvaluator:
    """Evaluation metrics for point cloud generation quality"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics = {}
        
    def chamfer_distance(self, pc1: np.ndarray, pc2: np.ndarray) -> float:
        """
        Compute Chamfer Distance between two point clouds
        
        Args:
            pc1: First point cloud (N, 3)
            pc2: Second point cloud (M, 3)
            
        Returns:
            Chamfer distance
        """
        # Only use coordinates (first 3 dimensions)
        points1 = pc1[:, :3]
        points2 = pc2[:, :3]
        
        # Build KD-trees for efficient nearest neighbor search
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Find nearest neighbors
        distances1, _ = tree1.query(points2)
        distances2, _ = tree2.query(points1)
        
        # Chamfer distance is the mean of squared distances
        chamfer_dist = np.mean(distances1**2) + np.mean(distances2**2)
        
        return float(chamfer_dist)
    
    def hausdorff_distance(self, pc1: np.ndarray, pc2: np.ndarray) -> float:
        """
        Compute Hausdorff Distance between two point clouds
        
        Args:
            pc1: First point cloud (N, 3)
            pc2: Second point cloud (M, 3)
            
        Returns:
            Hausdorff distance
        """
        points1 = pc1[:, :3]
        points2 = pc2[:, :3]
        
        # Compute directed Hausdorff distances
        h1 = directed_hausdorff(points1, points2)[0]
        h2 = directed_hausdorff(points2, points1)[0]
        
        # Hausdorff distance is the maximum of the two directed distances
        return float(max(h1, h2))
    
    def earth_movers_distance(self, pc1: np.ndarray, pc2: np.ndarray) -> float:
        """
        Compute Earth Mover's Distance (Wasserstein distance) approximation
        
        Args:
            pc1: First point cloud (N, 3)
            pc2: Second point cloud (M, 3)
            
        Returns:
            EMD approximation
        """
        points1 = pc1[:, :3]
        points2 = pc2[:, :3]
        
        # Simple approximation using mean distances
        # For exact EMD, you would need optimal transport solvers
        distances = pairwise_distances(points1, points2)
        min_distances1 = np.min(distances, axis=1)
        min_distances2 = np.min(distances, axis=0)
        
        emd_approx = (np.mean(min_distances1) + np.mean(min_distances2)) / 2
        
        return float(emd_approx)
    
    def point_cloud_density_distribution(self, point_cloud: np.ndarray, k: int = 10) -> Dict[str, float]:
        """
        Analyze point cloud density distribution
        
        Args:
            point_cloud: Point cloud (N, 3)
            k: Number of nearest neighbors to consider
            
        Returns:
            Dictionary of density statistics
        """
        points = point_cloud[:, :3]
        tree = cKDTree(points)
        
        # Find k nearest neighbors for each point
        distances, _ = tree.query(points, k=k+1)  # +1 because first is the point itself
        
        # Use mean distance to k-th nearest neighbor as density measure
        densities = np.mean(distances[:, 1:], axis=1)  # Exclude the point itself
        
        return {
            'mean_density': float(np.mean(densities)),
            'std_density': float(np.std(densities)),
            'min_density': float(np.min(densities)),
            'max_density': float(np.max(densities)),
            'density_uniformity': float(1.0 / (1.0 + np.std(densities)))  # Higher is more uniform
        }
    
    def structural_coherence(self, point_cloud: np.ndarray) -> Dict[str, float]:
        """
        Measure structural coherence of point cloud
        
        Args:
            point_cloud: Point cloud (N, 3)
            
        Returns:
            Dictionary of structural metrics
        """
        points = point_cloud[:, :3]
        
        # Compute centroid
        centroid = np.mean(points, axis=0)
        
        # Compute distances from centroid
        distances_from_centroid = np.linalg.norm(points - centroid, axis=1)
        
        # Principal component analysis for shape analysis
        centered_points = points - centroid
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
        
        # Shape descriptors
        compactness = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        elongation = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        flatness = eigenvalues[2] / eigenvalues[1] if eigenvalues[1] > 0 else 0
        
        return {
            'compactness': float(compactness),
            'elongation': float(elongation),
            'flatness': float(flatness),
            'spread': float(np.std(distances_from_centroid)),
            'volume_ratio': float(np.prod(eigenvalues) / (np.max(eigenvalues)**3))
        }
    
    def coverage_uniformity(self, point_cloud: np.ndarray, grid_resolution: int = 20) -> float:
        """
        Measure how uniformly points cover the space
        
        Args:
            point_cloud: Point cloud (N, 3)
            grid_resolution: Resolution of the grid for coverage analysis
            
        Returns:
            Coverage uniformity score (0-1, higher is better)
        """
        points = point_cloud[:, :3]
        
        # Create bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Create 3D grid
        grid_edges = [
            np.linspace(min_coords[i], max_coords[i], grid_resolution + 1)
            for i in range(3)
        ]
        
        # Count points in each grid cell
        grid_counts = np.zeros((grid_resolution, grid_resolution, grid_resolution))
        
        for point in points:
            # Find grid cell indices
            indices = []
            for i in range(3):
                idx = np.searchsorted(grid_edges[i], point[i]) - 1
                idx = np.clip(idx, 0, grid_resolution - 1)
                indices.append(idx)
            
            grid_counts[indices[0], indices[1], indices[2]] += 1
        
        # Compute uniformity as inverse of variance
        non_empty_cells = grid_counts[grid_counts > 0]
        if len(non_empty_cells) == 0:
            return 0.0
        
        uniformity = 1.0 / (1.0 + np.var(non_empty_cells))
        return float(uniformity)
    
    def geometric_quality_score(self, point_cloud: np.ndarray) -> Dict[str, float]:
        """
        Compute overall geometric quality score
        
        Args:
            point_cloud: Point cloud (N, 3)
            
        Returns:
            Dictionary of quality metrics
        """
        # Individual metrics
        density_stats = self.point_cloud_density_distribution(point_cloud)
        structure_stats = self.structural_coherence(point_cloud)
        coverage_score = self.coverage_uniformity(point_cloud)
        
        # Composite quality score
        quality_components = {
            'density_uniformity': density_stats['density_uniformity'],
            'structural_compactness': structure_stats['compactness'],
            'coverage_uniformity': coverage_score
        }
        
        # Weighted average
        weights = {'density_uniformity': 0.3, 'structural_compactness': 0.4, 'coverage_uniformity': 0.3}
        overall_quality = sum(quality_components[k] * weights[k] for k in weights)
        
        return {
            'overall_quality': float(overall_quality),
            **quality_components,
            **density_stats,
            **structure_stats
        }
    
    def evaluate_generation_batch(
        self,
        generated_clouds: List[np.ndarray],
        reference_clouds: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Union[float, Dict]]:
        """
        Evaluate a batch of generated point clouds
        
        Args:
            generated_clouds: List of generated point clouds
            reference_clouds: Optional reference point clouds for comparison
            
        Returns:
            Dictionary of evaluation results
        """
        results = {
            'num_clouds': len(generated_clouds),
            'individual_quality': [],
            'average_quality': {},
            'comparison_metrics': {}
        }
        
        # Evaluate individual clouds
        quality_scores = []
        for i, cloud in enumerate(generated_clouds):
            quality = self.geometric_quality_score(cloud)
            quality_scores.append(quality)
            results['individual_quality'].append(quality)
        
        # Compute average quality metrics
        if quality_scores:
            avg_quality = {}
            for key in quality_scores[0]:
                values = [q[key] for q in quality_scores]
                avg_quality[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            results['average_quality'] = avg_quality
        
        # Compare with reference clouds if provided
        if reference_clouds is not None:
            results['comparison_metrics'] = self._compare_cloud_batches(
                generated_clouds, reference_clouds
            )
        
        return results
    
    def _compare_cloud_batches(
        self,
        generated_clouds: List[np.ndarray],
        reference_clouds: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compare generated clouds with reference clouds"""
        chamfer_distances = []
        hausdorff_distances = []
        emd_distances = []
        
        # Compare each generated cloud with the most similar reference cloud
        for gen_cloud in generated_clouds:
            min_chamfer = float('inf')
            min_hausdorff = float('inf')
            min_emd = float('inf')
            
            for ref_cloud in reference_clouds:
                try:
                    chamfer = self.chamfer_distance(gen_cloud, ref_cloud)
                    hausdorff = self.hausdorff_distance(gen_cloud, ref_cloud)
                    emd = self.earth_movers_distance(gen_cloud, ref_cloud)
                    
                    min_chamfer = min(min_chamfer, chamfer)
                    min_hausdorff = min(min_hausdorff, hausdorff)
                    min_emd = min(min_emd, emd)
                except Exception as e:
                    logger.warning(f"Failed to compute distance: {e}")
                    continue
            
            if min_chamfer != float('inf'):
                chamfer_distances.append(min_chamfer)
                hausdorff_distances.append(min_hausdorff)
                emd_distances.append(min_emd)
        
        # Compute statistics
        if chamfer_distances:
            return {
                'chamfer_distance': {
                    'mean': float(np.mean(chamfer_distances)),
                    'std': float(np.std(chamfer_distances)),
                    'min': float(np.min(chamfer_distances)),
                    'max': float(np.max(chamfer_distances))
                },
                'hausdorff_distance': {
                    'mean': float(np.mean(hausdorff_distances)),
                    'std': float(np.std(hausdorff_distances)),
                    'min': float(np.min(hausdorff_distances)),
                    'max': float(np.max(hausdorff_distances))
                },
                'emd_distance': {
                    'mean': float(np.mean(emd_distances)),
                    'std': float(np.std(emd_distances)),
                    'min': float(np.min(emd_distances)),
                    'max': float(np.max(emd_distances))
                }
            }
        else:
            return {}


class AerodynamicConsistencyEvaluator:
    """Evaluate consistency between aerodynamic conditions and generated point clouds"""
    
    def __init__(self):
        """Initialize aerodynamic evaluator"""
        from .aero_condition_adapter import AeroConditionAdapter
        self.aero_adapter = AeroConditionAdapter()
    
    def analyze_aero_shape_correlation(
        self,
        point_clouds: List[np.ndarray],
        aero_conditions: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Analyze correlation between aerodynamic conditions and point cloud shapes
        
        Args:
            point_clouds: List of generated point clouds
            aero_conditions: Corresponding aerodynamic conditions
            
        Returns:
            Dictionary of correlation metrics
        """
        if len(point_clouds) != len(aero_conditions):
            raise ValueError("Number of point clouds and aerodynamic conditions must match")
        
        # Extract aerodynamic features
        aero_features = []
        for aero_cond in aero_conditions:
            aero_data = self.aero_adapter.parse_aero_vector(aero_cond)
            analysis = self.aero_adapter.analyze_aerodynamic_characteristics(aero_data)
            
            features = [
                analysis['max_lift_coefficient'],
                analysis['min_drag_coefficient'],
                analysis['max_ld_ratio'],
                analysis['stall_angle'] or 30,  # Default if no stall
                1.0 if analysis['moment_stability'] == 'stable' else 0.0
            ]
            aero_features.append(features)
        
        aero_features = np.array(aero_features)
        
        # Extract geometric features from point clouds
        pc_evaluator = PointCloudEvaluator()
        geometric_features = []
        
        for pc in point_clouds:
            quality = pc_evaluator.geometric_quality_score(pc)
            structure = pc_evaluator.structural_coherence(pc)
            
            features = [
                quality['overall_quality'],
                structure['compactness'],
                structure['elongation'],
                structure['flatness'],
                quality['coverage_uniformity']
            ]
            geometric_features.append(features)
        
        geometric_features = np.array(geometric_features)
        
        # Compute correlations
        correlations = {}
        aero_names = ['max_CL', 'min_CD', 'max_LD', 'stall_angle', 'stability']
        geom_names = ['quality', 'compactness', 'elongation', 'flatness', 'coverage']
        
        for i, aero_name in enumerate(aero_names):
            for j, geom_name in enumerate(geom_names):
                corr = np.corrcoef(aero_features[:, i], geometric_features[:, j])[0, 1]
                correlations[f'{aero_name}_vs_{geom_name}'] = float(corr) if not np.isnan(corr) else 0.0
        
        # Overall consistency score
        abs_correlations = [abs(v) for v in correlations.values()]
        consistency_score = np.mean(abs_correlations)
        
        return {
            'consistency_score': float(consistency_score),
            'correlations': correlations,
            'aero_feature_stats': {
                'mean': aero_features.mean(axis=0).tolist(),
                'std': aero_features.std(axis=0).tolist()
            },
            'geometric_feature_stats': {
                'mean': geometric_features.mean(axis=0).tolist(),
                'std': geometric_features.std(axis=0).tolist()
            }
        }
    
    def evaluate_design_space_coverage(
        self,
        aero_conditions: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate how well the generated samples cover the aerodynamic design space
        
        Args:
            aero_conditions: List of aerodynamic conditions
            
        Returns:
            Dictionary of coverage metrics
        """
        # Convert to matrix
        aero_matrix = np.array(aero_conditions)
        
        # Analyze coverage for each coefficient type
        coverage_metrics = {}
        
        for coeff_idx, coeff_name in enumerate(['CL', 'CD', 'CMy']):
            # Extract values for this coefficient across all angles
            coeff_values = aero_matrix[:, coeff_idx::3]  # Every 3rd element starting from coeff_idx
            
            # Compute range coverage
            min_vals = np.min(coeff_values, axis=0)
            max_vals = np.max(coeff_values, axis=0)
            ranges = max_vals - min_vals
            
            coverage_metrics[f'{coeff_name}_range_coverage'] = float(np.mean(ranges))
            coverage_metrics[f'{coeff_name}_min_range'] = float(np.min(ranges))
            coverage_metrics[f'{coeff_name}_max_range'] = float(np.max(ranges))
        
        # Overall design space coverage
        # Compute volume of the convex hull (simplified measure)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(aero_matrix)
            coverage_metrics['design_space_volume'] = float(hull.volume)
        except Exception:
            coverage_metrics['design_space_volume'] = 0.0
        
        return coverage_metrics


def comprehensive_evaluation(
    generated_clouds: List[np.ndarray],
    aero_conditions: List[np.ndarray],
    reference_clouds: Optional[List[np.ndarray]] = None,
    output_path: Optional[str] = None
) -> Dict[str, any]:
    """
    Perform comprehensive evaluation of generated point clouds
    
    Args:
        generated_clouds: List of generated point clouds
        aero_conditions: Corresponding aerodynamic conditions
        reference_clouds: Optional reference point clouds
        output_path: Path to save evaluation results
        
    Returns:
        Comprehensive evaluation results
    """
    pc_evaluator = PointCloudEvaluator()
    aero_evaluator = AerodynamicConsistencyEvaluator()
    
    logger.info(f"Starting comprehensive evaluation of {len(generated_clouds)} point clouds")
    
    # Point cloud quality evaluation
    pc_results = pc_evaluator.evaluate_generation_batch(
        generated_clouds, reference_clouds
    )
    
    # Aerodynamic consistency evaluation
    aero_results = aero_evaluator.analyze_aero_shape_correlation(
        generated_clouds, aero_conditions
    )
    
    # Design space coverage
    coverage_results = aero_evaluator.evaluate_design_space_coverage(
        aero_conditions
    )
    
    # Combine results
    comprehensive_results = {
        'summary': {
            'num_generated_clouds': len(generated_clouds),
            'num_reference_clouds': len(reference_clouds) if reference_clouds else 0,
            'overall_quality_score': pc_results.get('average_quality', {}).get('overall_quality', {}).get('mean', 0.0),
            'aerodynamic_consistency': aero_results.get('consistency_score', 0.0),
            'design_space_coverage': coverage_results
        },
        'point_cloud_quality': pc_results,
        'aerodynamic_consistency': aero_results,
        'design_space_analysis': coverage_results,
        'timestamp': str(np.datetime64('now'))
    }
    
    # Save results if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    return comprehensive_results


if __name__ == "__main__":
    # Test evaluation tools
    print("Testing evaluation tools...")
    
    # Create synthetic test data
    from .data_processor import PointCloudProcessor
    from .aero_condition_adapter import create_sample_aero_conditions
    
    processor = PointCloudProcessor()
    
    # Generate test point clouds
    test_aero_conditions = []
    test_point_clouds = []
    
    for i in range(5):
        aero_cond = create_sample_aero_conditions()
        # Add some variation
        aero_cond = aero_cond * (1 + 0.1 * np.random.randn(21))
        
        point_cloud = processor.generate_starship_pointcloud(aero_cond, num_points=1024)
        
        test_aero_conditions.append(aero_cond)
        test_point_clouds.append(point_cloud)
    
    # Test evaluations
    evaluator = PointCloudEvaluator()
    
    # Test individual metrics
    print("Testing Chamfer distance...")
    chamfer_dist = evaluator.chamfer_distance(test_point_clouds[0], test_point_clouds[1])
    print(f"Chamfer distance: {chamfer_dist:.6f}")
    
    print("Testing geometric quality...")
    quality = evaluator.geometric_quality_score(test_point_clouds[0])
    print(f"Quality score: {quality['overall_quality']:.3f}")
    
    # Test comprehensive evaluation
    print("Testing comprehensive evaluation...")
    results = comprehensive_evaluation(
        generated_clouds=test_point_clouds,
        aero_conditions=test_aero_conditions,
        output_path="/tmp/evaluation_results.json"
    )
    
    print(f"Overall quality: {results['summary']['overall_quality_score']:.3f}")
    print(f"Aerodynamic consistency: {results['summary']['aerodynamic_consistency']:.3f}")
    
    print("Evaluation tests completed!")