"""
Data Processing Module
Handles point cloud data loading, preprocessing, and batch preparation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from scipy.spatial.transform import Rotation
import trimesh

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for when torch is not available
    class Dataset:
        pass
    class DataLoader:
        pass
import h5py
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from scipy.spatial.transform import Rotation
import trimesh

logger = logging.getLogger(__name__)


class PointCloudDataset(Dataset):
    """Dataset class for point cloud data with aerodynamic conditions"""
    
    def __init__(
        self, 
        data_path: str,
        aero_conditions: np.ndarray,
        num_points: int = 1024,
        augment: bool = True,
        normalize: bool = True
    ):
        """
        Initialize point cloud dataset
        
        Args:
            data_path: Path to point cloud data
            aero_conditions: Aerodynamic conditions (N, 21)
            num_points: Number of points per cloud
            augment: Whether to apply data augmentation
            normalize: Whether to normalize point clouds
        """
        self.data_path = Path(data_path)
        self.aero_conditions = aero_conditions
        self.num_points = num_points
        self.augment = augment
        self.normalize = normalize
        
        # Load point cloud data
        self.point_clouds = self._load_point_clouds()
        
        # Validate data consistency
        assert len(self.point_clouds) == len(self.aero_conditions), \
            f"Point clouds ({len(self.point_clouds)}) and aero conditions ({len(self.aero_conditions)}) count mismatch"
    
    def _load_point_clouds(self) -> List[np.ndarray]:
        """Load point cloud data from files"""
        point_clouds = []
        
        if self.data_path.suffix == '.h5':
            # Load from HDF5 file
            with h5py.File(self.data_path, 'r') as f:
                for i in range(len(f.keys())):
                    cloud_key = f"cloud_{i:06d}"
                    if cloud_key in f:
                        point_clouds.append(f[cloud_key][:])
                    else:
                        break
        elif self.data_path.is_dir():
            # Load from directory of files
            cloud_files = sorted(list(self.data_path.glob("*.npy")))
            for cloud_file in cloud_files:
                point_clouds.append(np.load(cloud_file))
        else:
            raise ValueError(f"Unsupported data path: {self.data_path}")
        
        logger.info(f"Loaded {len(point_clouds)} point clouds from {self.data_path}")
        return point_clouds
    
    def __len__(self) -> int:
        return len(self.point_clouds)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Get a single data sample"""
        point_cloud = self.point_clouds[idx].copy()
        aero_condition = self.aero_conditions[idx].copy()
        
        # Resample to target number of points
        point_cloud = self._resample_points(point_cloud)
        
        # Apply augmentation
        if self.augment:
            point_cloud = self._augment_point_cloud(point_cloud)
        
        # Normalize
        if self.normalize:
            point_cloud = self._normalize_point_cloud(point_cloud)
        
        return {
            'point_cloud': torch.tensor(point_cloud, dtype=torch.float32) if TORCH_AVAILABLE else point_cloud.astype(np.float32),
            'aero_condition': torch.tensor(aero_condition, dtype=torch.float32) if TORCH_AVAILABLE else aero_condition.astype(np.float32)
        }
    
    def _resample_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """Resample point cloud to target number of points"""
        if len(point_cloud) == self.num_points:
            return point_cloud
        elif len(point_cloud) > self.num_points:
            # Random sampling
            indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
            return point_cloud[indices]
        else:
            # Upsampling with repetition
            indices = np.random.choice(len(point_cloud), self.num_points, replace=True)
            return point_cloud[indices]
    
    def _augment_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply data augmentation to point cloud"""
        # Random rotation
        if np.random.random() < 0.5:
            rotation = Rotation.random().as_matrix()
            point_cloud[:, :3] = point_cloud[:, :3] @ rotation.T
        
        # Random scaling
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            point_cloud[:, :3] *= scale
        
        # Random translation
        if np.random.random() < 0.3:
            translation = np.random.uniform(-0.1, 0.1, 3)
            point_cloud[:, :3] += translation
        
        # Random jitter
        if np.random.random() < 0.5:
            jitter = np.random.normal(0, 0.01, point_cloud[:, :3].shape)
            point_cloud[:, :3] += jitter
        
        return point_cloud
    
    def _normalize_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Normalize point cloud to unit sphere"""
        # Center at origin
        centroid = np.mean(point_cloud[:, :3], axis=0)
        point_cloud[:, :3] -= centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(point_cloud[:, :3], axis=1))
        if max_dist > 0:
            point_cloud[:, :3] /= max_dist
        
        return point_cloud


class PointCloudProcessor:
    """Point cloud processing utilities"""
    
    def __init__(self):
        self.supported_formats = ['.ply', '.obj', '.stl', '.off', '.xyz', '.pcd']
    
    def load_mesh_as_pointcloud(self, mesh_path: str, num_points: int = 1024) -> np.ndarray:
        """
        Load 3D mesh and convert to point cloud
        
        Args:
            mesh_path: Path to mesh file
            num_points: Number of points to sample
            
        Returns:
            Point cloud array of shape (num_points, 6) with XYZ and RGB
        """
        mesh_path = Path(mesh_path)
        
        if mesh_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {mesh_path.suffix}")
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Sample points on surface
        points, face_indices = mesh.sample(num_points, return_index=True)
        
        # Get colors if available
        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            colors = mesh.visual.face_colors[face_indices][:, :3] / 255.0
        else:
            # Default gray color
            colors = np.full((num_points, 3), 0.5)
        
        # Combine points and colors
        point_cloud = np.concatenate([points, colors], axis=1)
        
        return point_cloud
    
    def generate_starship_pointcloud(self, aero_conditions: np.ndarray, num_points: int = 1024) -> np.ndarray:
        """
        Generate synthetic Starship-like point cloud based on aerodynamic conditions
        
        Args:
            aero_conditions: 21-dimensional aerodynamic conditions
            num_points: Number of points to generate
            
        Returns:
            Synthetic point cloud
        """
        # Parse aerodynamic conditions to extract key parameters
        cl_values = [aero_conditions[i*3] for i in range(7)]  # Lift coefficients
        cd_values = [aero_conditions[i*3+1] for i in range(7)]  # Drag coefficients
        
        # Base Starship-like geometry parameters
        height = 50.0  # meters
        diameter = 9.0  # meters
        nose_ratio = 0.15  # Nose cone length ratio
        
        # Modify geometry based on aerodynamic characteristics
        max_cl = max(cl_values)
        avg_cd = np.mean(cd_values)
        
        # Adjust proportions based on performance
        if max_cl > 1.0:  # High lift -> more slender
            diameter *= 0.9
            height *= 1.1
        if avg_cd < 0.4:  # Low drag -> more streamlined
            nose_ratio *= 1.2
        
        # Generate body points
        points = []
        
        # Cylindrical body
        body_height = height * (1 - nose_ratio)
        for i in range(int(num_points * 0.6)):
            z = np.random.uniform(0, body_height)
            theta = np.random.uniform(0, 2*np.pi)
            r = diameter/2 * (1 + 0.1 * np.random.normal())  # Slight variation
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        
        # Nose cone
        nose_height = height * nose_ratio
        for i in range(int(num_points * 0.3)):
            z = np.random.uniform(body_height, height)
            theta = np.random.uniform(0, 2*np.pi)
            
            # Cone radius decreases linearly
            cone_progress = (z - body_height) / nose_height
            r = diameter/2 * (1 - cone_progress) * (1 + 0.1 * np.random.normal())
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        
        # Fins/control surfaces (simplified)
        fin_positions = [height * 0.2, height * 0.4, height * 0.6]
        for fin_z in fin_positions:
            for i in range(int(num_points * 0.03)):
                theta = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])  # 4 fins
                r = np.random.uniform(diameter/2, diameter/2 + 2.0)
                z = fin_z + np.random.uniform(-1.0, 1.0)
                
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append([x, y, z])
        
        # Convert to numpy array
        points = np.array(points)
        
        # Add some random internal points
        remaining = num_points - len(points)
        if remaining > 0:
            for i in range(remaining):
                z = np.random.uniform(0, height)
                theta = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, diameter/2)
                
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points = np.vstack([points, [x, y, z]])
        
        # Trim to exact number
        points = points[:num_points]
        
        # Add colors based on aerodynamic properties
        colors = np.zeros((num_points, 3))
        
        # Color coding: Red for high stress areas, Blue for low stress
        for i, (x, y, z) in enumerate(points):
            # Distance from centerline
            r = np.sqrt(x**2 + y**2)
            
            # High drag -> more red
            red_intensity = min(1.0, avg_cd / 0.8)
            # High lift -> more blue
            blue_intensity = min(1.0, max_cl / 1.5)
            # Green for moderate areas
            green_intensity = 0.3 + 0.4 * (1 - r / (diameter/2))
            
            colors[i] = [red_intensity, green_intensity, blue_intensity]
        
        # Combine geometry and colors
        point_cloud = np.concatenate([points, colors], axis=1)
        
        return point_cloud
    
    def save_point_cloud(self, point_cloud: np.ndarray, save_path: str, format: str = 'npy'):
        """
        Save point cloud to file
        
        Args:
            point_cloud: Point cloud array
            save_path: Output file path
            format: Output format ('npy', 'ply', 'xyz')
        """
        save_path = Path(save_path)
        
        if format == 'npy':
            np.save(save_path, point_cloud)
        elif format == 'ply':
            self._save_ply(point_cloud, save_path)
        elif format == 'xyz':
            self._save_xyz(point_cloud, save_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_ply(self, point_cloud: np.ndarray, save_path: Path):
        """Save point cloud in PLY format"""
        with open(save_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if point_cloud.shape[1] >= 6:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for point in point_cloud:
                if point_cloud.shape[1] >= 6:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                           f"{int(point[3]*255)} {int(point[4]*255)} {int(point[5]*255)}\n")
                else:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    def _save_xyz(self, point_cloud: np.ndarray, save_path: Path):
        """Save point cloud in XYZ format"""
        with open(save_path, 'w') as f:
            for point in point_cloud:
                if point_cloud.shape[1] >= 6:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                           f"{point[3]:.3f} {point[4]:.3f} {point[5]:.3f}\n")
                else:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    def create_synthetic_dataset(
        self, 
        num_samples: int,
        output_dir: str,
        num_points: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic dataset of Starship point clouds with aerodynamic conditions
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save data
            num_points: Points per cloud
            
        Returns:
            Tuple of (point_clouds, aero_conditions)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        point_clouds = []
        aero_conditions = []
        
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        for i in range(num_samples):
            # Generate random but realistic aerodynamic conditions
            aero_condition = self._generate_random_aero_conditions()
            
            # Generate corresponding point cloud
            point_cloud = self.generate_starship_pointcloud(aero_condition, num_points)
            
            # Save individual files
            self.save_point_cloud(
                point_cloud, 
                output_dir / f"cloud_{i:06d}.npy"
            )
            
            point_clouds.append(point_cloud)
            aero_conditions.append(aero_condition)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        # Save as HDF5 for efficient loading
        point_clouds = np.array(point_clouds)
        aero_conditions = np.array(aero_conditions)
        
        with h5py.File(output_dir / "dataset.h5", 'w') as f:
            f.create_dataset("point_clouds", data=point_clouds)
            f.create_dataset("aero_conditions", data=aero_conditions)
        
        # Save metadata
        metadata = {
            "num_samples": num_samples,
            "num_points": num_points,
            "point_cloud_shape": point_clouds.shape,
            "aero_conditions_shape": aero_conditions.shape
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}")
        return point_clouds, aero_conditions
    
    def _generate_random_aero_conditions(self) -> np.ndarray:
        """Generate realistic random aerodynamic conditions"""
        angles = [0, 5, 10, 15, 20, 25, 30]
        aero_vector = []
        
        # Base characteristics with random variation
        base_cl_slope = np.random.uniform(0.08, 0.12)  # per degree
        base_cd0 = np.random.uniform(0.2, 0.4)  # Zero-lift drag
        base_cm0 = np.random.uniform(-0.1, 0.1)  # Zero-lift moment
        
        stall_angle = np.random.uniform(12, 18)  # Stall angle
        
        for angle in angles:
            # Lift coefficient (linear until stall)
            if angle <= stall_angle:
                cl = base_cl_slope * angle + np.random.normal(0, 0.02)
            else:
                # Post-stall decrease
                cl_max = base_cl_slope * stall_angle
                cl = cl_max * (1 - 0.1 * (angle - stall_angle)) + np.random.normal(0, 0.05)
            
            # Drag coefficient (quadratic)
            cd = base_cd0 + 0.01 * angle**2 + np.random.normal(0, 0.01)
            
            # Moment coefficient
            cm = base_cm0 - 0.01 * angle + np.random.normal(0, 0.005)
            
            aero_vector.extend([cl, cd, cm])
        
        return np.array(aero_vector)


def create_dataloader(
    dataset: PointCloudDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
):
    """Create DataLoader for point cloud dataset"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoader functionality")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    # Test data processing
    processor = PointCloudProcessor()
    
    # Generate synthetic dataset
    print("Creating synthetic dataset...")
    point_clouds, aero_conditions = processor.create_synthetic_dataset(
        num_samples=100,
        output_dir="/tmp/starship_synthetic_data",
        num_points=1024
    )
    
    print(f"Generated {len(point_clouds)} point clouds")
    print(f"Point cloud shape: {point_clouds.shape}")
    print(f"Aero conditions shape: {aero_conditions.shape}")
    
    # Test dataset
    dataset = PointCloudDataset(
        data_path="/tmp/starship_synthetic_data/dataset.h5",
        aero_conditions=aero_conditions,
        num_points=1024
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test data loading
    sample = dataset[0]
    print(f"Sample point cloud shape: {sample['point_cloud'].shape}")
    print(f"Sample aero condition shape: {sample['aero_condition'].shape}")
    
    # Test dataloader
    dataloader = create_dataloader(dataset, batch_size=8)
    batch = next(iter(dataloader))
    print(f"Batch point clouds shape: {batch['point_cloud'].shape}")
    print(f"Batch aero conditions shape: {batch['aero_condition'].shape}")