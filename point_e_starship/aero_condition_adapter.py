"""
Aerodynamic Condition Adapter
Converts 21-dimensional aerodynamic conditions to Point-E compatible inputs
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AeroConditionAdapter:
    """
    Adapter for converting aerodynamic conditions to Point-E inputs
    
    Handles 21-dimensional input: 7 angles of attack × 3 coefficients (CL, CD, CMy)
    """
    
    def __init__(self):
        """Initialize the aerodynamic condition adapter"""
        self.angle_names = [
            "0_degrees", "5_degrees", "10_degrees", "15_degrees", 
            "20_degrees", "25_degrees", "30_degrees"
        ]
        self.coeff_names = ["CL", "CD", "CMy"]
        
        # Define aerodynamic parameter ranges for normalization
        self.param_ranges = {
            "CL": (-1.5, 2.5),     # Lift coefficient range
            "CD": (0.0, 2.0),      # Drag coefficient range  
            "CMy": (-2.0, 2.0)     # Moment coefficient range
        }
        
    def parse_aero_vector(self, aero_vector: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Parse 21-dimensional aerodynamic vector into structured format
        
        Args:
            aero_vector: 21-dimensional array [CL0,CD0,CMy0, CL5,CD5,CMy5, ...]
            
        Returns:
            Dictionary with angle keys and coefficient values
        """
        if len(aero_vector) != 21:
            raise ValueError(f"Expected 21-dimensional vector, got {len(aero_vector)}")
        
        result = {}
        for i, angle in enumerate(self.angle_names):
            start_idx = i * 3
            result[angle] = {
                "CL": float(aero_vector[start_idx]),
                "CD": float(aero_vector[start_idx + 1]), 
                "CMy": float(aero_vector[start_idx + 2])
            }
        
        return result
    
    def normalize_coefficients(self, aero_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Normalize aerodynamic coefficients to [0, 1] range
        
        Args:
            aero_data: Parsed aerodynamic data
            
        Returns:
            Normalized aerodynamic data
        """
        normalized = {}
        
        for angle, coeffs in aero_data.items():
            normalized[angle] = {}
            for coeff_name, value in coeffs.items():
                min_val, max_val = self.param_ranges[coeff_name]
                normalized_val = (value - min_val) / (max_val - min_val)
                normalized_val = np.clip(normalized_val, 0.0, 1.0)
                normalized[angle][coeff_name] = normalized_val
                
        return normalized
    
    def analyze_aerodynamic_characteristics(self, aero_data: Dict[str, Dict[str, float]]) -> Dict[str, any]:
        """
        Analyze aerodynamic characteristics for text generation
        
        Args:
            aero_data: Parsed aerodynamic data
            
        Returns:
            Analysis results including key characteristics
        """
        # Extract coefficient arrays
        angles = [0, 5, 10, 15, 20, 25, 30]
        cl_values = [aero_data[f"{angle}_degrees"]["CL"] for angle in angles]
        cd_values = [aero_data[f"{angle}_degrees"]["CD"] for angle in angles]
        cmy_values = [aero_data[f"{angle}_degrees"]["CMy"] for angle in angles]
        
        # Find key characteristics
        max_cl_idx = np.argmax(cl_values)
        min_cd_idx = np.argmin(cd_values)
        max_ld_ratio = max(cl_values[i] / max(cd_values[i], 0.001) for i in range(len(angles)))
        
        # Stall analysis
        cl_derivative = np.diff(cl_values)
        stall_indicators = np.where(cl_derivative < -0.1)[0]
        stall_angle = angles[stall_indicators[0] + 1] if len(stall_indicators) > 0 else None
        
        # Moment stability
        moment_stability = "stable" if np.mean(cmy_values) < 0 else "unstable"
        
        analysis = {
            "max_lift_angle": angles[max_cl_idx],
            "max_lift_coefficient": cl_values[max_cl_idx],
            "min_drag_angle": angles[min_cd_idx], 
            "min_drag_coefficient": cd_values[min_cd_idx],
            "max_ld_ratio": max_ld_ratio,
            "stall_angle": stall_angle,
            "moment_stability": moment_stability,
            "cl_range": (min(cl_values), max(cl_values)),
            "cd_range": (min(cd_values), max(cd_values)),
            "operating_envelope": self._classify_operating_envelope(cl_values, cd_values)
        }
        
        return analysis
    
    def _classify_operating_envelope(self, cl_values: List[float], cd_values: List[float]) -> str:
        """Classify the aerodynamic operating envelope"""
        max_cl = max(cl_values)
        min_cd = min(cd_values)
        
        if max_cl > 1.5 and min_cd < 0.3:
            return "high_performance"
        elif max_cl > 1.0 and min_cd < 0.5:
            return "moderate_performance"
        elif max_cl < 0.8:
            return "low_lift"
        else:
            return "high_drag"
    
    def to_text_description(self, aero_vector: np.ndarray) -> str:
        """
        Convert aerodynamic vector to natural language description
        
        Args:
            aero_vector: 21-dimensional aerodynamic conditions
            
        Returns:
            Text description of aerodynamic characteristics
        """
        aero_data = self.parse_aero_vector(aero_vector)
        analysis = self.analyze_aerodynamic_characteristics(aero_data)
        
        # Build description
        description_parts = []
        
        # Performance classification
        envelope = analysis["operating_envelope"]
        if envelope == "high_performance":
            description_parts.append("high-performance aerospace vehicle")
        elif envelope == "moderate_performance":
            description_parts.append("moderate-performance aircraft configuration")
        elif envelope == "low_lift":
            description_parts.append("low-lift aerodynamic body")
        else:
            description_parts.append("high-drag configuration")
        
        # Lift characteristics
        max_cl = analysis["max_lift_coefficient"]
        max_cl_angle = analysis["max_lift_angle"]
        description_parts.append(f"with maximum lift coefficient {max_cl:.2f} at {max_cl_angle} degrees")
        
        # Drag characteristics
        min_cd = analysis["min_drag_coefficient"]
        min_cd_angle = analysis["min_drag_angle"]
        description_parts.append(f"minimum drag coefficient {min_cd:.3f} at {min_cd_angle} degrees")
        
        # Stall behavior
        if analysis["stall_angle"]:
            description_parts.append(f"stall onset at {analysis['stall_angle']} degrees")
        
        # Stability
        stability = analysis["moment_stability"]
        description_parts.append(f"{stability} pitch moment characteristics")
        
        # L/D ratio
        ld_ratio = analysis["max_ld_ratio"]
        description_parts.append(f"maximum lift-to-drag ratio of {ld_ratio:.1f}")
        
        return "Starship-like " + ", ".join(description_parts)
    
    def to_embedding_vector(self, aero_vector: np.ndarray, embed_dim: int = 512) -> torch.Tensor:
        """
        Convert aerodynamic vector to embedding for direct model input
        
        Args:
            aero_vector: 21-dimensional aerodynamic conditions
            embed_dim: Target embedding dimension
            
        Returns:
            Embedding tensor
        """
        aero_data = self.parse_aero_vector(aero_vector)
        normalized_data = self.normalize_coefficients(aero_data)
        
        # Create feature vector from normalized data
        features = []
        for angle in self.angle_names:
            for coeff in self.coeff_names:
                features.append(normalized_data[angle][coeff])
        
        features = np.array(features)  # 21-dimensional
        
        # Add derived features
        analysis = self.analyze_aerodynamic_characteristics(aero_data)
        derived_features = [
            analysis["max_lift_coefficient"] / 3.0,  # Normalized max CL
            analysis["min_drag_coefficient"] / 2.0,  # Normalized min CD
            analysis["max_ld_ratio"] / 50.0,         # Normalized L/D ratio
            1.0 if analysis["moment_stability"] == "stable" else 0.0,
            (analysis["stall_angle"] or 30) / 30.0,  # Normalized stall angle
        ]
        
        # Combine original and derived features
        all_features = np.concatenate([features, derived_features])  # 26-dimensional
        
        # Expand to target embedding dimension using learned transformation
        if embed_dim > len(all_features):
            # Repeat and transform features to reach target dimension
            repeats = embed_dim // len(all_features) + 1
            expanded = np.tile(all_features, repeats)[:embed_dim]
            
            # Add some structured variation
            for i in range(embed_dim):
                if i % 4 == 1:
                    expanded[i] *= 0.8  # Slight variation
                elif i % 4 == 2:
                    expanded[i] *= 1.2
                elif i % 4 == 3:
                    expanded[i] = np.sin(expanded[i] * np.pi)
        else:
            # Truncate if needed
            expanded = all_features[:embed_dim]
        
        return torch.tensor(expanded, dtype=torch.float32)
    
    def create_batch_descriptions(self, aero_vectors: np.ndarray) -> List[str]:
        """
        Create batch of text descriptions from multiple aerodynamic vectors
        
        Args:
            aero_vectors: Array of shape (batch_size, 21)
            
        Returns:
            List of text descriptions
        """
        descriptions = []
        for aero_vector in aero_vectors:
            descriptions.append(self.to_text_description(aero_vector))
        return descriptions
    
    def create_batch_embeddings(self, aero_vectors: np.ndarray, embed_dim: int = 512) -> torch.Tensor:
        """
        Create batch of embedding vectors from multiple aerodynamic vectors
        
        Args:
            aero_vectors: Array of shape (batch_size, 21)
            embed_dim: Target embedding dimension
            
        Returns:
            Embedding tensor of shape (batch_size, embed_dim)
        """
        embeddings = []
        for aero_vector in aero_vectors:
            embeddings.append(self.to_embedding_vector(aero_vector, embed_dim))
        return torch.stack(embeddings)


def create_sample_aero_conditions() -> np.ndarray:
    """Create sample aerodynamic conditions for testing"""
    # Sample Starship-like aerodynamic data
    angles = [0, 5, 10, 15, 20, 25, 30]
    
    # Typical values for a lifting body like Starship
    cl_values = [0.1, 0.3, 0.6, 0.9, 1.1, 1.0, 0.8]  # Lift coefficient
    cd_values = [0.25, 0.28, 0.35, 0.48, 0.65, 0.85, 1.1]  # Drag coefficient  
    cmy_values = [-0.05, -0.08, -0.12, -0.15, -0.18, -0.20, -0.18]  # Moment coefficient
    
    # Combine into 21-dimensional vector
    aero_vector = []
    for i in range(len(angles)):
        aero_vector.extend([cl_values[i], cd_values[i], cmy_values[i]])
    
    return np.array(aero_vector)


if __name__ == "__main__":
    # Test the adapter
    adapter = AeroConditionAdapter()
    
    # Create sample data
    sample_aero = create_sample_aero_conditions()
    print(f"Sample aero vector shape: {sample_aero.shape}")
    
    # Test text description
    description = adapter.to_text_description(sample_aero)
    print(f"\nText description:\n{description}")
    
    # Test embedding
    embedding = adapter.to_embedding_vector(sample_aero)
    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    # Test batch processing
    batch_aero = np.stack([sample_aero, sample_aero * 1.1, sample_aero * 0.9])
    batch_descriptions = adapter.create_batch_descriptions(batch_aero)
    print(f"\nBatch descriptions ({len(batch_descriptions)} items):")
    for i, desc in enumerate(batch_descriptions):
        print(f"  {i+1}: {desc[:100]}...")