# Point-E Starship Point Cloud Generation

A comprehensive solution for generating Starship point clouds using OpenAI's Point-E model based on aerodynamic conditions. This project replaces unstable custom diffusion models with the proven Point-E architecture for reliable and high-quality point cloud generation.

## 🚀 Features

- **Aerodynamic Condition Processing**: Convert 21-dimensional aerodynamic parameters (7 angles of attack × 3 coefficients: CL, CD, CMy) into Point-E compatible inputs
- **Point-E Integration**: Seamless integration with OpenAI's Point-E model for stable point cloud generation
- **Synthetic Data Generation**: Create realistic Starship point clouds based on aerodynamic characteristics
- **Batch Processing**: Efficient batch generation with GPU acceleration support
- **Comprehensive Evaluation**: Quality metrics including Chamfer distance, Hausdorff distance, and geometric analysis
- **Rich Visualizations**: Interactive 3D visualizations using Plotly and matplotlib
- **Training Interface**: Fine-tune Point-E models on custom Starship data
- **Condition Interpolation**: Generate smooth transitions between different aerodynamic configurations

## 📋 Requirements

### Core Dependencies
```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib plotly
pip install scikit-learn tqdm h5py pandas pyyaml
pip install trimesh pillow
```

### Point-E Model
```bash
pip install git+https://github.com/openai/point-e.git
```

### Optional Dependencies
```bash
pip install open3d  # For advanced 3D visualization
pip install mayavi  # For additional visualization options
```

Or install all at once:
```bash
pip install -r point_e_starship/requirements.txt
```

## 🏗️ Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd desktop-tutorial
```

2. Install dependencies:
```bash
pip install -r point_e_starship/requirements.txt
```

3. Install Point-E:
```bash
pip install git+https://github.com/openai/point-e.git
```

## 🚀 Quick Start

### Basic Usage

```python
from point_e_starship import PointEGenerator, create_sample_aero_conditions

# Create sample aerodynamic conditions
aero_condition = create_sample_aero_conditions()

# Initialize generator
generator = PointEGenerator()

# Generate point cloud from aerodynamic conditions
point_clouds = generator.generate_from_aero_conditions(
    aero_conditions=aero_condition,
    num_samples=1,
    guidance_scale=15.0,
    upsample=True
)

# Save results
generator.save_point_clouds(point_clouds, "./output", format="ply")
```

### Text-Based Generation

```python
from point_e_starship import PointEGenerator

generator = PointEGenerator()

# Generate from text description
point_clouds = generator.generate_from_text(
    text_prompt="Starship-like high-performance aerospace vehicle with maximum lift coefficient 1.2 at 15 degrees, minimum drag coefficient 0.25 at 0 degrees, stable pitch moment characteristics",
    num_samples=2
)
```

### Synthetic Generation (No Point-E Required)

```python
from point_e_starship import PointCloudProcessor, create_sample_aero_conditions

processor = PointCloudProcessor()
aero_condition = create_sample_aero_conditions()

# Generate synthetic Starship point cloud
point_cloud = processor.generate_starship_pointcloud(
    aero_conditions=aero_condition,
    num_points=2048
)

# Save as PLY file
processor.save_point_cloud(point_cloud, "starship.ply", format="ply")
```

## 📁 Project Structure

```
point_e_starship/
├── requirements.txt          # Dependencies
├── __init__.py              # Package initialization
├── setup_point_e.py         # Point-E environment configuration
├── aero_condition_adapter.py # Aerodynamic condition processing
├── data_processor.py        # Point cloud data handling
├── point_e_generator.py     # Point-E generation interface
├── point_e_trainer.py       # Training interface
├── visualization.py         # Visualization tools
├── evaluation.py           # Evaluation metrics
└── demo.py                 # Complete demonstration

examples/
├── basic_generation.py     # Basic generation examples
├── conditional_generation.py # Conditional generation examples
└── batch_generation.py     # Batch processing examples
```

## 📊 Aerodynamic Conditions Format

The system expects 21-dimensional aerodynamic conditions representing:
- **7 angles of attack**: 0°, 5°, 10°, 15°, 20°, 25°, 30°
- **3 coefficients per angle**: CL (lift), CD (drag), CMy (moment)

Array format: `[CL0, CD0, CMy0, CL5, CD5, CMy5, ..., CL30, CD30, CMy30]`

### Example Aerodynamic Analysis

```python
from point_e_starship import AeroConditionAdapter, create_sample_aero_conditions

adapter = AeroConditionAdapter()
aero_condition = create_sample_aero_conditions()

# Parse and analyze
aero_data = adapter.parse_aero_vector(aero_condition)
analysis = adapter.analyze_aerodynamic_characteristics(aero_data)

print(f"Max lift coefficient: {analysis['max_lift_coefficient']:.3f}")
print(f"Min drag coefficient: {analysis['min_drag_coefficient']:.3f}")
print(f"Max L/D ratio: {analysis['max_ld_ratio']:.1f}")
print(f"Stall angle: {analysis['stall_angle']}°")

# Convert to text description
text_description = adapter.to_text_description(aero_condition)
print(f"Description: {text_description}")
```

## 🎨 Visualization

### Static Plots
```python
from point_e_starship import PointCloudVisualizer

visualizer = PointCloudVisualizer()

# Create matplotlib visualization
fig = visualizer.plot_point_cloud_matplotlib(
    point_cloud, 
    title="Starship Point Cloud",
    color_by="height",
    save_path="starship_plot.png"
)
```

### Interactive Visualizations
```python
# Create interactive Plotly visualization
fig = visualizer.plot_point_cloud_plotly(
    point_cloud,
    title="Interactive Starship Visualization",
    save_path="starship_interactive.html"
)
```

### Comparison Plots
```python
# Compare multiple point clouds
fig = visualizer.compare_point_clouds(
    point_clouds=[cloud1, cloud2, cloud3],
    labels=["Baseline", "High Lift", "Low Drag"],
    save_path="comparison.html"
)
```

## 📈 Evaluation

### Quality Assessment
```python
from point_e_starship import comprehensive_evaluation

results = comprehensive_evaluation(
    generated_clouds=point_clouds,
    aero_conditions=aero_conditions,
    reference_clouds=reference_clouds,  # Optional
    output_path="evaluation_results.json"
)

print(f"Overall quality: {results['summary']['overall_quality_score']:.3f}")
print(f"Aerodynamic consistency: {results['summary']['aerodynamic_consistency']:.3f}")
```

### Individual Metrics
```python
from point_e_starship import PointCloudEvaluator

evaluator = PointCloudEvaluator()

# Compute distances between point clouds
chamfer_dist = evaluator.chamfer_distance(cloud1, cloud2)
hausdorff_dist = evaluator.hausdorff_distance(cloud1, cloud2)

# Analyze geometric quality
quality = evaluator.geometric_quality_score(point_cloud)
print(f"Quality score: {quality['overall_quality']:.3f}")
```

## 🔧 Training

### Prepare Training Data
```python
from point_e_starship import PointCloudProcessor

processor = PointCloudProcessor()

# Create synthetic training dataset
point_clouds, aero_conditions = processor.create_synthetic_dataset(
    num_samples=1000,
    output_dir="./training_data",
    num_points=1024
)
```

### Fine-tune Point-E Model
```python
from point_e_starship import train_point_e_model

trainer = train_point_e_model(
    data_path="./training_data/dataset.h5",
    aero_conditions=aero_conditions,
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    save_dir="./checkpoints"
)
```

## 🎯 Examples

### Run Complete Demo
```bash
python point_e_starship/demo.py --output-dir ./demo_results
```

### Basic Generation Example
```bash
python examples/basic_generation.py
```

### Conditional Generation Example
```bash
python examples/conditional_generation.py
```

### Batch Processing Example
```bash
python examples/batch_generation.py
```

## 🔍 Key Improvements Over Previous Models

1. **Stability**: Point-E provides stable training and generation compared to custom diffusion models
2. **Convergence**: No more training convergence issues - Point-E models are pre-trained and proven
3. **Quality**: Higher quality point clouds with better structural coherence
4. **Efficiency**: Faster generation with batch processing capabilities
5. **Flexibility**: Support for both text-based and direct aerodynamic conditioning

## 📊 Performance Benchmarks

- **Synthetic Generation**: ~0.1-0.5 seconds per point cloud (CPU)
- **Point-E Generation**: ~10-30 seconds per point cloud (GPU recommended)
- **Batch Processing**: 10-50% efficiency improvement for large batches
- **Memory Usage**: ~2-4GB GPU memory for batch generation

## 🛠️ Troubleshooting

### Point-E Installation Issues
```bash
# If installation fails, try:
pip install --no-cache-dir git+https://github.com/openai/point-e.git

# For CUDA issues:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
- Reduce batch size for Point-E generation
- Use `upsample=False` for faster generation
- Enable gradient checkpointing for training

### Visualization Issues
```bash
# Install additional dependencies:
pip install open3d mayavi

# For headless servers:
export DISPLAY=:99
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

For questions and support:
- Create an issue in the GitHub repository
- Check the examples for common use cases
- Review the comprehensive evaluation results

## 🙏 Acknowledgments

- OpenAI for the Point-E model
- The aerospace community for aerodynamic insights
- Contributors to the scientific Python ecosystem