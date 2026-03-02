# Installation Guide

This guide covers different ways to install VisProbe and handle common installation issues.

## 📦 Requirements

- **Python**: 3.9 or higher
- **PyTorch**: 2.0.0 or higher
- **Operating System**: Linux, macOS, or Windows

## 🚀 Quick Install

### Basic Installation

```bash
pip install visprobe
```

### With Optional Dependencies

```bash
# For adversarial attacks
pip install visprobe[adversarial]

# For enhanced visualizations
pip install visprobe[viz]

# For Bayesian optimization search
pip install visprobe[bayesian]

# Everything
pip install visprobe[all]
```

## 🔧 Installation Methods

### 1. From PyPI (Recommended)

```bash
pip install visprobe
```

### 2. From Source

```bash
# Clone repository
git clone https://github.com/bilgedemirkaya/VisProbe.git
cd VisProbe

# Install in editable mode
pip install -e .

# Or build and install
pip install .
```

### 3. Using Poetry

```bash
poetry add visprobe
```

### 4. Using Conda

First install PyTorch with conda, then install VisProbe with pip:

```bash
# Install PyTorch (example for CUDA 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Then install VisProbe
pip install visprobe
```

## 🎯 Platform-Specific Instructions

### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-pip python3-dev

# Install VisProbe
pip install visprobe
```

### macOS

```bash
# Using Homebrew
brew install python@3.9

# Install VisProbe
pip3 install visprobe
```

### Windows

```powershell
# Using PowerShell
python -m pip install --upgrade pip
pip install visprobe
```

## 🔌 GPU Support

### CUDA Installation

VisProbe automatically uses GPU if available. Ensure you have the correct CUDA version:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch for your CUDA version
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install VisProbe
pip install visprobe
```

### Verify GPU Support

```python
import torch
import visprobe

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"VisProbe version: {visprobe.__version__}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 📚 Optional Dependencies

### Adversarial Attacks

For adversarial robustness testing:

```bash
pip install adversarial-robustness-toolbox>=1.18.0
```

### Enhanced Visualizations

For better visualization support:

```bash
pip install altair>=4.2.0
```

### Bayesian Search

For Bayesian optimization search method:

```bash
pip install scipy>=1.9.0 scikit-learn>=1.0.0
```

### Development Tools

For contributing to VisProbe:

```bash
pip install black flake8 isort mypy pytest
```

## 🐳 Docker Installation

### Using Pre-built Image

```bash
docker pull visprobe/visprobe:latest
docker run -it visprobe/visprobe:latest
```

### Building Custom Image

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install visprobe[all]

# Copy your code
COPY . .

CMD ["python", "your_script.py"]
```

Build and run:

```bash
docker build -t my-visprobe-app .
docker run -it my-visprobe-app
```

## 🔍 Verifying Installation

### Basic Verification

```python
import visprobe

# Check version
print(visprobe.__version__)

# List available functions
print(dir(visprobe))

# Test basic functionality
from visprobe import list_presets, list_perturbations

print("Presets:", list(list_presets().keys()))
print("Perturbations:", len(list_perturbations()))
```

### Full Verification

```python
import torch
import torchvision.models as models
from visprobe import search

# Create dummy model and data
model = models.resnet18(pretrained=True)
model.eval()

# Create dummy data
data = [(torch.randn(3, 224, 224), 0) for _ in range(10)]

# Run basic test
try:
    report = search(
        model, data,
        perturbation="gaussian_noise",
        level_hi=0.1,
        num_steps=3
    )
    print("✅ VisProbe is working correctly!")
    print(f"Test score: {report.score}%")
except Exception as e:
    print(f"❌ Error: {e}")
```

## 🔧 Troubleshooting

### Import Error: No module named 'visprobe'

```bash
# Ensure pip is updated
pip install --upgrade pip

# Reinstall VisProbe
pip uninstall visprobe
pip install visprobe
```

### PyTorch Not Found

```bash
# Install PyTorch first
pip install torch torchvision

# Then install VisProbe
pip install visprobe
```

### CUDA/GPU Issues

```python
# Force CPU usage if GPU issues
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or specify device in search
from visprobe import search
report = search(model, data, device='cpu')
```

### Memory Issues

If you encounter out-of-memory errors:

```python
# Reduce batch size
report = search(model, data, batch_size=8)

# Use fewer samples
report = search(model, data[:50])

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Version Conflicts

```bash
# Create fresh virtual environment
python -m venv visprobe_env
source visprobe_env/bin/activate  # Windows: visprobe_env\Scripts\activate

# Install with specific versions
pip install torch==2.0.0 torchvision==0.15.0
pip install visprobe
```

## 🔄 Updating VisProbe

### Update to Latest Version

```bash
pip install --upgrade visprobe
```

### Update to Specific Version

```bash
pip install visprobe==0.2.0
```

### Check Current Version

```python
import visprobe
print(visprobe.__version__)
```

## 🗑️ Uninstallation

```bash
pip uninstall visprobe
```

## 💡 Environment Setup Tips

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install VisProbe
pip install visprobe

# When done, deactivate
deactivate
```

### Conda Environment

```bash
# Create conda environment
conda create -n visprobe python=3.10

# Activate it
conda activate visprobe

# Install PyTorch and VisProbe
conda install pytorch torchvision -c pytorch
pip install visprobe
```

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Search [GitHub Issues](https://github.com/bilgedemirkaya/VisProbe/issues)
3. Create a new issue with:
   - Python version
   - PyTorch version
   - VisProbe version
   - Full error message
   - Minimal code example

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 8 GB
- **Storage**: 1 GB free space
- **Python**: 3.9+

### Recommended Requirements
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 5 GB free space
- **Python**: 3.10 or 3.11

### For Large-Scale Testing
- **RAM**: 32 GB or more
- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **Multiple GPUs** supported for parallel testing