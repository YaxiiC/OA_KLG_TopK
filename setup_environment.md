# Environment Setup for OA KLG TopK Project

This guide helps you set up the Python environment for running the radiomics extraction and KLG mapping pipeline.

## Prerequisites

- **CUDA**: nnU-Net and PyTorch Radiomics require CUDA for GPU acceleration
  - Check your CUDA version: `nvidia-smi` or `nvcc --version`
  - Recommended: CUDA 11.8 or 12.1
- **Python**: 3.7 (required for pyradiomics compatibility)
  - Note: Some newer package versions may not support Python 3.7, so version constraints are set accordingly

## Option 1: Using Conda (Recommended)

### Step 1: Create the environment

```bash
conda env create -f environment.yml
```

### Step 2: Activate the environment

```bash
conda activate oa_klg_topk
```

### Step 3: Install torchradiomics (if not installed automatically)

If torchradiomics installation failed during conda env create, install it manually:

**On Windows:**
```bash
pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git
```

**Or use the provided script:**
```bash
install_torchradiomics.bat
```

**On Linux/Mac:**
```bash
pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git
```

**Or use the provided script:**
```bash
chmod +x install_torchradiomics.sh
./install_torchradiomics.sh
```

**Note:** If you get an error about git not being found, you may need to install Git first or use a different method.

### Step 4: Verify installation

```bash
python -c "import torch; import torchradiomics; import nnunetv2; print('All packages installed successfully!')"
```

### Step 4: Install nnU-Net (if needed)

nnU-Net v2 may require additional setup:

```bash
# Install nnU-Net dataset utilities
pip install nnunetv2[all]

# Or if you need specific features:
pip install nnunetv2[inference]
```

## Option 2: Using pip/venv

### Step 1: Create virtual environment

```bash
python -m venv venv_oa_klg

# On Windows:
venv_oa_klg\Scripts\activate

# On Linux/Mac:
source venv_oa_klg/bin/activate
```

### Step 2: Install PyTorch (with CUDA support)

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (not recommended for nnU-Net):**
```bash
pip install torch torchvision torchaudio
```

### Step 3: Install all requirements

```bash
pip install -r requirements.txt
```

**Note:** `torchradiomics` is not on PyPI, so you need to install it manually from GitHub:

```bash
pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git
```

**On Windows, use the provided script:**
```bash
install_torchradiomics.bat
```

### Step 4: Verify installation

```bash
python -c "import torch; import torchradiomics; import nnunetv2; print('All packages installed successfully!')"
```

## Troubleshooting

### CUDA Version Mismatch

If you get CUDA errors, check your CUDA version and install matching PyTorch:

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### nnU-Net Installation Issues

If nnU-Net fails to install:

```bash
# Try installing from source
pip install git+https://github.com/MIC-DKFZ/nnUNet.git

# Or install specific version
pip install nnunetv2==2.2.0
```

### torchradiomics Installation Issues

If torchradiomics fails to install:

```bash
# Install directly from GitHub
pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git

# If you don't have git, you can download and install manually:
# 1. Download the repository as ZIP
# 2. Extract and navigate to the directory
# 3. Run: pip install .
```

### SimpleITK Issues

If SimpleITK installation fails:

```bash
# On Linux/Mac, you may need:
conda install -c conda-forge simpleitk

# Or use pip with pre-built wheels
pip install SimpleITK
```

### Memory Issues

If you run out of memory during training:

1. Reduce batch size in nnU-Net configuration
2. Use mixed precision training
3. Reduce number of parallel jobs (`--n-jobs`)

## Testing the Installation

Run a quick test to verify everything works:

```python
import sys
print(f"Python version: {sys.version}")

import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Test torchradiomics (if using PyTorch-based)
try:
    from torchradiomics import TorchRadiomicsFirstOrder
    print("torchradiomics imported successfully")
except ImportError:
    print("torchradiomics not available (optional)")

# Test pyradiomics (if using original PyRadiomics)
try:
    import radiomics
    print("pyradiomics imported successfully")
except ImportError:
    print("pyradiomics not available (optional)")

import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("All core packages imported successfully!")
```

## Python 3.7 Compatibility Notes

Since Python 3.7 is required for pyradiomics, some package versions have been constrained:
- **numpy**: <1.24.0 (numpy 1.24+ requires Python 3.8+)
- **pandas**: <2.0.0 (pandas 2.0+ requires Python 3.8+)
- **scikit-learn**: <1.2.0 (scikit-learn 1.2+ requires Python 3.8+)
- **scipy**: <1.10.0 (scipy 1.10+ requires Python 3.8+)

If you encounter compatibility issues, you may need to adjust these version constraints.

## Environment Variables (Optional)

You may want to set these environment variables:

```bash
# nnU-Net paths (if using custom locations)
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# For Windows (PowerShell):
$env:nnUNet_raw="C:\path\to\nnUNet_raw"
$env:nnUNet_preprocessed="C:\path\to\nnUNet_preprocessed"
$env:nnUNet_results="C:\path\to\nnUNet_results"
```

## Next Steps

1. Extract radiomics features:
   ```bash
   python Radiomics_KLG_mapping_dataloader.py --split train --output-dir ./radiomics_output
   ```

2. Train KLG mapping model:
   ```bash
   python train_klg_topk_enum.py --use-dataloader --dataset-dir ... --target-col KLGrade --k 15
   ```

