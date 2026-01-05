# Radiomics Extraction from nnU-Net Predicted ROIs

This script extracts radiomics features from ROIs defined by nnU-Net ensemble predictions.

## Features

- **Ensemble Prediction**: Uses multiple nnU-Net folds (default: 0-4) for robust segmentation
- **Multi-ROI Support**: Extracts features for all 5 ROIs (labels) separately
- **Configurable Feature Groups**: Select which radiomics groups to extract
- **Flexible Output**: CSV or Parquet format, both long and wide formats
- **Robust Error Handling**: Gracefully handles empty ROIs and missing cases

## Requirements

- PyTorch
- nnU-Net v2
- torchradiomics
- SimpleITK
- pandas
- numpy

## Usage

### Basic Usage

```bash
python torchradiomics_from_ROIs.py \
    --images-dir /path/to/images \
    --nnunet-model-dir /path/to/nnUNet_results/Dataset360_oaizib/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output-dir /path/to/output
```

### Advanced Usage

```bash
python torchradiomics_from_ROIs.py \
    --images-dir Dataset360_oaizib/imagesTs \
    --nnunet-model-dir nnUNet/nnUNet_results/Dataset360_oaizib/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output-dir results/radiomics \
    --folds 0 1 2 3 4 \
    --radiomics-groups firstorder shape glcm glrlm \
    --device cuda:0 \
    --save-masks \
    --output-format parquet \
    --checkpoint-name checkpoint_best.pth
```

## Arguments

### Required Arguments

- `--images-dir`: Directory containing input NIfTI images (`.nii.gz` or `.nii`)
- `--nnunet-model-dir`: Path to nnU-Net model directory (contains `fold_X` subdirectories)
- `--output-dir`: Output directory for results and predicted masks

### Optional Arguments

- `--folds`: Folds to use for ensemble (default: `0 1 2 3 4`)
- `--radiomics-groups`: Feature groups to extract (default: all groups)
  - Options: `firstorder`, `shape`, `glcm`, `gldm`, `glrlm`, `glszm`, `ngtdm`
- `--device`: PyTorch device (default: auto-detect, `cuda:0` or `cpu`)
- `--checkpoint-name`: Checkpoint filename (default: `checkpoint_best.pth`)
- `--save-masks`: Save predicted segmentation masks to disk
- `--voxel-array-shift`: Shift to apply to avoid negative values (default: 0)
- `--bin-width`: Histogram bin width (default: None, uses 256 bins)
- `--output-format`: Output format: `csv` or `parquet` (default: `csv`)

## Output Structure

The script generates two output files:

1. **Long Format** (`radiomics_results.csv` or `.parquet`):
   - Columns: `case_id`, `roi_name`, `feature_name`, `value`
   - One row per case-ROI-feature combination

2. **Wide Format** (`radiomics_results_wide.csv` or `.parquet`):
   - One row per case-ROI combination
   - One column per feature

### Example Output (Long Format)

```
case_id,roi_name,feature_name,value
case_001,Femur,Mean,1234.56
case_001,Femur,StdDev,234.56
case_001,Femoral_Cartilage,Mean,567.89
...
```

### Example Output (Wide Format)

```
case_id,roi_name,Mean,StdDev,Energy,Entropy,...
case_001,Femur,1234.56,234.56,1.23e6,5.67,...
case_001,Femoral_Cartilage,567.89,123.45,2.34e5,4.56,...
...
```

## ROI Labels

ROI labels are automatically extracted from `dataset.json` in the model directory. If not found, ROIs are named `roi_1`, `roi_2`, etc.

For Dataset360_oaizib, the labels are:
- `Femur` (label 1)
- `Femoral_Cartilage` (label 2)
- `Tibia` (label 3)
- `Medial_Tibial_Cartilage` (label 4)
- `Lateral_Tibial_Cartilage` (label 5)

## Feature Groups

### First-Order Features
Statistical features computed from the histogram of ROI intensities:
- Mean, Median, StdDev, Variance
- Min, Max, Range, Percentiles
- Energy, Entropy, Uniformity
- Skewness, Kurtosis
- MAD, Robust MAD, RMS

### Shape Features
Geometric features describing ROI shape:
- Volume, Surface Area
- Sphericity, Compactness
- Flatness, Elongation
- Axis lengths, Diameters

### Texture Features
- **GLCM**: Gray Level Co-occurrence Matrix
- **GLDM**: Gray Level Dependence Matrix
- **GLRLM**: Gray Level Run Length Matrix
- **GLSZM**: Gray Level Size Zone Matrix
- **NGTDM**: Neighboring Gray Tone Difference Matrix

## Ensemble Prediction

The script performs ensemble prediction by:
1. Loading checkpoints from specified folds
2. Running inference with each fold
3. Averaging softmax probabilities across folds
4. Taking argmax to get final segmentation

This provides more robust predictions than single-fold inference.

## Error Handling

- Empty ROIs (no voxels) are skipped with a warning
- Invalid features (NaN/Inf) are logged and excluded
- Failed cases are logged but processing continues
- Missing checkpoints raise an error

## Example Workflow

```bash
# 1. Extract all features for all folds
python torchradiomics_from_ROIs.py \
    --images-dir Dataset360_oaizib/imagesTs \
    --nnunet-model-dir nnUNet/nnUNet_results/Dataset360_oaizib/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output-dir results/radiomics_all \
    --save-masks

# 2. Extract only first-order and shape features
python torchradiomics_from_ROIs.py \
    --images-dir Dataset360_oaizib/imagesTs \
    --nnunet-model-dir nnUNet/nnUNet_results/Dataset360_oaizib/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output-dir results/radiomics_basic \
    --radiomics-groups firstorder shape

# 3. Use only 3 folds for faster processing
python torchradiomics_from_ROIs.py \
    --images-dir Dataset360_oaizib/imagesTs \
    --nnunet-model-dir nnUNet/nnUNet_results/Dataset360_oaizib/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output-dir results/radiomics_3folds \
    --folds 0 1 2
```

## Notes

- The script preserves image spacing/origin/direction from the original images
- Predicted masks are saved with the same geometry as input images
- Processing is sequential (one case at a time) to manage memory
- GPU is recommended for faster inference

## Troubleshooting

### Out of Memory
- Use fewer folds: `--folds 0 1 2`
- Use CPU: `--device cpu` (slower but less memory)
- Process fewer cases at a time

### Missing Checkpoints
- Ensure all specified folds exist in the model directory
- Check `--checkpoint-name` matches the checkpoint files

### Empty ROIs
- Check that the segmentation mask contains the expected labels
- Verify the model was trained on the same dataset structure

### Feature Extraction Errors
- Some texture features may fail on very small ROIs
- Check logs for specific error messages
- Try excluding problematic feature groups

