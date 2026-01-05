# Radiomics Extraction and Training Workflow

This document describes the separated workflow for radiomics extraction and model training to handle version conflicts between pyradiomics/pytorchradiomics and nnunetv2.

## Overview

The workflow is now split into two separate environments:

1. **Radiomics Extraction Environment** (pyradiomics/pytorchradiomics)
   - Extract radiomics features from predicted masks
   - Save results to CSV

2. **Training Environment** (no nnunetv2 dependency)
   - Load radiomics features from CSV
   - Perform feature selection and model training

## Step 1: Run Segmentation Inference (nnU-Net Environment)

First, run segmentation inference using `nnunet_segmentation_inference.py`:

```bash
python nnunet_segmentation_inference.py \
    --dataset-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib" \
    --model-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_results\Dataset360_oaizib\nnUNetTrainer__nnUNetPlans__3d_fullres" \
    --output-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_results" \
    --split both \
    --folds 0 1 2 3 4
```

This will:
- Run inference on each fold separately
- Perform majority voting across folds
- Save predictions as `{case_id}_majority_vote.nii.gz` in:
  - `{output_dir}/predicted_masks/train/` for training cases
  - `{output_dir}/predicted_masks/test/` for test cases
- Calculate and print Dice and HD95 metrics

## Step 2: Extract Radiomics from Predicted Masks (Radiomics Environment)

Use `torchradiomics_from_ROIs.py` in the radiomics extraction environment to extract features from the predicted masks.

### Extract from Pre-saved Predicted Masks (Recommended)

```bash
python torchradiomics_from_ROIs.py \
    --images-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" \
    --predicted-masks-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_results\predicted_masks" \
    --mode extract_from_masks \
    --split both \
    --output-dir /path/to/output \
    --radiomics-groups firstorder shape glcm gldm glrlm glszm ngtdm \
    --output-format csv \
    --voxel-array-shift 0 \
    --bin-width None
```

This will:
- Automatically detect train/test subdirectories in `--predicted-masks-dir`
- Look for `{case_id}_majority_vote.nii.gz` files in the appropriate subdirectory
- Extract radiomics features for each ROI
- Save results to CSV format: `radiomics_results.csv` (long format) and `radiomics_results_wide.csv` (wide format)

### Option B: Run nnU-Net Inference + Extract Radiomics

If you need to run inference first (requires nnunetv2):

```bash
python torchradiomics_from_ROIs.py \
    --images-dir /path/to/images \
    --nnunet-model-dir /path/to/nnunet_model \
    --output-dir /path/to/output \
    --folds 0 1 2 3 4 \
    --radiomics-groups firstorder shape glcm gldm glrlm glszm ngtdm \
    --output-format csv \
    --save-masks
```

## Step 2: Combine with Metadata (Optional)

If you need to combine radiomics with metadata (knee side, subject info), use `Radiomics_KLG_mapping_dataloader.py`:

```bash
python Radiomics_KLG_mapping_dataloader.py \
    --dataset-dir /path/to/Dataset360_oaizib \
    --knee-side-csv /path/to/kneeSideInfo.csv \
    --subinfo-train /path/to/subInfo_train.xlsx \
    --subinfo-test /path/to/subInfo_test.xlsx \
    --output-dir /path/to/output \
    --split train \
    --load-existing /path/to/radiomics_results.csv \
    --output-format csv
```

This will:
- Load radiomics from CSV
- Merge with knee side and subject information
- Save combined data: `radiomics_klg_data_train.csv`

## Step 3: Train Model with Feature Selection

Use `train_klg_topk_enum.py` in the training environment (no nnunetv2 needed):

```bash
python train_klg_topk_enum.py \
    --input-table /path/to/radiomics_klg_data_train.csv \
    --output-dir /path/to/klg_topk_results \
    --target-col KLGrade \
    --task multiclass \
    --roi-mode all_rois \
    --k 15 \
    --candidate-pool-size 60 \
    --cv-folds 5 \
    --score macro_f1 \
    --balanced
```

### Key Parameters:

- `--input-table`: Path to CSV file with radiomics features (from Step 1 or 2)
- `--roi-mode all_rois`: Merges all ROIs into one row per case (whole image features)
- `--k`: Number of top features to select
- `--candidate-pool-size`: Number of top-ranked features to consider
- `--score`: Primary metric for feature selection

### Output:

The script will:
1. Load radiomics features from CSV
2. Merge ROIs into whole-image features (one row per case)
3. Perform nested CV feature selection
4. Train final model with best feature subset
5. Save:
   - `results_subsets.csv`: Summary of results
   - `best_subset.json`: Best feature subset for each ROI
   - `final_model_*.joblib`: Trained models with prediction pipeline

### Using the Trained Model:

The saved model (`FullPipeline`) can be used to predict on new data:

```python
from joblib import load
import pandas as pd

# Load model
model = load('final_model_all_rois.joblib')

# Load new data (same format as training)
X_new = pd.read_csv('new_radiomics_data.csv')

# Predict probabilities (whole image features)
probabilities = model.predict_proba(X_new)

# Predict labels
labels = model.predict(X_new)
```

## File Formats

### Long Format (from extraction):
```
case_id,roi_name,feature_name,value
oaizib_001,Femur,firstorder_Mean,123.45
oaizib_001,Femur,firstorder_StdDev,12.34
...
```

### Wide Format (for training):
```
case_id,roi_name,firstorder_Mean,firstorder_StdDev,...
oaizib_001,Femur,123.45,12.34,...
oaizib_001,Femoral_Cartilage,98.76,10.23,...
```

### Whole Image Format (after merging ROIs):
```
case_id,roi_Femur__firstorder_Mean,roi_Femur__firstorder_StdDev,roi_Femoral_Cartilage__firstorder_Mean,...
oaizib_001,123.45,12.34,98.76,...
```

## Notes

1. **Version Conflicts**: Keep radiomics extraction and training in separate environments
2. **Feature Selector**: The feature selector works on whole image features (all ROIs combined) and outputs class probabilities
3. **CSV Format**: Both long and wide formats are supported; the dataloader automatically detects the format
4. **ROI Merging**: When using `--roi-mode all_rois`, features are prefixed with `roi_{roi_name}__` to avoid conflicts

