# Radiomics KLG Mapping DataLoader

This module extracts radiomics features from ground truth ROIs in the Dataset360_oaizib dataset and combines them with knee side information and subject metadata for KLG (Kellgren-Lawrence Grade) mapping model training.

## Features

- **Radiomics Extraction**: Extracts radiomics features from ground truth segmentation masks using torchradiomics
- **Metadata Integration**: Combines radiomics with knee side information and subject metadata
- **Train/Test Split**: Handles both training and test splits automatically
- **ROI-specific Data**: Provides easy access to data for specific ROIs (Femur, Femoral Cartilage, Tibia, etc.)

## Dataset Structure

The script expects the following structure:

```
Dataset360_oaizib/
├── dataset.json          # Dataset configuration with ROI labels
├── imagesTr/             # Training images
├── imagesTs/             # Test images
├── labelsTr/             # Training ground truth labels
└── labelsTs/             # Test ground truth labels

OAI-ZIB-CM/info/
├── kneeSideInfo.csv      # Knee side information (right/left)
├── subInfo_train.xlsx    # Training subject information
└── subInfo_test.xlsx     # Test subject information
```

## ROI Labels

Based on `dataset.json`, the following ROIs are available:
- **Femur** (label 1)
- **Femoral Cartilage** (label 2)
- **Tibia** (label 3)
- **Medial Tibial Cartilage** (label 4)
- **Lateral Tibial Cartilage** (label 5)

## Usage

### Command Line Interface

#### Extract radiomics features for both train and test splits:

```bash
python Radiomica_KLG_mapping_dataloader.py \
    --dataset-dir ../Dataset360_oaizib \
    --knee-side-csv ../OAI-ZIB-CM/info/kneeSideInfo.csv \
    --subinfo-train ../OAI-ZIB-CM/info/subInfo_train.xlsx \
    --subinfo-test ../OAI-ZIB-CM/info/subInfo_test.xlsx \
    --output-dir ./radiomics_output \
    --split both \
    --radiomics-groups firstorder shape glcm gldm glrlm glszm ngtdm \
    --output-format parquet
```

#### Extract only training data:

```bash
python Radiomica_KLG_mapping_dataloader.py \
    --dataset-dir ../Dataset360_oaizib \
    --knee-side-csv ../OAI-ZIB-CM/info/kneeSideInfo.csv \
    --subinfo-train ../OAI-ZIB-CM/info/subInfo_train.xlsx \
    --subinfo-test ../OAI-ZIB-CM/info/subInfo_test.xlsx \
    --output-dir ./radiomics_output \
    --split train \
    --output-format csv
```

#### Load existing features and combine with metadata:

```bash
python Radiomica_KLG_mapping_dataloader.py \
    --dataset-dir ../Dataset360_oaizib \
    --knee-side-csv ../OAI-ZIB-CM/info/kneeSideInfo.csv \
    --subinfo-train ../OAI-ZIB-CM/info/subInfo_train.xlsx \
    --subinfo-test ../OAI-ZIB-CM/info/subInfo_test.xlsx \
    --output-dir ./radiomics_output \
    --split train \
    --load-existing ./radiomics_output/radiomics_features_train.parquet
```

### Python API

```python
from Radiomica_KLG_mapping_dataloader import RadiomicsKLGDataLoader
from pathlib import Path

# Initialize data loader
loader = RadiomicsKLGDataLoader(
    dataset_dir=Path("../Dataset360_oaizib"),
    knee_side_csv=Path("../OAI-ZIB-CM/info/kneeSideInfo.csv"),
    subinfo_train=Path("../OAI-ZIB-CM/info/subInfo_train.xlsx"),
    subinfo_test=Path("../OAI-ZIB-CM/info/subInfo_test.xlsx"),
    radiomics_groups=['firstorder', 'shape', 'glcm', 'gldm'],
    split='train'
)

# Extract radiomics features
loader.extract_all_radiomics(
    save_path=Path("./output/radiomics_train.parquet")
)

# Get combined data with metadata
df = loader.get_training_data()

# Get data for a specific ROI
df_femoral_cartilage = loader.get_roi_specific_data('Femoral_Cartilage')

# Prepare features and target for training
# Assuming 'KLG' is the target column in subinfo_df
X = df_femoral_cartilage.drop(columns=['case_id', 'roi_name', 'KLG'])
y = df_femoral_cartilage['KLG']
```

## Output Files

The script generates the following output files:

1. **`radiomics_features_{split}.{format}`**: Long format radiomics features (case_id, roi_name, feature_name, value)
2. **`radiomics_klg_data_{split}.{format}`**: Wide format combined data with metadata (one row per case+ROI)
3. **`radiomics_klg_data_{split}_{roi_name}.{format}`**: ROI-specific data files

## Radiomics Feature Groups

Available feature groups:
- **firstorder**: First-order statistics (mean, variance, skewness, etc.)
- **shape**: Shape features (volume, surface area, sphericity, etc.)
- **glcm**: Gray Level Co-occurrence Matrix features
- **gldm**: Gray Level Dependence Matrix features
- **glrlm**: Gray Level Run Length Matrix features
- **glszm**: Gray Level Size Zone Matrix features
- **ngtdm**: Neighboring Gray Tone Difference Matrix features

## Parameters

### Command Line Arguments

- `--dataset-dir`: Path to Dataset360_oaizib directory
- `--knee-side-csv`: Path to kneeSideInfo.csv
- `--subinfo-train`: Path to subInfo_train.xlsx
- `--subinfo-test`: Path to subInfo_test.xlsx
- `--output-dir`: Output directory for results
- `--split`: Which split to process ('train', 'test', or 'both')
- `--radiomics-groups`: List of radiomics feature groups to extract
- `--device`: PyTorch device (e.g., 'cuda:0', 'cpu')
- `--voxel-array-shift`: Shift to apply to voxel values (default: 0)
- `--bin-width`: Histogram bin width (None for default 256 bins)
- `--output-format`: Output format ('csv' or 'parquet')
- `--load-existing`: Path to existing features file to load

### Class Parameters

- `dataset_dir`: Path to Dataset360_oaizib directory
- `knee_side_csv`: Path to kneeSideInfo.csv
- `subinfo_train`: Path to subInfo_train.xlsx
- `subinfo_test`: Path to subInfo_test.xlsx
- `radiomics_groups`: List of feature groups to extract
- `device`: PyTorch device
- `voxelArrayShift`: Voxel array shift value
- `binWidth`: Histogram bin width
- `split`: 'train' or 'test'

## Data Format

### Input Format

- **Images**: NIfTI format (.nii.gz)
- **Labels**: NIfTI format with integer labels for each ROI
- **kneeSideInfo.csv**: CSV with columns: filename, side
- **subInfo_train/test.xlsx**: Excel files with subject information

### Output Format

The combined data DataFrame contains:
- **case_id**: Case identifier
- **roi_name**: ROI name (e.g., 'Femur', 'Femoral_Cartilage')
- **knee_side**: Knee side ('right' or 'left')
- **Radiomics features**: One column per extracted feature
- **Subject metadata**: Columns from subInfo_train/test.xlsx

## Example Workflow

1. **Extract radiomics features**:
   ```bash
   python Radiomica_KLG_mapping_dataloader.py --split both --output-dir ./output
   ```

2. **Load and prepare data for training**:
   ```python
   loader = RadiomicsKLGDataLoader(...)
   loader.load_radiomics('./output/radiomics_features_train.parquet')
   df = loader.get_training_data()
   ```

3. **Train model**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   
   df_roi = loader.get_roi_specific_data('Femoral_Cartilage')
   X = df_roi.drop(columns=['case_id', 'roi_name', 'KLG'])
   y = df_roi['KLG']
   
   model = RandomForestClassifier()
   model.fit(X, y)
   ```

## Notes

- The script automatically handles case ID matching between files
- Ground truth labels are used (not predicted masks)
- Invalid features (NaN, Inf) are automatically filtered
- Empty ROIs are skipped with a warning
- The script supports both CSV and Parquet output formats

## Dependencies

- torch
- numpy
- pandas
- SimpleITK
- torchradiomics
- openpyxl (for Excel file reading)

## Author

Created for OAI-ZIB-CM dataset radiomics extraction and KLG mapping.

