"""
Example usage of RadiomicsKLGDataLoader

This script demonstrates how to use the RadiomicsKLGDataLoader class
to extract radiomics features and prepare data for KLG mapping model training.
"""

from pathlib import Path
import pandas as pd
from Radiomics_KLG_mapping_dataloader import RadiomicsKLGDataLoader

# Configuration
DATASET_DIR = Path("../Dataset360_oaizib")
KNEE_SIDE_CSV = Path("../OAI-ZIB-CM/info/kneeSideInfo.csv")
SUBINFO_TRAIN = Path("../OAI-ZIB-CM/info/subInfo_train.xlsx")
SUBINFO_TEST = Path("../OAI-ZIB-CM/info/subInfo_test.xlsx")
OUTPUT_DIR = Path("./radiomics_output")

# All available radiomics feature groups
ALL_RADIOMICS_GROUPS = ['firstorder', 'shape', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']

print("="*80)
print("RADIOMICS KLG MODEL TRAINING - DEBUG MODE")
print("="*80)
print(f"\n[DEBUG] Configuration:")
print(f"  - Dataset directory: {DATASET_DIR}")
print(f"  - Output directory: {OUTPUT_DIR}")
print(f"  - Radiomics groups to extract: {ALL_RADIOMICS_GROUPS}")
print(f"  - Total feature groups: {len(ALL_RADIOMICS_GROUPS)}")

# Initialize data loader for training set
print("\n[DEBUG] Initializing training data loader...")
train_loader = RadiomicsKLGDataLoader(
    dataset_dir=DATASET_DIR,
    knee_side_csv=KNEE_SIDE_CSV,
    subinfo_train=SUBINFO_TRAIN,
    subinfo_test=SUBINFO_TEST,
    radiomics_groups=ALL_RADIOMICS_GROUPS,  # Load ALL radiomics feature groups
    split='train'
)

print(f"[DEBUG] Training loader initialized successfully")
print(f"[DEBUG] Device: {train_loader.device}")
print(f"[DEBUG] ROI labels loaded: {train_loader.roi_labels}")
print(f"[DEBUG] Number of training images: {len(train_loader.image_files)}")

# Extract radiomics features (or load from existing file)
print("\n" + "="*80)
print("[DEBUG] Extracting radiomics features for ALL ROIs...")
print("="*80)

if (OUTPUT_DIR / 'radiomics_features_train.parquet').exists():
    print(f"[DEBUG] Found existing features file: {OUTPUT_DIR / 'radiomics_features_train.parquet'}")
    print("[DEBUG] Loading existing features instead of re-extracting...")
    train_loader.load_radiomics(OUTPUT_DIR / 'radiomics_features_train.parquet')
    print("[DEBUG] Features loaded successfully")
else:
    print("[DEBUG] No existing features file found. Extracting from scratch...")
    train_loader.extract_all_radiomics(
        save_path=OUTPUT_DIR / 'radiomics_features_train.parquet'
    )
    print("[DEBUG] Feature extraction completed")

# Get combined training data with metadata
print("\n" + "="*80)
print("[DEBUG] Combining radiomics features with metadata...")
print("="*80)
df_train = train_loader.get_training_data()

print(f"\n[DEBUG] Training data summary:")
print(f"  - Shape: {df_train.shape}")
print(f"  - Total cases: {df_train['case_id'].nunique() if 'case_id' in df_train.columns else 'N/A'}")
print(f"  - Total ROIs: {df_train['roi_name'].nunique() if 'roi_name' in df_train.columns else 'N/A'}")
print(f"  - Total feature columns: {len(df_train.columns)}")

if 'roi_name' in df_train.columns:
    print(f"\n[DEBUG] ROIs found in training data:")
    roi_counts = df_train['roi_name'].value_counts()
    for roi_name, count in roi_counts.items():
        print(f"  - {roi_name}: {count} cases")

print(f"\n[DEBUG] First 15 column names:")
print(f"  {df_train.columns.tolist()[:15]}")
print(f"\n[DEBUG] Sample of training data (first 3 rows):")
print(df_train.head(3))

# Get data for ALL ROIs
print("\n" + "="*80)
print("[DEBUG] Extracting data for ALL ROIs...")
print("="*80)

if 'roi_name' in df_train.columns:
    unique_rois = df_train['roi_name'].unique()
    print(f"[DEBUG] Found {len(unique_rois)} unique ROIs: {list(unique_rois)}")
    
    roi_data_dict = {}
    for roi_name in unique_rois:
        print(f"\n[DEBUG] Processing ROI: {roi_name}")
        df_roi = train_loader.get_roi_specific_data(roi_name)
        roi_data_dict[roi_name] = df_roi
        print(f"  - Shape: {df_roi.shape}")
        print(f"  - Number of cases: {len(df_roi)}")
        print(f"  - Feature columns: {len(df_roi.columns)}")
    
    # Example: Get data for Femoral Cartilage (if it exists)
    if 'Femoral_Cartilage' in roi_data_dict:
        print(f"\n[DEBUG] Femoral Cartilage data summary:")
        df_femoral_cartilage = roi_data_dict['Femoral_Cartilage']
        print(f"  - Shape: {df_femoral_cartilage.shape}")
        print(f"  - Sample columns: {df_femoral_cartilage.columns.tolist()[:10]}")
else:
    print("[DEBUG] Warning: 'roi_name' column not found in training data")

# Example: Prepare data for a specific ROI and target variable
# Assuming subinfo_df has a column like 'KLG' or 'KL_grade'
# You would do:
# df_roi = train_loader.get_roi_specific_data('Femoral_Cartilage')
# X = df_roi.drop(columns=['case_id', 'roi_name', 'KLG'])  # Features
# y = df_roi['KLG']  # Target variable

# Initialize test loader
print("\n" + "="*80)
print("[DEBUG] Initializing test data loader...")
print("="*80)
test_loader = RadiomicsKLGDataLoader(
    dataset_dir=DATASET_DIR,
    knee_side_csv=KNEE_SIDE_CSV,
    subinfo_train=SUBINFO_TRAIN,
    subinfo_test=SUBINFO_TEST,
    radiomics_groups=ALL_RADIOMICS_GROUPS,  # Load ALL radiomics feature groups
    split='test'
)

print(f"[DEBUG] Test loader initialized successfully")
print(f"[DEBUG] Device: {test_loader.device}")
print(f"[DEBUG] ROI labels loaded: {test_loader.roi_labels}")
print(f"[DEBUG] Number of test images: {len(test_loader.image_files)}")

# Load pre-extracted features (if available)
print("\n" + "="*80)
print("[DEBUG] Processing test set features...")
print("="*80)

if (OUTPUT_DIR / 'radiomics_features_test.parquet').exists():
    print(f"[DEBUG] Found existing test features file: {OUTPUT_DIR / 'radiomics_features_test.parquet'}")
    print("[DEBUG] Loading existing test features...")
    test_loader.load_radiomics(OUTPUT_DIR / 'radiomics_features_test.parquet')
    print("[DEBUG] Test features loaded successfully")
else:
    print("[DEBUG] No existing test features file found. Extracting from scratch...")
    test_loader.extract_all_radiomics(
        save_path=OUTPUT_DIR / 'radiomics_features_test.parquet'
    )
    print("[DEBUG] Test feature extraction completed")

# Get test data
print("\n[DEBUG] Combining test radiomics features with metadata...")
df_test = test_loader.get_training_data()

print(f"\n[DEBUG] Test data summary:")
print(f"  - Shape: {df_test.shape}")
print(f"  - Total cases: {df_test['case_id'].nunique() if 'case_id' in df_test.columns else 'N/A'}")
print(f"  - Total ROIs: {df_test['roi_name'].nunique() if 'roi_name' in df_test.columns else 'N/A'}")
print(f"  - Total feature columns: {len(df_test.columns)}")

if 'roi_name' in df_test.columns:
    print(f"\n[DEBUG] ROIs found in test data:")
    roi_counts = df_test['roi_name'].value_counts()
    for roi_name, count in roi_counts.items():
        print(f"  - {roi_name}: {count} cases")

print("\n" + "="*80)
print("[DEBUG] All processing completed successfully!")
print("="*80)
print("\nDone!")

