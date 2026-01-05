"""
Radiomics KLG Mapping DataLoader

This script:
1. Extracts radiomics features from ground truth ROIs in Dataset360_oaizib
2. Loads knee side information from kneeSideInfo.csv
3. Loads subject information from subInfo_train.xlsx and subInfo_test.xlsx
4. Combines all information for model training

Author: Created for OAI-ZIB-CM dataset
"""

import torch
import numpy as np
import math
import logging
import warnings
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict
import pandas as pd
import csv

import SimpleITK as sitk

# Import reusable functions from torchradiomics_from_ROIs.py
from torchradiomics_from_ROIs import (
    extract_radiomics_with_groups,
    extract_rois
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Load Metadata
# ============================================================================

def load_knee_side_info(csv_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load knee side information from CSV file.
    
    Args:
        csv_path: Path to kneeSideInfo.csv
    
    Returns:
        Dict mapping filename -> side ('right' or 'left')
    """
    side_info = {}
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        logger.warning(f"Knee side info file not found: {csv_path}")
        return side_info
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                filename = row[0].strip()
                side = row[1].strip().lower()
                side_info[filename] = side
    
    logger.info(f"Loaded knee side info for {len(side_info)} cases")
    return side_info


def load_subject_info(excel_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load subject information from Excel file.
    
    Args:
        excel_path: Path to subInfo_train.xlsx or subInfo_test.xlsx
    
    Returns:
        DataFrame with subject information
    """
    excel_path = Path(excel_path)
    
    if not excel_path.exists():
        logger.warning(f"Subject info file not found: {excel_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded subject info from {excel_path.name}: {len(df)} cases, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Failed to load subject info from {excel_path}: {e}")
        return pd.DataFrame()


def extract_case_id_from_filename(filename: str) -> str:
    """
    Extract case ID from filename.
    Handles formats like: oaizib_001.nii.gz, oaizib_001_0000.nii.gz
    
    Returns:
        Case ID (e.g., 'oaizib_001')
    """
    # Remove extensions
    name = filename.replace('.nii.gz', '').replace('.nii', '')
    # Remove _0000 suffix if present (nnU-Net format)
    if name.endswith('_0000'):
        name = name[:-5]
    return name


# ============================================================================
# Process Single Case
# ============================================================================

def process_case_gt(
    case_image_path: Union[str, Path],
    case_label_path: Union[str, Path],
    roi_labels: Dict[int, str] = None,
    radiomics_groups: List[str] = None,
    device: torch.device = None,
    voxelArrayShift: int = 0,
    binWidth: Optional[float] = None
) -> pd.DataFrame:
    """
    Process a single case: extract radiomics from ground truth ROIs.
    
    Args:
        case_image_path: Path to input image (NIfTI)
        case_label_path: Path to ground truth label (NIfTI)
        roi_labels: Dict mapping label value -> ROI name
        radiomics_groups: List of radiomics feature groups to extract
        device: PyTorch device
        voxelArrayShift: Voxel array shift to apply
        binWidth: Histogram bin width
    
    Returns:
        DataFrame with columns: case_id, roi_name, feature_name, value
    """
    case_image_path = Path(case_image_path)
    case_label_path = Path(case_label_path)
    
    case_id = extract_case_id_from_filename(case_image_path.name)
    logger.info(f"Processing case: {case_id}")
    
    # Load image
    sitk_image = sitk.ReadImage(str(case_image_path))
    image_array = sitk.GetArrayFromImage(sitk_image)
    image_tensor = torch.from_numpy(image_array).float()
    
    # Load ground truth label
    sitk_label = sitk.ReadImage(str(case_label_path))
    label_array = sitk.GetArrayFromImage(sitk_label)
    
    # Get spacing (SITK order is x,y,z, but we need z,y,x)
    spacing_xyz = sitk_image.GetSpacing()
    spacing_zyx = [spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]]
    
    # Extract ROIs
    rois = extract_rois(label_array, roi_labels=roi_labels)
    
    if len(rois) == 0:
        logger.warning(f"No valid ROIs found for case {case_id}")
        return pd.DataFrame(columns=['case_id', 'roi_name', 'feature_name', 'value'])
    
    # Extract radiomics for each ROI
    results = []
    
    for roi_name, roi_mask in rois.items():
        logger.info(f"  Extracting radiomics for {roi_name}...")
        
        roi_mask_tensor = torch.from_numpy(roi_mask).float()
        
        try:
            features_dict, feature_names = extract_radiomics_with_groups(
                image_tensor,
                roi_mask_tensor,
                voxelArrayShift=voxelArrayShift,
                pixelSpacing=spacing_zyx,
                binWidth=binWidth,
                groups=radiomics_groups,
                device=device
            )
            
            # Convert to DataFrame rows
            for feat_name, feat_value in features_dict.items():
                if isinstance(feat_value, torch.Tensor):
                    feat_value = feat_value.item()
                
                # Check for invalid values
                if math.isnan(feat_value) or math.isinf(feat_value):
                    logger.warning(f"  Invalid feature {feat_name} for {roi_name}: {feat_value}")
                    continue
                
                results.append({
                    'case_id': case_id,
                    'roi_name': roi_name,
                    'feature_name': feat_name,
                    'value': feat_value
                })
        
        except Exception as e:
            logger.error(f"  Failed to extract radiomics for {roi_name}: {e}")
            continue
    
    return pd.DataFrame(results)


# ============================================================================
# Main DataLoader Class
# ============================================================================

class RadiomicsKLGDataLoader:
    """
    DataLoader for radiomics features with KLG mapping.
    
    Combines:
    - Radiomics features extracted from ground truth ROIs
    - Knee side information
    - Subject information (train/test split)
    """
    
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        knee_side_csv: Union[str, Path],
        subinfo_train: Union[str, Path],
        subinfo_test: Union[str, Path],
        radiomics_groups: List[str] = None,
        device: torch.device = None,
        voxelArrayShift: int = 0,
        binWidth: Optional[float] = None,
        split: str = 'train'
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset_dir: Path to Dataset360_oaizib directory
            knee_side_csv: Path to kneeSideInfo.csv
            subinfo_train: Path to subInfo_train.xlsx
            subinfo_test: Path to subInfo_test.xlsx
            radiomics_groups: List of radiomics feature groups to extract
            device: PyTorch device
            voxelArrayShift: Voxel array shift to apply
            binWidth: Histogram bin width
            split: 'train' or 'test'
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.radiomics_groups = radiomics_groups or ['firstorder', 'shape', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.voxelArrayShift = voxelArrayShift
        self.binWidth = binWidth
        
        # Load dataset.json for ROI labels
        dataset_json_path = self.dataset_dir / 'dataset.json'
        self.roi_labels = None
        if dataset_json_path.exists():
            import json
            with open(dataset_json_path, 'r') as f:
                dataset_json = json.load(f)
                if 'labels' in dataset_json:
                    # Reverse mapping: label value -> name
                    self.roi_labels = {v: k.replace(' ', '_') for k, v in dataset_json['labels'].items() if v > 0}
                    logger.info(f"ROI labels: {self.roi_labels}")
        
        # Load metadata
        self.knee_side_info = load_knee_side_info(knee_side_csv)
        if split == 'train':
            self.subinfo_df = load_subject_info(subinfo_train)
        else:
            self.subinfo_df = load_subject_info(subinfo_test)
        
        # Get image and label paths
        if split == 'train':
            self.images_dir = self.dataset_dir / 'imagesTr'
            self.labels_dir = self.dataset_dir / 'labelsTr'
        else:
            self.images_dir = self.dataset_dir / 'imagesTs'
            self.labels_dir = self.dataset_dir / 'labelsTs'
        
        # Find all image files
        self.image_files = sorted(self.images_dir.glob('*.nii.gz')) + sorted(self.images_dir.glob('*.nii'))
        logger.info(f"Found {len(self.image_files)} {split} images")
        
        # Cache for radiomics features
        self.radiomics_cache = {}
        self.features_df = None
    
    def extract_all_radiomics(self, save_path: Optional[Union[str, Path]] = None):
        """
        Extract radiomics features for all cases.
        
        Args:
            save_path: Optional path to save extracted features (CSV/Parquet)
        """
        logger.info(f"Extracting radiomics features for {len(self.image_files)} cases...")
        
        all_results = []
        
        for idx, image_path in enumerate(self.image_files, 1):
            logger.info(f"\n[{idx}/{len(self.image_files)}] Processing {image_path.name}")
            
            # Find corresponding label file
            label_path = self.labels_dir / image_path.name
            
            if not label_path.exists():
                # Try removing _0000 suffix
                label_name = image_path.name.replace('_0000.nii.gz', '.nii.gz')
                label_path = self.labels_dir / label_name
            
            if not label_path.exists():
                logger.warning(f"Label file not found for {image_path.name}")
                continue
            
            try:
                case_results = process_case_gt(
                    image_path,
                    label_path,
                    roi_labels=self.roi_labels,
                    radiomics_groups=self.radiomics_groups,
                    device=self.device,
                    voxelArrayShift=self.voxelArrayShift,
                    binWidth=self.binWidth
                )
                
                if len(case_results) > 0:
                    all_results.append(case_results)
                else:
                    logger.warning(f"No results for {image_path.name}")
            
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
                continue
        
        if len(all_results) == 0:
            logger.error("No results to save!")
            return
        
        self.features_df = pd.concat(all_results, ignore_index=True)
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix == '.parquet':
                self.features_df.to_parquet(save_path, index=False)
            else:
                self.features_df.to_csv(save_path, index=False)
            
            logger.info(f"Saved radiomics features to {save_path}")
        
        logger.info(f"\nTotal cases: {len(self.features_df['case_id'].unique())}")
        logger.info(f"Total ROIs: {len(self.features_df.groupby(['case_id', 'roi_name']))}")
        logger.info(f"Total features: {len(self.features_df['feature_name'].unique())}")
    
    def load_radiomics(self, features_path: Union[str, Path]):
        """
        Load pre-extracted radiomics features.
        
        Args:
            features_path: Path to saved radiomics features (CSV/Parquet)
        """
        features_path = Path(features_path)
        
        if features_path.suffix == '.parquet':
            self.features_df = pd.read_parquet(features_path)
        else:
            self.features_df = pd.read_csv(features_path)
        
        logger.info(f"Loaded radiomics features from {features_path}")
        logger.info(f"Total cases: {len(self.features_df['case_id'].unique())}")
    
    def get_training_data(self, target_column: str = None) -> pd.DataFrame:
        """
        Get combined training data with radiomics, side info, and subject info.
        
        Args:
            target_column: Column name in subinfo_df to use as target variable
        
        Returns:
            DataFrame with all features and metadata
        """
        if self.features_df is None:
            raise ValueError("Radiomics features not extracted. Call extract_all_radiomics() or load_radiomics() first.")
        
        # Convert to wide format (one row per case+ROI)
        df_wide = self.features_df.pivot_table(
            index=['case_id', 'roi_name'],
            columns='feature_name',
            values='value'
        ).reset_index()
        
        # Add knee side information
        df_wide['knee_side'] = df_wide['case_id'].apply(
            lambda x: self.knee_side_info.get(f"{x}.nii.gz", "unknown")
        )
        
        # Merge with subject information
        # Extract case ID from subinfo_df (assuming first column or a specific column contains case IDs)
        if len(self.subinfo_df) > 0:
            # Try to find case ID column
            case_id_col = None
            for col in self.subinfo_df.columns:
                if 'case' in col.lower() or 'id' in col.lower() or 'file' in col.lower():
                    case_id_col = col
                    break
            
            if case_id_col:
                # Normalize case IDs in subinfo_df
                self.subinfo_df['case_id_normalized'] = self.subinfo_df[case_id_col].apply(
                    lambda x: extract_case_id_from_filename(str(x))
                )
                
                # Merge
                df_wide['case_id_normalized'] = df_wide['case_id']
                df_wide = df_wide.merge(
                    self.subinfo_df,
                    left_on='case_id_normalized',
                    right_on='case_id_normalized',
                    how='left'
                )
                df_wide = df_wide.drop(columns=['case_id_normalized'])
            else:
                logger.warning("Could not find case ID column in subinfo_df. Skipping merge.")
        
        return df_wide
    
    def get_roi_specific_data(self, roi_name: str, target_column: str = None) -> pd.DataFrame:
        """
        Get data for a specific ROI.
        
        Args:
            roi_name: Name of ROI (e.g., 'Femur', 'Femoral_Cartilage')
            target_column: Column name in subinfo_df to use as target variable
        
        Returns:
            DataFrame filtered for the specified ROI
        """
        df_all = self.get_training_data(target_column=target_column)
        df_roi = df_all[df_all['roi_name'] == roi_name].copy()
        return df_roi


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract radiomics features from ground truth ROIs and prepare for KLG mapping',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Path to Dataset360_oaizib directory'
    )
    
    parser.add_argument(
        '--knee-side-csv',
        type=str,
        required=True,
        help='Path to kneeSideInfo.csv'
    )
    
    parser.add_argument(
        '--subinfo-train',
        type=str,
        required=True,
        help='Path to subInfo_train.xlsx'
    )
    
    parser.add_argument(
        '--subinfo-test',
        type=str,
        required=True,
        help='Path to subInfo_test.xlsx'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for extracted features'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'both'],
        default='both',
        help='Which split to process'
    )
    
    parser.add_argument(
        '--radiomics-groups',
        type=str,
        nargs='+',
        default=['firstorder', 'shape', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'],
        choices=['firstorder', 'shape', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'],
        help='Radiomics feature groups to extract'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='PyTorch device (e.g., cuda:0, cpu). Auto-detected if not specified'
    )
    
    parser.add_argument(
        '--voxel-array-shift',
        type=int,
        default=0,
        help='Voxel array shift to apply (to avoid negative values)'
    )
    
    parser.add_argument(
        '--bin-width',
        type=float,
        default=None,
        help='Histogram bin width (None for default 256 bins)'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        default='parquet',
        choices=['csv', 'parquet'],
        help='Output format for results'
    )
    
    parser.add_argument(
        '--load-existing',
        type=str,
        default=None,
        help='Path to existing radiomics features file to load instead of extracting'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Radiomics groups: {args.radiomics_groups}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_to_process = []
    if args.split == 'both':
        splits_to_process = ['train', 'test']
    else:
        splits_to_process = [args.split]
    
    for split in splits_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split.upper()} split")
        logger.info(f"{'='*60}")
        
        # Initialize data loader
        loader = RadiomicsKLGDataLoader(
            dataset_dir=args.dataset_dir,
            knee_side_csv=args.knee_side_csv,
            subinfo_train=args.subinfo_train,
            subinfo_test=args.subinfo_test,
            radiomics_groups=args.radiomics_groups,
            device=device,
            voxelArrayShift=args.voxel_array_shift,
            binWidth=args.bin_width,
            split=split
        )
        
        # Extract or load radiomics
        if args.load_existing and Path(args.load_existing).exists():
            logger.info(f"Loading existing features from {args.load_existing}")
            loader.load_radiomics(args.load_existing)
        else:
            features_file = output_dir / f'radiomics_features_{split}.{args.output_format}'
            loader.extract_all_radiomics(save_path=features_file)
        
        # Get combined training data
        logger.info("\nCombining radiomics with metadata...")
        df_combined = loader.get_training_data()
        
        # Save combined data
        combined_file = output_dir / f'radiomics_klg_data_{split}.{args.output_format}'
        if args.output_format == 'parquet':
            df_combined.to_parquet(combined_file, index=False)
        else:
            df_combined.to_csv(combined_file, index=False)
        
        logger.info(f"Saved combined data to {combined_file}")
        logger.info(f"Shape: {df_combined.shape}")
        logger.info(f"Columns: {df_combined.columns.tolist()[:10]}...")  # Show first 10 columns
        
        # Also save wide format for each ROI
        for roi_name in df_combined['roi_name'].unique():
            df_roi = loader.get_roi_specific_data(roi_name)
            roi_file = output_dir / f'radiomics_klg_data_{split}_{roi_name}.{args.output_format}'
            if args.output_format == 'parquet':
                df_roi.to_parquet(roi_file, index=False)
            else:
                df_roi.to_csv(roi_file, index=False)
            logger.info(f"Saved {roi_name} data to {roi_file} ({len(df_roi)} cases)")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()

