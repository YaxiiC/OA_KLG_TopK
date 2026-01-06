"""
Data Loading and Preprocessing for KLGrade Prediction

This module contains:
- load_radiomics_long_format: Load radiomics from long format CSV
- load_klgrade_labels: Load KLGrade labels from CSV
- preprocess_image: Load and preprocess NIfTI images
- KLGradeDataset: PyTorch dataset for images + radiomics + labels
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy import ndimage

logger = logging.getLogger(__name__)


def load_radiomics_long_format(
    csv_path: Path,
    expected_rois: Optional[List[str]] = None,
    expected_features: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], List[str], List[str], Dict[str, int]]:
    """
    Load radiomics from long format CSV and build fixed-length vectors.
    
    Args:
        csv_path: Path to CSV with columns: case_id, roi_name, feature_name, value
        expected_rois: Optional list of expected ROI names (for validation)
        expected_features: Optional list of expected feature names (for validation)
    
    Returns:
        Tuple of:
        - case_radiomics: Dict[case_id, np.ndarray] of shape (n_features_total,)
        - roi_names: Sorted list of ROI names
        - feature_names: Sorted list of feature names
        - missing_stats: Dict with missing ROI/feature counts
    """
    logger.info(f"Loading radiomics from {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ["case_id", "roi_name", "feature_name", "value"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    # Get unique ROIs and features
    all_rois = sorted(df["roi_name"].unique())
    all_features = sorted(df["feature_name"].unique())
    
    logger.info(f"Found {len(all_rois)} unique ROIs: {all_rois}")
    logger.info(f"Found {len(all_features)} unique features per ROI")
    
    # Validate expected ROIs/features if provided
    if expected_rois is not None:
        missing_rois = set(expected_rois) - set(all_rois)
        if missing_rois:
            logger.warning(f"Missing expected ROIs: {missing_rois}")
    
    if expected_features is not None:
        missing_features = set(expected_features) - set(all_features)
        if missing_features:
            logger.warning(f"Missing expected features: {missing_features}")
    
    # Build fixed mapping: index -> (roi_name, feature_name)
    feature_mapping = []
    for roi in all_rois:
        for feat in all_features:
            feature_mapping.append(f"{roi}:{feat}")
    
    n_features_total = len(feature_mapping)
    logger.info(f"Total feature vector length: {n_features_total} ({len(all_rois)} ROIs × {len(all_features)} features)")
    
    # Build vectors for each case
    case_radiomics = {}
    missing_stats = {
        "cases_with_missing_rois": 0,
        "cases_with_missing_features": 0,
        "total_missing_values": 0
    }
    
    for case_id in df["case_id"].unique():
        case_df = df[df["case_id"] == case_id]
        
        # Build vector
        vector = np.zeros(n_features_total, dtype=np.float32)
        missing_count = 0
        
        for idx, (roi, feat) in enumerate([(r, f) for r in all_rois for f in all_features]):
            matching = case_df[(case_df["roi_name"] == roi) & (case_df["feature_name"] == feat)]
            if len(matching) > 0:
                value = matching["value"].iloc[0]
                if pd.notna(value):
                    vector[idx] = float(value)
                else:
                    missing_count += 1
            else:
                missing_count += 1
        
        if missing_count > 0:
            missing_stats["total_missing_values"] += missing_count
            # Check if entire ROI is missing
            case_rois = set(case_df["roi_name"].unique())
            missing_rois = set(all_rois) - case_rois
            if missing_rois:
                missing_stats["cases_with_missing_rois"] += 1
            # Check if features are missing
            for roi in case_rois:
                roi_features = set(case_df[case_df["roi_name"] == roi]["feature_name"].unique())
                expected_roi_features = set(all_features)
                if roi_features != expected_roi_features:
                    missing_stats["cases_with_missing_features"] += 1
                    break
        
        case_radiomics[case_id] = vector
    
    logger.info(f"Loaded {len(case_radiomics)} cases")
    logger.info(f"Missing stats: {missing_stats}")
    
    return case_radiomics, all_rois, all_features, missing_stats


def load_klgrade_labels(csv_path: Path) -> Dict[str, int]:
    """
    Load KLGrade labels from CSV.
    
    Args:
        csv_path: Path to CSV with columns: case_id, KLGrade
    
    Returns:
        Dict[case_id, KLGrade] where KLGrade is int 0-4
    """
    logger.info(f"Loading KLGrade labels from {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"KLGrade CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if "case_id" not in df.columns or "KLGrade" not in df.columns:
        raise ValueError("CSV must have columns: case_id, KLGrade")
    
    labels = {}
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        klgrade = int(row["KLGrade"])
        if klgrade < 0 or klgrade > 4:
            logger.warning(f"Invalid KLGrade {klgrade} for case {case_id}, skipping")
            continue
        labels[case_id] = klgrade
    
    logger.info(f"Loaded {len(labels)} KLGrade labels")
    logger.info(f"Label distribution: {pd.Series(list(labels.values())).value_counts().sort_index().to_dict()}")
    
    return labels


def preprocess_image(
    image_path: Path,
    target_shape: Tuple[int, int, int] = (32, 128, 128)
) -> torch.Tensor:
    """
    Load and preprocess NIfTI image.
    
    Args:
        image_path: Path to NIfTI file
        target_shape: Target (D, H, W) shape
    
    Returns:
        Tensor of shape [1, D, H, W] (single channel)
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load with SimpleITK
    sitk_img = sitk.ReadImage(str(image_path))
    img_array = sitk.GetArrayFromImage(sitk_img)  # [z, y, x]
    
    # Z-score normalize per volume
    img_array = img_array.astype(np.float32)
    mean = img_array.mean()
    std = img_array.std()
    if std > 0:
        img_array = (img_array - mean) / std
    else:
        logger.warning(f"Zero std for {image_path.name}, skipping normalization")
    
    # Resize to target shape
    current_shape = img_array.shape
    if current_shape != target_shape:
        # Use scipy.ndimage.zoom for resizing
        zoom_factors = [
            target_shape[0] / current_shape[0],
            target_shape[1] / current_shape[1],
            target_shape[2] / current_shape[2]
        ]
        img_array = ndimage.zoom(img_array, zoom_factors, order=1, mode='nearest')
    
    # Convert to tensor and add channel dimension
    tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, D, H, W]
    
    return tensor


class KLGradeDataset(Dataset):
    """Dataset for KLGrade prediction with images and radiomics."""
    
    def __init__(
        self,
        case_ids: List[str],
        images_dir: Path,
        radiomics_dict: Dict[str, np.ndarray],
        labels_dict: Optional[Dict[str, int]] = None,
        target_shape: Tuple[int, int, int] = (32, 128, 128),
        transform: Optional[callable] = None
    ):
        """
        Args:
            case_ids: List of case IDs
            images_dir: Directory containing NIfTI images
            radiomics_dict: Dict[case_id, radiomics_vector]
            labels_dict: Optional Dict[case_id, KLGrade]
            target_shape: Target image shape (D, H, W)
            transform: Optional transform function
        """
        self.case_ids = case_ids
        self.images_dir = images_dir
        self.radiomics_dict = radiomics_dict
        self.labels_dict = labels_dict
        self.target_shape = target_shape
        self.transform = transform
        
        # Validate all cases have radiomics
        missing_radiomics = [cid for cid in case_ids if cid not in radiomics_dict]
        if missing_radiomics:
            logger.warning(f"Missing radiomics for {len(missing_radiomics)} cases")
    
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        
        # Load image
        # Try different naming conventions
        image_paths = [
            self.images_dir / f"{case_id}_0000.nii.gz",
            self.images_dir / f"{case_id}_0000.nii",
            self.images_dir / f"{case_id}.nii.gz",
            self.images_dir / f"{case_id}.nii"
        ]
        
        image_path = None
        for path in image_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"Image not found for case {case_id} in {self.images_dir}")
        
        image = preprocess_image(image_path, self.target_shape)
        
        if self.transform:
            image = self.transform(image)
        
        # Get radiomics
        default_length = len(next(iter(self.radiomics_dict.values()))) if self.radiomics_dict else 535
        radiomics = self.radiomics_dict.get(case_id, np.zeros(default_length, dtype=np.float32))
        radiomics = torch.from_numpy(radiomics).float()
        
        # Get label
        if self.labels_dict is not None:
            label = self.labels_dict.get(case_id, 0)
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)  # Dummy label for inference
        
        return {
            "case_id": case_id,
            "image": image,
            "radiomics": radiomics,
            "label": label
        }

