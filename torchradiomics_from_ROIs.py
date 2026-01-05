"""
PyTorch Radiomics Extraction with nnU-Net Ensemble Prediction

This script:
1. Loads trained nnU-Net models from multiple folds
2. Performs ensemble prediction (averaging probabilities across folds)
3. Extracts ROIs from the predicted segmentation mask
4. Extracts configurable radiomics features for each ROI
5. Saves results in CSV/Parquet format

Author: Refactored for nnU-Net ensemble + multi-ROI radiomics extraction
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

import SimpleITK as sitk

# nnU-Net imports
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from batchgenerators.utilities.file_and_folder_operations import load_json, join

from torchradiomics import (
    TorchRadiomicsGLCM,
    TorchRadiomicsGLDM,
    TorchRadiomicsGLRLM,
    TorchRadiomicsGLSZM,
    TorchRadiomicsNGTDM,
    TorchRadiomicsFirstOrder,
    inject_torch_radiomics
)

logging.getLogger("torchradiomics").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Radiomics Feature Extraction (Using torchradiomics extractors)
# ============================================================================


def extract_radiomics_with_groups(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    voxelArrayShift: int = 0,
    pixelSpacing: List[float] = [1.0, 1.0, 1.0],
    binWidth: Optional[float] = None,
    groups: List[str] = None,
    device: torch.device = None
) -> Tuple[Dict, List[str]]:
    """
    Extract radiomics features with configurable groups using torchradiomics extractors.
    
    Args:
        image_tensor: Image tensor (3D)
        mask_tensor: Binary mask tensor (3D)
        voxelArrayShift: Shift to apply to avoid negative values
        pixelSpacing: Voxel spacing [z, y, x]
        binWidth: Histogram bin width (None for default 256 bins)
        groups: List of feature groups to extract. Options:
            - 'firstorder': First-order statistics (using TorchRadiomicsFirstOrder)
            - 'shape': Shape features (using TorchRadiomicsFirstOrder - shape features included)
            - 'glcm': Gray Level Co-occurrence Matrix
            - 'gldm': Gray Level Dependence Matrix
            - 'glrlm': Gray Level Run Length Matrix
            - 'glszm': Gray Level Size Zone Matrix
            - 'ngtdm': Neighboring Gray Tone Difference Matrix
        device: PyTorch device
    
    Returns:
        Tuple of (features_dict, feature_names_list)
    """
    if groups is None:
        groups = ['firstorder', 'shape', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']
    
    if device is None:
        device = image_tensor.device
    
    # Extract masked ROI values
    roi_values = image_tensor[mask_tensor > 0]
    if roi_values.numel() == 0:
        logger.warning("Empty ROI mask - returning empty features")
        return {}, []
    
    # Create masked image tensor (set background to 0)
    masked_image = image_tensor.clone()
    masked_image[mask_tensor == 0] = 0
    
    # Apply voxel array shift if needed
    if voxelArrayShift != 0:
        masked_image = masked_image + voxelArrayShift
    
    # Convert to SimpleITK format for torchradiomics
    img_np = masked_image.to(dtype=torch.float64, device=device).cpu().numpy()
    mask_np = mask_tensor.to(dtype=torch.uint8, device=device).cpu().numpy()
    sitk_img = sitk.GetImageFromArray(img_np)
    sitk_mask = sitk.GetImageFromArray(mask_np)
    # SITK spacing expects (x,y,z) - reverse order
    sitk_img.SetSpacing((pixelSpacing[2], pixelSpacing[1], pixelSpacing[0]))
    sitk_mask.SetSpacing((pixelSpacing[2], pixelSpacing[1], pixelSpacing[0]))
    
    # Inject defaults & build extractors
    inject_torch_radiomics()
    base_kwargs = dict(
        voxelBased=False,
        padDistance=1,
        kernelRadius=1,
        maskedKernel=False,
        voxelBatch=512,
        dtype=torch.float64,
        device=device
    )
    
    # Configure binning if specified
    if binWidth is not None:
        # Calculate number of bins from binWidth
        roi_values_np = img_np[mask_np > 0]
        if len(roi_values_np) > 0:
            min_val, max_val = roi_values_np.min(), roi_values_np.max()
            num_bins = int((max_val - min_val) / binWidth) + 1
            base_kwargs['binCount'] = num_bins
    
    features_dict = {}
    feature_names = []
    
    # Shape feature keywords for filtering
    shape_keywords = ['Volume', 'Surface', 'Sphericity', 'Compactness', 
                     'Flatness', 'Elongation', 'AxisLength', 'Diameter']
    
    # Extract first-order and/or shape features using TorchRadiomicsFirstOrder
    # (shape features are included in FirstOrder extractor)
    if 'firstorder' in groups or 'shape' in groups:
        try:
            fo_extractor = TorchRadiomicsFirstOrder(sitk_img, sitk_mask, **base_kwargs)
            fo_feats = fo_extractor.execute()
            
            for k, v in fo_feats.items():
                # Skip feature maps (SimpleITK Image) and non-numerical entries
                if isinstance(v, sitk.Image):
                    continue
                
                # Determine if this is a shape feature
                is_shape_feature = any(keyword in k for keyword in shape_keywords)
                
                # Include based on requested groups
                include_feature = False
                if 'firstorder' in groups and not is_shape_feature:
                    include_feature = True
                elif 'shape' in groups and is_shape_feature:
                    include_feature = True
                
                if include_feature:
                    # Convert scalar to tensor
                    features_dict[k] = torch.as_tensor(v, device=device) \
                                      if not isinstance(v, torch.Tensor) else v
                    feature_names.append(k)
        except Exception as e:
            logger.warning(f"Failed to extract first-order/shape features: {e}")
    
    # Texture features (GLCM, GLDM, GLRLM, GLSZM, NGTDM)
    texture_groups = ['glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']
    extractors_map = {
        'glcm': TorchRadiomicsGLCM,
        'gldm': TorchRadiomicsGLDM,
        'glrlm': TorchRadiomicsGLRLM,
        'glszm': TorchRadiomicsGLSZM,
        'ngtdm': TorchRadiomicsNGTDM,
    }
    
    for group in texture_groups:
        if group in groups:
            try:
                extractor = extractors_map[group](sitk_img, sitk_mask, **base_kwargs)
                feats = extractor.execute()
                for k, v in feats.items():
                    # Skip feature maps (SimpleITK Image) and non-numerical entries
                    if isinstance(v, sitk.Image):
                        continue
                    # Convert scalar to tensor
                    features_dict[k] = torch.as_tensor(v, device=device) \
                                      if not isinstance(v, torch.Tensor) else v
                    feature_names.append(k)
            except Exception as e:
                logger.warning(f"Failed to extract {group} features: {e}")
    
    return features_dict, feature_names


# ============================================================================
# nnU-Net Ensemble Prediction
# ============================================================================

def predict_mask_ensemble(
    case_image_path: Union[str, Path],
    model_dir: Union[str, Path],
    folds: List[int] = [0, 1, 2, 3, 4],
    device: torch.device = None,
    checkpoint_name: str = 'checkpoint_best.pth',
    save_mask_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Predict segmentation mask using ensemble of nnU-Net folds.
    
    Args:
        case_image_path: Path to input image (NIfTI)
        model_dir: Path to nnU-Net model directory (contains fold_X subdirs)
        folds: List of fold indices to ensemble
        device: PyTorch device
        checkpoint_name: Checkpoint filename (e.g., 'checkpoint_best.pth')
        save_mask_path: Optional path to save predicted mask
    
    Returns:
        Tuple of (segmentation_array, image_properties_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading nnU-Net predictor for folds {folds}")
    
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=False  # Disable tqdm for cleaner logs
    )
    
    # Initialize from trained model folder
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=folds,
        checkpoint_name=checkpoint_name
    )
    
    # Use nnU-Net's predict_from_files which handles everything automatically
    # We'll use a temporary approach: predict and read back, or use the return value
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    image_reader = SimpleITKIO()
    
    # Read image to get properties
    image_data, image_properties = image_reader.read_images([str(case_image_path)])
    
    # Preprocess
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    preprocessed_data, _, data_properties = preprocessor.run_case_npy(
        image_data,
        None,  # no segmentation from previous stage
        image_properties,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json
    )
    
    # Convert to tensor
    preprocessed_tensor = torch.from_numpy(preprocessed_data).to(
        dtype=torch.float32,
        device=device,
        memory_format=torch.contiguous_format
    )
    
    # Predict logits (ensemble averaging happens internally when multiple folds are loaded)
    logger.info("Running inference...")
    predicted_logits = predictor.predict_logits_from_preprocessed_data(preprocessed_tensor)
    
    # Convert logits to segmentation
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits.cpu(),
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.label_manager,
        data_properties,
        return_probabilities=False
    )
    
    # Ensure segmentation is 3D numpy array
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    
    # Save if requested
    if save_mask_path:
        save_mask_path = Path(save_mask_path)
        save_mask_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure data_properties has sitk_stuff for writing
        # The segmentation is already in original space after convert_predicted_logits_to_segmentation_with_correct_shape
        if 'sitk_stuff' not in data_properties and 'sitk_stuff' in image_properties:
            data_properties['sitk_stuff'] = image_properties['sitk_stuff']
        image_reader.write_seg(segmentation, str(save_mask_path), data_properties)
        logger.info(f"Saved predicted mask to {save_mask_path}")
    
    # Return original image properties for radiomics extraction
    return segmentation, image_properties


# ============================================================================
# ROI Extraction
# ============================================================================

def extract_rois(
    segmentation_mask: np.ndarray,
    roi_labels: Dict[int, str] = None,
    min_voxels: int = 1
) -> Dict[str, np.ndarray]:
    """
    Extract binary masks for each ROI from segmentation.
    
    Args:
        segmentation_mask: Segmentation array with label values
        roi_labels: Dict mapping label value -> ROI name (e.g., {1: 'Femur', 2: 'Femoral_Cartilage'})
                   If None, uses 'roi_1', 'roi_2', etc.
        min_voxels: Minimum number of voxels required for ROI to be valid
    
    Returns:
        Dict mapping ROI name -> binary mask array
    """
    unique_labels = np.unique(segmentation_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    rois = {}
    
    for label_val in unique_labels:
        # Generate ROI name
        if roi_labels and label_val in roi_labels:
            roi_name = roi_labels[label_val]
        else:
            roi_name = f"roi_{int(label_val)}"
        
        # Create binary mask
        binary_mask = (segmentation_mask == label_val).astype(np.uint8)
        
        # Check if ROI has enough voxels
        if binary_mask.sum() >= min_voxels:
            rois[roi_name] = binary_mask
        else:
            logger.warning(f"ROI {roi_name} (label {label_val}) has only {binary_mask.sum()} voxels, skipping")
    
    return rois


# ============================================================================
# Main Workflow
# ============================================================================

def process_case_from_predicted_mask(
    case_image_path: Union[str, Path],
    predicted_mask_path: Union[str, Path],
    roi_labels: Dict[int, str] = None,
    radiomics_groups: List[str] = None,
    device: torch.device = None,
    voxelArrayShift: int = 0,
    binWidth: Optional[float] = None
) -> pd.DataFrame:
    """
    Extract radiomics from a case using a pre-saved predicted mask.
    This function is for use in a separate environment (radiomics extraction only).
    
    Args:
        case_image_path: Path to input image (NIfTI)
        predicted_mask_path: Path to predicted segmentation mask (NIfTI)
        roi_labels: Dict mapping label value -> ROI name
        radiomics_groups: List of radiomics feature groups to extract
        device: PyTorch device
        voxelArrayShift: Voxel array shift to apply
        binWidth: Histogram bin width
    
    Returns:
        DataFrame with columns: case_id, roi_name, feature_name, value
    """
    case_image_path = Path(case_image_path)
    predicted_mask_path = Path(predicted_mask_path)
    
    case_id = case_image_path.stem.replace('.nii', '').replace('.gz', '').replace('_0000', '')
    logger.info(f"Processing case: {case_id}")
    
    # Load predicted mask
    sitk_mask = sitk.ReadImage(str(predicted_mask_path))
    segmentation = sitk.GetArrayFromImage(sitk_mask)
    
    # Extract ROIs
    rois = extract_rois(segmentation, roi_labels=roi_labels)
    
    if len(rois) == 0:
        logger.warning(f"No valid ROIs found for case {case_id}")
        return pd.DataFrame(columns=['case_id', 'roi_name', 'feature_name', 'value'])
    
    # Load original image for radiomics
    sitk_image = sitk.ReadImage(str(case_image_path))
    image_array = sitk.GetArrayFromImage(sitk_image)
    image_tensor = torch.from_numpy(image_array).float()
    
    # Get spacing (SITK order is x,y,z, but we need z,y,x)
    spacing_xyz = sitk_image.GetSpacing()
    spacing_zyx = [spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]]
    
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


def process_case(
    case_image_path: Union[str, Path],
    model_dir: Union[str, Path],
    output_dir: Union[str, Path],
    folds: List[int] = [0, 1, 2, 3, 4],
    radiomics_groups: List[str] = None,
    device: torch.device = None,
    save_masks: bool = False,
    roi_labels: Dict[int, str] = None,
    voxelArrayShift: int = 0,
    binWidth: Optional[float] = None
) -> pd.DataFrame:
    """
    Process a single case: predict mask, extract ROIs, extract radiomics.
    
    Returns:
        DataFrame with columns: case_id, roi_name, feature_name, value
    """
    case_image_path = Path(case_image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    case_id = case_image_path.stem.replace('.nii', '').replace('.gz', '')
    logger.info(f"Processing case: {case_id}")
    
    # 1. Predict segmentation mask
    mask_save_path = output_dir / f"{case_id}_pred_mask.nii.gz" if save_masks else None
    segmentation, data_properties = predict_mask_ensemble(
        case_image_path,
        model_dir,
        folds=folds,
        device=device,
        save_mask_path=mask_save_path
    )
    
    # 2. Extract ROIs
    rois = extract_rois(segmentation, roi_labels=roi_labels)
    
    if len(rois) == 0:
        logger.warning(f"No valid ROIs found for case {case_id}")
        return pd.DataFrame(columns=['case_id', 'roi_name', 'feature_name', 'value'])
    
    # 3. Load original image for radiomics
    import SimpleITK as sitk
    sitk_image = sitk.ReadImage(str(case_image_path))
    image_array = sitk.GetArrayFromImage(sitk_image)
    image_tensor = torch.from_numpy(image_array).float()
    
    # Get spacing (SITK order is x,y,z, but we need z,y,x)
    spacing_xyz = sitk_image.GetSpacing()
    spacing_zyx = [spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]]
    
    # 4. Extract radiomics for each ROI
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


def main():
    parser = argparse.ArgumentParser(
        description='Extract radiomics features from nnU-Net predicted ROIs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory containing input images (NIfTI format)'
    )
    
    parser.add_argument(
        '--nnunet-model-dir',
        type=str,
        required=True,
        help='Path to nnU-Net model directory (contains fold_X subdirectories)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results and predicted masks'
    )
    
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='Folds to use for ensemble (default: all 5 folds)'
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
        '--checkpoint-name',
        type=str,
        default='checkpoint_best.pth',
        help='Checkpoint filename to load'
    )
    
    parser.add_argument(
        '--save-masks',
        action='store_true',
        help='Save predicted segmentation masks to disk'
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
        default='csv',
        choices=['csv', 'parquet'],
        help='Output format for results'
    )
    
    # New arguments for extracting from pre-saved predicted masks
    parser.add_argument(
        '--predicted-masks-dir',
        type=str,
        default=None,
        help='Directory containing pre-saved predicted masks (NIfTI format). If provided, extracts radiomics from these masks instead of running nnU-Net inference. Can be root directory with train/test subdirs (from nnunet_segmentation_inference.py) or flat directory.'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='inference',
        choices=['inference', 'extract_from_masks'],
        help='Mode: inference (run nnU-Net) or extract_from_masks (extract from pre-saved masks)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'both'],
        default='both',
        help='Which split to process when extracting from predicted masks (train, test, or both). Only used if masks are in train/test subdirectories.'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Radiomics groups: {args.radiomics_groups}")
    logger.info(f"Using folds: {args.folds}")
    
    # Load dataset.json to get ROI label mapping
    dataset_json_path = Path(args.nnunet_model_dir) / 'dataset.json'
    roi_labels = None
    if dataset_json_path.exists():
        dataset_json = load_json(str(dataset_json_path))
        if 'labels' in dataset_json:
            # Reverse mapping: label value -> name
            roi_labels = {v: k.replace(' ', '_') for k, v in dataset_json['labels'].items() if v > 0}
            logger.info(f"ROI labels: {roi_labels}")
    
    # Find all images
    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob('*.nii.gz')) + sorted(images_dir.glob('*.nii'))
    
    if len(image_files) == 0:
        logger.error(f"No images found in {images_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each case
    all_results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mode: extract from pre-saved predicted masks
    if args.mode == 'extract_from_masks' or args.predicted_masks_dir:
        if not args.predicted_masks_dir:
            raise ValueError("--predicted-masks-dir is required when using extract_from_masks mode")
        
        predicted_masks_dir = Path(args.predicted_masks_dir)
        logger.info(f"Extracting radiomics from pre-saved predicted masks in {predicted_masks_dir}")
        
        # Check if masks are organized in train/test subdirectories (from nnunet_segmentation_inference.py)
        train_dir = predicted_masks_dir / 'train'
        test_dir = predicted_masks_dir / 'test'
        has_subdirs = train_dir.exists() or test_dir.exists()
        
        # Determine which splits to search
        splits_to_search = []
        if has_subdirs:
            if args.split == 'both':
                splits_to_search = ['train', 'test']
            else:
                splits_to_search = [args.split]
        else:
            splits_to_search = [None]  # Search in root directory
        
        for idx, image_path in enumerate(image_files, 1):
            logger.info(f"\n[{idx}/{len(image_files)}] Processing {image_path.name}")
            
            # Extract case ID from image filename
            case_id = image_path.stem.replace('.nii', '').replace('.gz', '').replace('_0000', '')
            
            # Find corresponding predicted mask
            mask_path = None
            
            if has_subdirs:
                # Search in train/test subdirectories
                for split_name in splits_to_search:
                    if split_name == 'train':
                        mask_path = train_dir / f"{case_id}_majority_vote.nii.gz"
                    elif split_name == 'test':
                        mask_path = test_dir / f"{case_id}_majority_vote.nii.gz"
                    
                    if mask_path.exists():
                        break
            else:
                # Old format: try multiple naming conventions in root directory
                mask_name = image_path.name
                mask_path = predicted_masks_dir / mask_name
                
                if not mask_path.exists():
                    # Try without _0000 suffix
                    mask_name_alt = mask_name.replace('_0000.nii.gz', '.nii.gz').replace('_0000.nii', '.nii')
                    mask_path = predicted_masks_dir / mask_name_alt
                
                if not mask_path.exists():
                    # Try with _pred_mask suffix
                    mask_name_alt2 = image_path.stem.replace('.nii', '').replace('.gz', '') + '_pred_mask.nii.gz'
                    mask_path = predicted_masks_dir / mask_name_alt2
                
                if not mask_path.exists():
                    # Try with _majority_vote suffix (new format)
                    mask_path = predicted_masks_dir / f"{case_id}_majority_vote.nii.gz"
            
            if not mask_path or not mask_path.exists():
                logger.warning(f"Predicted mask not found for {image_path.name} (case_id: {case_id}), skipping")
                continue
            
            try:
                case_results = process_case_from_predicted_mask(
                    image_path,
                    mask_path,
                    roi_labels=roi_labels,
                    radiomics_groups=args.radiomics_groups,
                    device=device,
                    voxelArrayShift=args.voxel_array_shift,
                    binWidth=args.bin_width
                )
                
                if len(case_results) > 0:
                    all_results.append(case_results)
                else:
                    logger.warning(f"No results for {image_path.name}")
            
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
                continue
    
    # Mode: inference (run nnU-Net)
    else:
        for idx, image_path in enumerate(image_files, 1):
            logger.info(f"\n[{idx}/{len(image_files)}] Processing {image_path.name}")
            
            try:
                case_results = process_case(
                    image_path,
                    args.nnunet_model_dir,
                    output_dir,
                    folds=args.folds,
                    radiomics_groups=args.radiomics_groups,
                    device=device,
                    save_masks=args.save_masks,
                    roi_labels=roi_labels,
                    voxelArrayShift=args.voxel_array_shift,
                    binWidth=args.bin_width
                )
                
                if len(case_results) > 0:
                    all_results.append(case_results)
                else:
                    logger.warning(f"No results for {image_path.name}")
            
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
                continue
    
    # Combine and save results
    if len(all_results) == 0:
        logger.error("No results to save!")
        return
    
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Save in requested format
    if args.output_format == 'csv':
        output_file = output_dir / 'radiomics_results.csv'
        df_all.to_csv(output_file, index=False)
        logger.info(f"\nSaved results to {output_file}")
    else:
        output_file = output_dir / 'radiomics_results.parquet'
        df_all.to_parquet(output_file, index=False)
        logger.info(f"\nSaved results to {output_file}")
    
    # Also save in wide format (one row per case+ROI)
    df_wide = df_all.pivot_table(
        index=['case_id', 'roi_name'],
        columns='feature_name',
        values='value'
    ).reset_index()
    
    wide_file = output_dir / f'radiomics_results_wide.{args.output_format}'
    if args.output_format == 'csv':
        df_wide.to_csv(wide_file, index=False)
    else:
        df_wide.to_parquet(wide_file, index=False)
    
    logger.info(f"Saved wide format to {wide_file}")
    logger.info(f"\nTotal cases processed: {len(df_all['case_id'].unique())}")
    logger.info(f"Total ROIs: {len(df_all.groupby(['case_id', 'roi_name']))}")
    logger.info(f"Total features: {len(df_all['feature_name'].unique())}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
