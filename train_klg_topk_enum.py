"""
KLG TopK Feature Selection and Model Training

This script implements:
1. Pre-filtering + TopK enumeration feature selection
2. Logistic regression training with cross-validation
3. Nested CV evaluation
4. Best subset selection and model saving

Author: ML Engineer

Usage Examples:

# First time: Use dataloader and save combined data for reuse
python train_klg_topk_enum.py \
    --use-dataloader \
    --dataset-dir ../Dataset360_oaizib \
    --knee-side-csv ../OAI-ZIB-CM/info/kneeSideInfo.csv \
    --subinfo-train ../OAI-ZIB-CM/info/subInfo_train.xlsx \
    --subinfo-test ../OAI-ZIB-CM/info/subInfo_test.xlsx \
    --radiomics-features ./radiomics_output/radiomics_features_train.parquet \
    --save-combined-data ./radiomics_output/radiomics_klg_data_train.parquet \
    --output-dir ./klg_topk_results \
    --target-col KLGrade \
    --task multiclass \
    --k 15

# Next time: Use saved combined data (faster)
python train_klg_topk_enum.py \
    --input-table ./radiomics_output/radiomics_klg_data_train.parquet \
    --output-dir ./klg_topk_results \
    --target-col KLGrade \
    --task multiclass \
    --k 5
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
    VarianceThreshold
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import existing dataloader and utility functions
from Radiomics_KLG_mapping_dataloader import (
    RadiomicsKLGDataLoader,
    load_knee_side_info,
    load_subject_info,
    extract_case_id_from_filename,
    process_case_gt
)


# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_data_from_dataloader(
    dataset_dir: Union[str, Path],
    knee_side_csv: Union[str, Path],
    subinfo_train: Union[str, Path],
    subinfo_test: Union[str, Path],
    radiomics_features_path: Optional[Union[str, Path]] = None,
    split: str = 'train',
    radiomics_groups: List[str] = None,
    device = None,
    voxelArrayShift: int = 0,
    binWidth: Optional[float] = None
) -> pd.DataFrame:
    """
    Load data using RadiomicsKLGDataLoader.
    
    Args:
        dataset_dir: Path to Dataset360_oaizib directory
        knee_side_csv: Path to kneeSideInfo.csv
        subinfo_train: Path to subInfo_train.xlsx
        subinfo_test: Path to subInfo_test.xlsx
        radiomics_features_path: Optional path to pre-extracted radiomics features
        split: 'train' or 'test'
        radiomics_groups: List of radiomics feature groups
        device: PyTorch device
        voxelArrayShift: Voxel array shift
        binWidth: Histogram bin width
        
    Returns:
        DataFrame with combined radiomics and metadata
    """
    logger.info(f"Initializing RadiomicsKLGDataLoader for {split} split...")
    
    loader = RadiomicsKLGDataLoader(
        dataset_dir=dataset_dir,
        knee_side_csv=knee_side_csv,
        subinfo_train=subinfo_train,
        subinfo_test=subinfo_test,
        radiomics_groups=radiomics_groups,
        device=device,
        voxelArrayShift=voxelArrayShift,
        binWidth=binWidth,
        split=split
    )
    
    # Load or extract radiomics features
    if radiomics_features_path and Path(radiomics_features_path).exists():
        logger.info(f"Loading pre-extracted features from {radiomics_features_path}")
        loader.load_radiomics(radiomics_features_path)
    else:
        logger.info("No pre-extracted features found. You should extract features first.")
        logger.info("Use Radiomics_KLG_mapping_dataloader.py to extract features.")
        raise ValueError(
            f"Radiomics features not found at {radiomics_features_path}. "
            "Please extract features first using Radiomics_KLG_mapping_dataloader.py"
        )
    
    # Get combined training data
    df = loader.get_training_data()
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Debug: Print data structure
    logger.info(f"[DEBUG] Data shape: {df.shape}")
    logger.info(f"[DEBUG] Columns: {len(df.columns)} total")
    if 'case_id' in df.columns:
        logger.info(f"[DEBUG] Unique cases: {df['case_id'].nunique()}")
    if 'roi_name' in df.columns:
        logger.info(f"[DEBUG] Unique ROIs: {df['roi_name'].nunique()}")
        logger.info(f"[DEBUG] ROI distribution:\n{df['roi_name'].value_counts()}")
    
    # Count radiomics vs metadata columns
    metadata_keywords = ['case_id', 'roi_name', 'knee_side', 'side', 'age', 'sex', 'gender', 'bmi', 'weight', 'height']
    radiomics_cols = [c for c in df.columns if not any(kw in c.lower() for kw in metadata_keywords)]
    metadata_cols = [c for c in df.columns if c not in radiomics_cols]
    logger.info(f"[DEBUG] Radiomics feature columns: {len(radiomics_cols)}")
    logger.info(f"[DEBUG] Metadata columns: {len(metadata_cols)}")
    if len(metadata_cols) > 0:
        logger.info(f"[DEBUG] Metadata column names: {metadata_cols}")
    
    return df


def load_table(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load table from CSV or Parquet file (alternative to dataloader).
    
    Args:
        input_path: Path to input file
        
    Returns:
        DataFrame
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .csv or .parquet")
    
    logger.info(f"Loaded table: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Debug: Print data structure
    logger.info(f"[DEBUG] Data shape: {df.shape}")
    logger.info(f"[DEBUG] Columns: {len(df.columns)} total")
    if 'case_id' in df.columns:
        logger.info(f"[DEBUG] Unique cases: {df['case_id'].nunique()}")
    if 'roi_name' in df.columns:
        logger.info(f"[DEBUG] Unique ROIs: {df['roi_name'].nunique()}")
        logger.info(f"[DEBUG] ROI distribution:\n{df['roi_name'].value_counts()}")
    
    # Count radiomics vs metadata columns
    metadata_keywords = ['case_id', 'roi_name', 'knee_side', 'side', 'age', 'sex', 'gender', 'bmi', 'weight', 'height']
    radiomics_cols = [c for c in df.columns if not any(kw in c.lower() for kw in metadata_keywords)]
    metadata_cols = [c for c in df.columns if c not in radiomics_cols]
    logger.info(f"[DEBUG] Radiomics feature columns: {len(radiomics_cols)}")
    logger.info(f"[DEBUG] Metadata columns: {len(metadata_cols)}")
    if len(metadata_cols) > 0:
        logger.info(f"[DEBUG] Metadata column names: {metadata_cols}")
    
    return df


def prepare_Xy(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str] = None,
    include_metadata: bool = False,
    roi_mode: str = 'per_roi',
    roi_name: Optional[str] = None,
    use_dataloader: bool = False,
    dataloader: Optional[RadiomicsKLGDataLoader] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare feature matrix X and target vector y.
    
    Uses RadiomicsKLGDataLoader.get_roi_specific_data() if available.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        drop_cols: Columns to drop (default: ['case_id', 'roi_name'])
        include_metadata: Whether to include metadata columns
        roi_mode: 'per_roi' or 'all_rois'
        roi_name: ROI name if roi_mode='per_roi'
        use_dataloader: Whether to use dataloader's get_roi_specific_data()
        dataloader: RadiomicsKLGDataLoader instance (if use_dataloader=True)
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    if drop_cols is None:
        drop_cols = ['case_id', 'roi_name']
    
    # Check target column exists
    if target_col not in df.columns:
        available_cols = df.columns.tolist()
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame.\n"
            f"Available columns: {available_cols[:20]}..."
            f"{'...' if len(available_cols) > 20 else ''}"
        )
    
    # Handle ROI mode - use dataloader if available
    if roi_mode == 'per_roi':
        if roi_name is None:
            raise ValueError("roi_name must be provided when roi_mode='per_roi'")
        
        if use_dataloader and dataloader is not None:
            # Use dataloader's method (recommended)
            logger.info(f"Using RadiomicsKLGDataLoader.get_roi_specific_data() for ROI '{roi_name}'")
            df = dataloader.get_roi_specific_data(roi_name, target_column=target_col)
            logger.info(f"Loaded ROI data: {len(df)} rows")
        else:
            # Manual filtering
            if 'roi_name' not in df.columns:
                raise ValueError("'roi_name' column not found in DataFrame")
            df = df[df['roi_name'] == roi_name].copy()
            logger.info(f"Filtered to ROI '{roi_name}': {len(df)} rows")
    
    elif roi_mode == 'all_rois':
        if 'roi_name' not in df.columns:
            raise ValueError("'roi_name' column not found for all_rois mode")
        if 'case_id' not in df.columns:
            raise ValueError("'case_id' column not found for all_rois mode")
        
        # Pivot wide: one row per case, ROI features as columns
        logger.info("Merging ROIs into one row per case...")
        df_wide_list = []
        for case_id in df['case_id'].unique():
            case_df = df[df['case_id'] == case_id].copy()
            case_row = {'case_id': case_id}
            
            for _, row in case_df.iterrows():
                roi_name_row = row['roi_name']
                for col in case_df.columns:
                    if col not in ['case_id', 'roi_name']:
                        new_col = f"roi{roi_name_row}__{col}"
                        case_row[new_col] = row[col]
            
            df_wide_list.append(case_row)
        
        df = pd.DataFrame(df_wide_list)
        logger.info(f"Merged to {len(df)} rows (one per case)")
    
    # Extract target
    y = df[target_col].copy()
    
    # Drop target and specified columns
    cols_to_drop = [target_col] + [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    
    # Identify feature columns
    if include_metadata:
        # Keep all numeric columns + one-hot encodable categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        feature_cols = numeric_cols + categorical_cols
    else:
        # Only radiomics features (assumed to be numeric)
        # Exclude obvious metadata columns
        metadata_keywords = ['knee_side', 'side', 'age', 'sex', 'gender', 'bmi', 'weight', 'height']
        feature_cols = [
            col for col in X.select_dtypes(include=[np.number]).columns
            if not any(kw in col.lower() for kw in metadata_keywords)
        ]
        logger.info(f"Filtered to radiomics features only: {len(feature_cols)} columns")
    
    X = X[feature_cols]
    
    # Remove rows with missing target
    valid_mask = ~y.isna()
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    
    logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
    
    # Debug: Print detailed information
    logger.info(f"[DEBUG] Feature matrix X:")
    logger.info(f"  - Shape: {X.shape}")
    logger.info(f"  - Number of features: {X.shape[1]}")
    logger.info(f"  - Number of samples: {X.shape[0]}")
    logger.info(f"  - Missing values: {X.isna().sum().sum()} total")
    logger.info(f"  - Features with missing values: {(X.isna().sum() > 0).sum()}")
    
    logger.info(f"[DEBUG] Target vector y:")
    logger.info(f"  - Shape: {y.shape}")
    logger.info(f"  - Missing values: {y.isna().sum()}")
    logger.info(f"  - Unique values: {y.nunique()}")
    logger.info(f"  - Value counts:\n{y.value_counts().sort_index()}")
    
    if len(feature_cols) > 0:
        logger.info(f"[DEBUG] First 10 feature names: {feature_cols[:10]}")
        if len(feature_cols) > 10:
            logger.info(f"[DEBUG] ... and {len(feature_cols) - 10} more features")
    
    return X, y, feature_cols


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified CV splits.
    
    Args:
        X: Feature matrix
        y: Target vector
        cv_folds: Number of CV folds
        seed: Random seed
        
    Returns:
        List of (train_idx, val_idx) tuples
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    logger.info(f"Created {cv_folds} stratified CV splits")
    return splits


# ============================================================================
# Preprocessing Pipeline
# ============================================================================

def apply_variance_threshold(
    X: pd.DataFrame,
    threshold: float = 0.0,
    selector: Optional[VarianceThreshold] = None
) -> Tuple[pd.DataFrame, VarianceThreshold]:
    """
    Apply variance threshold and return DataFrame with preserved column names.
    
    Args:
        X: Input DataFrame
        threshold: Variance threshold
        selector: Optional pre-fitted selector (if None, fits on X)
        
    Returns:
        Tuple of (filtered DataFrame, fitted selector)
    """
    if threshold <= 0:
        return X, None
    
    if selector is None:
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
    
    X_transformed = selector.transform(X)
    
    # Get feature names that passed
    feature_mask = selector.get_support()
    feature_names = X.columns[feature_mask].tolist()
    
    # Return as DataFrame with preserved names
    return pd.DataFrame(X_transformed, columns=feature_names, index=X.index), selector


def build_preprocessing_pipeline(
    variance_threshold: float = 0.0,
    corr_threshold: Optional[float] = None
) -> Pipeline:
    """
    Build preprocessing pipeline (without variance threshold - apply separately).
    
    Args:
        variance_threshold: Variance threshold (not used here, apply separately)
        corr_threshold: Optional correlation threshold (not used here, apply separately)
        
    Returns:
        Preprocessing pipeline
    """
    steps = []
    
    # Impute missing values
    steps.append(('impute', SimpleImputer(strategy='median')))
    
    # Standardize
    steps.append(('scaler', StandardScaler()))
    
    pipeline = Pipeline(steps)
    return pipeline


def filter_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.95
) -> List[str]:
    """
    Filter highly correlated features (keep first occurrence).
    
    Args:
        X: Feature matrix
        threshold: Correlation threshold
        
    Returns:
        List of feature names to keep
    """
    if X.shape[1] == 0:
        return []
    
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [
        column for column in upper_tri.columns
        if any(upper_tri[column] > threshold)
    ]
    
    to_keep = [col for col in X.columns if col not in to_drop]
    logger.info(f"Correlation filtering: {len(to_keep)}/{len(X.columns)} features kept")
    
    return to_keep


# ============================================================================
# Feature Selection
# ============================================================================

def rank_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ranker: str = 'f_classif',
    candidate_pool_size: int = 60
) -> List[str]:
    """
    Rank features using univariate scoring.
    
    Args:
        X_train: Training features
        y_train: Training target
        ranker: 'f_classif' or 'mutual_info'
        candidate_pool_size: Number of top features to keep
        
    Returns:
        List of top feature names
    """
    if ranker == 'f_classif':
        scores, _ = f_classif(X_train, y_train)
    elif ranker == 'mutual_info':
        scores = mutual_info_classif(X_train, y_train, random_state=42)
    else:
        raise ValueError(f"Unknown ranker: {ranker}")
    
    # Handle NaN/Inf scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get top features
    top_indices = np.argsort(scores)[::-1][:candidate_pool_size]
    top_features = [X_train.columns[i] for i in top_indices]
    
    logger.info(f"Ranked features: top {len(top_features)} selected from {len(X_train.columns)}")
    
    return top_features


def generate_combinations(
    candidate_features: List[str],
    k: int,
    max_combos: int = 50000,
    seed: int = 42
) -> List[Tuple[str, ...]]:
    """
    Generate feature combinations of size k.
    
    Args:
        candidate_features: List of candidate feature names
        k: Subset size
        max_combos: Maximum number of combinations to generate
        seed: Random seed
        
    Returns:
        List of feature combination tuples
    """
    n = len(candidate_features)
    total_combos = math.comb(n, k) if n >= k else 0
    
    if total_combos == 0:
        logger.warning(f"Cannot generate combinations: n={n}, k={k}")
        return []
    
    if total_combos <= max_combos:
        # Enumerate all combinations
        from itertools import combinations
        combos = list(combinations(candidate_features, k))
        logger.info(f"Enumerating all {len(combos)} combinations")
    else:
        # Random sampling
        random.seed(seed)
        np.random.seed(seed)
        combos_set = set()
        max_attempts = max_combos * 10
        
        while len(combos_set) < max_combos and len(combos_set) < total_combos:
            combo = tuple(sorted(random.sample(candidate_features, k)))
            combos_set.add(combo)
            
            if len(combos_set) >= max_attempts:
                break
        
        combos = list(combos_set)
        logger.info(f"Sampled {len(combos)} unique combinations from {total_combos} total")
    
    return combos


# ============================================================================
# Model Evaluation
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task: str = 'multiclass'
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        task: 'binary' or 'multiclass'
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    if task == 'multiclass':
        # F1 scores
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
        
        # AUC (one-vs-rest)
        if y_proba is not None:
            try:
                metrics['auroc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                )
            except:
                metrics['auroc'] = np.nan
    else:
        # Binary metrics
        metrics['f1'] = f1_score(y_true, y_pred)
        
        if y_proba is not None:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_proba)
                metrics['auprc'] = average_precision_score(y_true, y_proba)
            except:
                metrics['auroc'] = np.nan
                metrics['auprc'] = np.nan
    
    return metrics


def eval_subset_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_subset: Tuple[str, ...],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    task: str,
    score: str,
    balanced: bool = False,
    variance_threshold: float = 0.0,
    corr_threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluate a feature subset using cross-validation.
    
    Args:
        X: Full feature matrix
        y: Target vector
        feature_subset: Tuple of feature names
        splits: CV splits
        task: 'binary' or 'multiclass'
        score: Primary scoring metric
        balanced: Use balanced class weights
        variance_threshold: Variance threshold
        corr_threshold: Correlation threshold
        
    Returns:
        Dictionary with mean_score, std_score, and other metrics
    """
    # Select features
    available_features = [f for f in feature_subset if f in X.columns]
    if len(available_features) == 0:
        return {
            'mean_score': -np.inf,
            'std_score': 0.0,
            'macro_f1': 0.0,
            'balanced_accuracy': 0.0,
            'auroc': 0.0
        }
    
    X_subset = X[available_features].copy()
    
    # Check if subset is valid
    if X_subset.shape[1] == 0:
        return {
            'mean_score': -np.inf,
            'std_score': 0.0,
            'macro_f1': 0.0,
            'balanced_accuracy': 0.0,
            'auroc': 0.0
        }
    
    fold_scores = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train_fold = X_subset.iloc[train_idx]
        X_val_fold = X_subset.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Apply variance threshold on training fold
        X_train_fold_variance, variance_selector = apply_variance_threshold(
            X_train_fold, threshold=variance_threshold
        )
        
        # Check which features remain
        remaining_features = X_train_fold_variance.columns.tolist()
        if len(remaining_features) == 0:
            continue
        
        X_train_fold = X_train_fold_variance
        
        # Apply same variance selector to validation fold
        if variance_selector is not None:
            X_val_fold_transformed = variance_selector.transform(X_val_fold)
            feature_mask = variance_selector.get_support()
            feature_names = X_val_fold.columns[feature_mask].tolist()
            X_val_fold = pd.DataFrame(
                X_val_fold_transformed,
                columns=feature_names,
                index=X_val_fold.index
            )
        else:
            X_val_fold = X_val_fold[remaining_features]
        
        # Optional correlation filtering on training fold only
        if corr_threshold is not None:
            features_to_keep = filter_correlated_features(
                X_train_fold, threshold=corr_threshold
            )
            if len(features_to_keep) == 0:
                continue
            X_train_fold = X_train_fold[features_to_keep]
            X_val_fold = X_val_fold[features_to_keep]
        
        # Build and fit preprocessing pipeline (without variance threshold)
        preprocessor = build_preprocessing_pipeline()
        
        X_train_processed = preprocessor.fit_transform(X_train_fold)
        X_val_processed = preprocessor.transform(X_val_fold)
        
        # Train model
        if task == 'multiclass':
            multi_class = 'multinomial'
        else:
            multi_class = 'ovr'
        
        model = LogisticRegression(
            solver='saga',
            max_iter=5000,
            class_weight='balanced' if balanced else None,
            multi_class=multi_class,
            random_state=42
        )
        
        try:
            model.fit(X_train_processed, y_train_fold)
            y_pred = model.predict(X_val_processed)
            y_proba = model.predict_proba(X_val_processed) if hasattr(model, 'predict_proba') else None
            
            # Compute metrics
            metrics = compute_metrics(y_val_fold, y_pred, y_proba, task=task)
            fold_metrics.append(metrics)
            fold_scores.append(metrics.get(score, 0.0))
        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed: {e}")
            continue
    
    if len(fold_scores) == 0:
        return {
            'mean_score': -np.inf,
            'std_score': 0.0,
            'macro_f1': 0.0,
            'balanced_accuracy': 0.0,
            'auroc': 0.0
        }
    
    # Aggregate metrics
    result = {
        'mean_score': np.mean(fold_scores),
        'std_score': np.std(fold_scores),
    }
    
    # Aggregate other metrics
    for metric_name in ['macro_f1', 'balanced_accuracy', 'auroc', 'auprc', 'f1']:
        if any(m.get(metric_name) is not None for m in fold_metrics):
            values = [m.get(metric_name, np.nan) for m in fold_metrics]
            values = [v for v in values if not np.isnan(v)]
            if len(values) > 0:
                result[metric_name] = np.mean(values)
            else:
                result[metric_name] = np.nan
    
    return result


# ============================================================================
# Main Training Function
# ============================================================================

def train_roi_model(
    df: pd.DataFrame,
    target_col: str,
    roi_name: str,
    args: argparse.Namespace,
    dataloader: Optional[RadiomicsKLGDataLoader] = None
) -> Dict:
    """
    Train model for a specific ROI.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        roi_name: ROI name
        args: Command-line arguments
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training model for ROI: {roi_name}")
    logger.info(f"{'='*80}")
    
    # Prepare data (use dataloader if available)
    use_dataloader = (dataloader is not None)
    X, y, feature_cols = prepare_Xy(
        df,
        target_col=target_col,
        drop_cols=args.drop_cols,
        include_metadata=args.include_metadata,
        roi_mode='per_roi',
        roi_name=roi_name,
        use_dataloader=use_dataloader,
        dataloader=dataloader
    )
    
    if len(X) == 0:
        logger.warning(f"No data for ROI {roi_name}")
        return None
    
    # Handle binary task
    if args.task == 'binary':
        y_binary = (y >= args.binary_threshold).astype(int)
        y = y_binary
        logger.info(f"Binary classification: threshold={args.binary_threshold}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Create CV splits
    splits = make_splits(X, y, cv_folds=args.cv_folds, seed=args.seed)
    
    # Debug: Print CV split information
    logger.info(f"[DEBUG] CV splits created:")
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")
        logger.info(f"    Train class distribution: {y.iloc[train_idx].value_counts().to_dict()}")
        logger.info(f"    Val class distribution: {y.iloc[val_idx].value_counts().to_dict()}")
    
    # Nested CV: outer loop for evaluation, inner loop for subset selection
    logger.info("Starting nested CV feature selection...")
    
    outer_scores = []
    best_subsets_per_outer_fold = []
    total_combos_evaluated = 0
    
    for outer_fold_idx, (outer_train_idx, outer_val_idx) in enumerate(splits):
        logger.info(f"\nOuter fold {outer_fold_idx + 1}/{args.cv_folds}")
        
        X_outer_train = X.iloc[outer_train_idx]
        y_outer_train = y.iloc[outer_train_idx]
        X_outer_val = X.iloc[outer_val_idx]
        y_outer_val = y.iloc[outer_val_idx]
        
        # Inner CV splits for subset selection
        inner_splits = make_splits(
            X_outer_train, y_outer_train,
            cv_folds=args.cv_folds, seed=args.seed + outer_fold_idx
        )
        
        # Pre-filter: rank features on outer training set
        # (Apply variance threshold first to preserve feature names)
        logger.info(f"[DEBUG] Outer fold {outer_fold_idx + 1} - Before variance threshold:")
        logger.info(f"  X_outer_train shape: {X_outer_train.shape}")
        
        X_outer_train_variance, _ = apply_variance_threshold(
            X_outer_train, threshold=args.variance_threshold
        )
        
        logger.info(f"[DEBUG] After variance threshold:")
        logger.info(f"  X_outer_train_variance shape: {X_outer_train_variance.shape}")
        logger.info(f"  Features removed: {X_outer_train.shape[1] - X_outer_train_variance.shape[1]}")
        
        # Then apply rest of preprocessing
        preprocessor = build_preprocessing_pipeline()
        
        try:
            X_outer_train_processed = pd.DataFrame(
                preprocessor.fit_transform(X_outer_train_variance),
                columns=X_outer_train_variance.columns,
                index=X_outer_train_variance.index
            )
            logger.info(f"[DEBUG] After preprocessing:")
            logger.info(f"  X_outer_train_processed shape: {X_outer_train_processed.shape}")
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, using variance-filtered features")
            X_outer_train_processed = X_outer_train_variance
        
        # Optional correlation filtering
        if args.corr_threshold is not None:
            logger.info(f"[DEBUG] Before correlation filtering: {X_outer_train_processed.shape[1]} features")
            features_to_keep = filter_correlated_features(
                X_outer_train_processed, threshold=args.corr_threshold
            )
            X_outer_train_processed = X_outer_train_processed[features_to_keep]
            logger.info(f"[DEBUG] After correlation filtering: {X_outer_train_processed.shape[1]} features")
            logger.info(f"[DEBUG] Features removed by correlation: {X_outer_train_processed.shape[1] - len(features_to_keep)}")
        
        # Rank features
        logger.info(f"[DEBUG] Ranking {X_outer_train_processed.shape[1]} features using {args.ranker}...")
        candidate_features = rank_features(
            X_outer_train_processed,
            y_outer_train,
            ranker=args.ranker,
            candidate_pool_size=args.candidate_pool_size
        )
        logger.info(f"[DEBUG] Selected {len(candidate_features)} candidate features for subset search")
        
        if len(candidate_features) < args.k:
            logger.warning(
                f"Only {len(candidate_features)} candidates available, "
                f"but k={args.k}. Using all candidates."
            )
            candidate_features_subset = candidate_features
            k_actual = len(candidate_features_subset)
        else:
            candidate_features_subset = candidate_features
            k_actual = args.k
        
        # Generate combinations
        logger.info(f"[DEBUG] Generating combinations: C({len(candidate_features_subset)}, {k_actual})")
        combinations = generate_combinations(
            candidate_features_subset,
            k=k_actual,
            max_combos=args.max_combos,
            seed=args.seed + outer_fold_idx
        )
        
        if len(combinations) == 0:
            logger.warning("No combinations generated")
            continue
        
        total_combos_evaluated += len(combinations)
        logger.info(f"[DEBUG] Generated {len(combinations)} unique combinations to evaluate")
        
        # Evaluate combinations using inner CV
        logger.info(f"Evaluating {len(combinations)} combinations...")
        
        if args.n_jobs > 1:
            results_list = Parallel(n_jobs=args.n_jobs)(
                delayed(eval_subset_cv)(
                    X_outer_train_processed,
                    y_outer_train,
                    combo,
                    inner_splits,
                    args.task,
                    args.score,
                    args.balanced,
                    args.variance_threshold,
                    None  # Correlation already filtered
                )
                for combo in combinations
            )
        else:
            results_list = [
                eval_subset_cv(
                    X_outer_train_processed,
                    y_outer_train,
                    combo,
                    inner_splits,
                    args.task,
                    args.score,
                    args.balanced,
                    args.variance_threshold,
                    None
                )
                for combo in combinations
            ]
        
        # Find best subset for this outer fold
        scores = [r['mean_score'] for r in results_list]
        best_idx = np.argmax(scores)
        best_subset = combinations[best_idx]
        best_result = results_list[best_idx]
        
        logger.info(f"[DEBUG] Outer fold {outer_fold_idx + 1} - Subset evaluation summary:")
        logger.info(f"  Best score: {best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}")
        logger.info(f"  Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        logger.info(f"  Best subset size: {len(best_subset)}")
        logger.info(f"  Best subset features: {list(best_subset)}")
        
        best_subsets_per_outer_fold.append({
            'subset': best_subset,
            'mean_score': best_result['mean_score'],
            'std_score': best_result['std_score']
        })
        
        logger.info(
            f"Best subset (outer fold {outer_fold_idx + 1}): "
            f"score={best_result['mean_score']:.4f}±{best_result['std_score']:.4f}"
        )
        
        # Evaluate best subset on outer validation set
        # Apply variance threshold first (fit on outer training, apply to validation)
        _, variance_selector_outer = apply_variance_threshold(
            X_outer_train, threshold=args.variance_threshold
        )
        if variance_selector_outer is not None:
            X_outer_val_transformed = variance_selector_outer.transform(X_outer_val)
            feature_mask = variance_selector_outer.get_support()
            feature_names = X_outer_val.columns[feature_mask].tolist()
            X_outer_val_variance = pd.DataFrame(
                X_outer_val_transformed,
                columns=feature_names,
                index=X_outer_val.index
            )
        else:
            X_outer_val_variance = X_outer_val
        
        # Check which features from best_subset are still available
        available_features = [f for f in best_subset if f in X_outer_val_variance.columns]
        if len(available_features) == 0:
            logger.warning(f"No features from best subset available in validation set")
            continue
        
        X_outer_val_subset = X_outer_val_variance[available_features].copy()
        
        # Apply same preprocessing
        try:
            X_outer_val_processed = pd.DataFrame(
                preprocessor.transform(X_outer_val_subset),
                columns=X_outer_val_subset.columns,
                index=X_outer_val_subset.index
            )
        except:
            X_outer_val_processed = X_outer_val_subset
        
        # Update best_subset to match available features
        best_subset = tuple(available_features)
        
        # Train final model on outer training set
        X_outer_train_best = X_outer_train_processed[list(best_subset)]
        
        if args.task == 'multiclass':
            multi_class = 'multinomial'
        else:
            multi_class = 'ovr'
        
        final_model = LogisticRegression(
            solver='saga',
            max_iter=5000,
            class_weight='balanced' if args.balanced else None,
            multi_class=multi_class,
            random_state=42
        )
        
        final_model.fit(X_outer_train_best, y_outer_train)
        
        # Evaluate on outer validation
        y_pred_outer = final_model.predict(X_outer_val_processed)
        y_proba_outer = final_model.predict_proba(X_outer_val_processed)
        
        outer_metrics = compute_metrics(
            y_outer_val.values, y_pred_outer, y_proba_outer, task=args.task
        )
        outer_scores.append(outer_metrics.get(args.score, 0.0))
        
        logger.info(
            f"Outer fold {outer_fold_idx + 1} validation {args.score}: "
            f"{outer_metrics.get(args.score, 0.0):.4f}"
        )
    
    # Select best subset across outer folds
    if len(best_subsets_per_outer_fold) == 0:
        logger.error("No valid subsets found")
        return None
    
    # Choose subset with best mean outer score
    best_outer_idx = np.argmax(outer_scores)
    best_global_subset = best_subsets_per_outer_fold[best_outer_idx]['subset']
    
    logger.info(f"\nBest global subset selected from outer fold {best_outer_idx + 1}")
    logger.info(f"Features: {list(best_global_subset)}")
    
    # Train final model on full data with best subset
    logger.info("Training final model on full data...")
    logger.info(f"[DEBUG] Full data before processing:")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    
    # Prepare full data: apply variance threshold first
    X_variance_full, variance_selector_final = apply_variance_threshold(
        X, threshold=args.variance_threshold
    )
    logger.info(f"[DEBUG] After variance threshold:")
    logger.info(f"  X_variance_full shape: {X_variance_full.shape}")
    
    # Check which features from best_global_subset are available
    available_features = [f for f in best_global_subset if f in X_variance_full.columns]
    if len(available_features) == 0:
        logger.error("No features from best subset available in full data")
        return None
    
    best_global_subset = tuple(available_features)
    
    # Apply rest of preprocessing
    preprocessor_final = build_preprocessing_pipeline()
    
    X_processed_full = pd.DataFrame(
        preprocessor_final.fit_transform(X_variance_full),
        columns=X_variance_full.columns,
        index=X_variance_full.index
    )
    logger.info(f"[DEBUG] After preprocessing:")
    logger.info(f"  X_processed_full shape: {X_processed_full.shape}")
    
    if args.corr_threshold is not None:
        logger.info(f"[DEBUG] Applying correlation filtering (threshold={args.corr_threshold})...")
        features_to_keep = filter_correlated_features(
            X_processed_full, threshold=args.corr_threshold
        )
        X_processed_full = X_processed_full[features_to_keep]
        logger.info(f"[DEBUG] After correlation filtering: {X_processed_full.shape[1]} features")
        # Update best subset if needed
        best_global_subset = tuple(
            f for f in best_global_subset if f in features_to_keep
        )
        logger.info(f"[DEBUG] Best subset after correlation filtering: {len(best_global_subset)} features")
    
    if len(best_global_subset) == 0:
        logger.error("Best subset is empty after filtering")
        return None
    
    X_final = X_processed_full[list(best_global_subset)]
    logger.info(f"[DEBUG] Final feature matrix for model training:")
    logger.info(f"  X_final shape: {X_final.shape}")
    logger.info(f"  Selected features: {list(best_global_subset)}")
    
    if args.task == 'multiclass':
        multi_class = 'multinomial'
    else:
        multi_class = 'ovr'
    
    final_model_full = LogisticRegression(
        solver='saga',
        max_iter=5000,
        class_weight='balanced' if args.balanced else None,
        multi_class=multi_class,
        random_state=42
    )
    
    logger.info(f"[DEBUG] Training final model...")
    logger.info(f"  Input shape: {X_final.shape}")
    logger.info(f"  Target shape: {y.shape}")
    logger.info(f"  Target distribution: {y.value_counts().to_dict()}")
    
    final_model_full.fit(X_final, y)
    
    logger.info(f"[DEBUG] Model training completed")
    logger.info(f"  Model classes: {final_model_full.classes_}")
    logger.info(f"  Model coefficients shape: {final_model_full.coef_.shape if hasattr(final_model_full, 'coef_') else 'N/A'}")
    
    # Create full pipeline
    from sklearn.pipeline import Pipeline as SklearnPipeline
    
    # Column selector (applied after variance threshold)
    def select_features(X):
        # X should be a DataFrame after variance threshold
        if isinstance(X, pd.DataFrame):
            return X[list(best_global_subset)]
        else:
            # If numpy array, we need to map indices
            # This shouldn't happen if we structure the pipeline correctly
            return X
    
    from sklearn.preprocessing import FunctionTransformer
    
    # Create a custom transformer that applies variance threshold + feature selection
    class VarianceAndFeatureSelector:
        def __init__(self, variance_threshold, feature_subset):
            self.variance_threshold = variance_threshold
            self.feature_subset = feature_subset
            self.variance_selector = None
        
        def fit(self, X, y=None):
            if self.variance_threshold > 0:
                self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
                self.variance_selector.fit(X)
            return self
        
        def transform(self, X):
            if self.variance_threshold > 0 and self.variance_selector is not None:
                X = self.variance_selector.transform(X)
                # Get feature mask
                feature_mask = self.variance_selector.get_support()
                # Map original feature names to remaining features
                if isinstance(X, pd.DataFrame):
                    remaining_features = X.columns[feature_mask].tolist()
                    X = X[remaining_features]
                    # Select subset
                    available = [f for f in self.feature_subset if f in X.columns]
                    return X[available]
                else:
                    # Numpy array - need to track feature mapping
                    # This is complex, so we'll handle it differently
                    return X
            else:
                if isinstance(X, pd.DataFrame):
                    available = [f for f in self.feature_subset if f in X.columns]
                    return X[available]
                return X
    
    # Create a simple wrapper class for the full pipeline
    class FullPipeline:
        """Wrapper for the complete preprocessing + classification pipeline."""
        def __init__(self, variance_threshold, variance_selector, preprocessor, 
                     feature_subset, classifier, corr_threshold=None):
            self.variance_threshold = variance_threshold
            self.variance_selector = variance_selector
            self.preprocessor = preprocessor
            self.feature_subset = feature_subset
            self.classifier = classifier
            self.corr_threshold = corr_threshold
        
        def predict(self, X):
            """Predict labels."""
            # Apply variance threshold using pre-fitted selector
            if self.variance_selector is not None:
                X_transformed = self.variance_selector.transform(X)
                feature_mask = self.variance_selector.get_support()
                feature_names = X.columns[feature_mask].tolist()
                X = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
            
            # Select features
            available = [f for f in self.feature_subset if f in X.columns]
            if len(available) == 0:
                raise ValueError("No features from subset available in X")
            X = X[available]
            
            # Apply correlation filtering if needed
            if self.corr_threshold is not None:
                features_to_keep = filter_correlated_features(
                    X, threshold=self.corr_threshold
                )
                X = X[features_to_keep]
            
            # Apply preprocessing
            X_processed = self.preprocessor.transform(X)
            
            # Predict
            return self.classifier.predict(X_processed)
        
        def predict_proba(self, X):
            """Predict probabilities."""
            # Apply variance threshold using pre-fitted selector
            if self.variance_selector is not None:
                X_transformed = self.variance_selector.transform(X)
                feature_mask = self.variance_selector.get_support()
                feature_names = X.columns[feature_mask].tolist()
                X = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
            
            # Select features
            available = [f for f in self.feature_subset if f in X.columns]
            if len(available) == 0:
                raise ValueError("No features from subset available in X")
            X = X[available]
            
            # Apply correlation filtering if needed
            if self.corr_threshold is not None:
                features_to_keep = filter_correlated_features(
                    X, threshold=self.corr_threshold
                )
                X = X[features_to_keep]
            
            # Apply preprocessing
            X_processed = self.preprocessor.transform(X)
            
            # Predict
            return self.classifier.predict_proba(X_processed)
    
    full_pipeline = FullPipeline(
        variance_threshold=args.variance_threshold,
        variance_selector=variance_selector_final,
        preprocessor=preprocessor_final,
        feature_subset=list(best_global_subset),
        classifier=final_model_full,
        corr_threshold=args.corr_threshold
    )
    
    # Evaluate on full data (for reference)
    # Apply same preprocessing as training
    X_eval_variance = apply_variance_threshold(X, threshold=args.variance_threshold)
    X_eval_processed = pd.DataFrame(
        preprocessor_final.transform(X_eval_variance),
        columns=X_eval_variance.columns,
        index=X_eval_variance.index
    )
    if args.corr_threshold is not None:
        features_to_keep = filter_correlated_features(
            X_eval_processed, threshold=args.corr_threshold
        )
        X_eval_processed = X_eval_processed[features_to_keep]
    X_eval_final = X_eval_processed[list(best_global_subset)]
    
    y_pred_full = final_model_full.predict(X_eval_final)
    y_proba_full = final_model_full.predict_proba(X_eval_final) if hasattr(final_model_full, 'predict_proba') else None
    
    final_metrics = compute_metrics(
        y.values, y_pred_full, y_proba_full, task=args.task
    )
    
    # Compile results
    result = {
        'roi_name': roi_name,
        'subset_features': list(best_global_subset),
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'n_combos_evaluated': total_combos_evaluated,
        'candidate_pool_size': args.candidate_pool_size,
        'k': len(best_global_subset),
        'ranker': args.ranker,
        'seed': args.seed,
        'final_metrics': final_metrics,
        'outer_scores': outer_scores,
        'model': full_pipeline,
        'feature_names': list(best_global_subset)
    }
    
    # Add aggregated metrics
    for metric_name in ['macro_f1', 'balanced_accuracy', 'auroc', 'auprc', 'f1']:
        if metric_name in final_metrics:
            result[metric_name] = final_metrics[metric_name]
    
    return result


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='KLG TopK Feature Selection and Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output - Two modes: direct table or dataloader
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-table',
        type=str,
        default=None,
        help='Path to input parquet/csv file (from RadiomicsKLGDataLoader)'
    )
    
    # Dataloader mode arguments
    input_group.add_argument(
        '--use-dataloader',
        action='store_true',
        help='Use RadiomicsKLGDataLoader directly instead of loading a table'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        help='Path to Dataset360_oaizib directory (required if --use-dataloader)'
    )
    
    parser.add_argument(
        '--knee-side-csv',
        type=str,
        default=None,
        help='Path to kneeSideInfo.csv (required if --use-dataloader)'
    )
    
    parser.add_argument(
        '--subinfo-train',
        type=str,
        default=None,
        help='Path to subInfo_train.xlsx (required if --use-dataloader)'
    )
    
    parser.add_argument(
        '--subinfo-test',
        type=str,
        default=None,
        help='Path to subInfo_test.xlsx (required if --use-dataloader)'
    )
    
    parser.add_argument(
        '--radiomics-features',
        type=str,
        default=None,
        help='Path to pre-extracted radiomics features (parquet/csv). Required if --use-dataloader'
    )
    
    parser.add_argument(
        '--radiomics-groups',
        type=str,
        nargs='+',
        default=['firstorder', 'shape', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'],
        choices=['firstorder', 'shape', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'],
        help='Radiomics feature groups (used if extracting features)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--test-table',
        type=str,
        default=None,
        help='Optional test table for final evaluation'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test'],
        default='train',
        help='Data split to use (train or test)'
    )
    
    parser.add_argument(
        '--save-combined-data',
        type=str,
        default=None,
        help='Path to save combined data when using --use-dataloader (for reuse with --input-table later)'
    )
    
    # Task configuration
    parser.add_argument(
        '--target-col',
        type=str,
        required=True,
        help='Target column name (e.g., KLG)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['binary', 'multiclass'],
        default='multiclass',
        help='Classification task type'
    )
    
    parser.add_argument(
        '--binary-threshold',
        type=float,
        default=2.0,
        help='Threshold for binary classification (label = KLG >= threshold)'
    )
    
    # ROI handling
    parser.add_argument(
        '--roi-mode',
        type=str,
        choices=['per_roi', 'all_rois'],
        default='per_roi',
        help='ROI handling mode'
    )
    
    # Feature selection
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Subset size (number of features to select)'
    )
    
    parser.add_argument(
        '--candidate-pool-size',
        type=int,
        default=60,
        help='Number of top-ranked features to consider'
    )
    
    parser.add_argument(
        '--max-combos',
        type=int,
        default=50000,
        help='Maximum number of combinations to evaluate'
    )
    
    parser.add_argument(
        '--ranker',
        type=str,
        choices=['f_classif', 'mutual_info'],
        default='f_classif',
        help='Feature ranking method'
    )
    
    # Cross-validation
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Preprocessing
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.0,
        help='Variance threshold for feature filtering'
    )
    
    parser.add_argument(
        '--corr-threshold',
        type=float,
        default=None,
        help='Correlation threshold for feature filtering (optional)'
    )
    
    # Data handling
    parser.add_argument(
        '--drop-cols',
        type=str,
        nargs='+',
        default=['case_id', 'roi_name'],
        help='Columns to drop from features'
    )
    
    parser.add_argument(
        '--include-metadata',
        action='store_true',
        help='Include metadata columns in features'
    )
    
    # Model
    parser.add_argument(
        '--balanced',
        action='store_true',
        help='Use balanced class weights'
    )
    
    parser.add_argument(
        '--score',
        type=str,
        default='macro_f1',
        help='Primary scoring metric'
    )
    
    # Performance
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = output_dir / 'log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log configuration
    logger.info("="*80)
    logger.info("KLG TopK Feature Selection and Model Training")
    logger.info("="*80)
    logger.info(f"Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Load data - either from table or using dataloader
    dataloader = None
    if args.use_dataloader:
        # Validate dataloader arguments
        if not args.dataset_dir:
            raise ValueError("--dataset-dir is required when using --use-dataloader")
        if not args.knee_side_csv:
            raise ValueError("--knee-side-csv is required when using --use-dataloader")
        if not args.subinfo_train:
            raise ValueError("--subinfo-train is required when using --use-dataloader")
        if not args.subinfo_test:
            raise ValueError("--subinfo-test is required when using --use-dataloader")
        if not args.radiomics_features:
            raise ValueError("--radiomics-features is required when using --use-dataloader")
        
        logger.info("Using RadiomicsKLGDataLoader to load data...")
        df = load_data_from_dataloader(
            dataset_dir=args.dataset_dir,
            knee_side_csv=args.knee_side_csv,
            subinfo_train=args.subinfo_train,
            subinfo_test=args.subinfo_test,
            radiomics_features_path=args.radiomics_features,
            split=args.split,
            radiomics_groups=args.radiomics_groups
        )
        
        # Create dataloader instance for later use
        dataloader = RadiomicsKLGDataLoader(
            dataset_dir=args.dataset_dir,
            knee_side_csv=args.knee_side_csv,
            subinfo_train=args.subinfo_train,
            subinfo_test=args.subinfo_test,
            radiomics_groups=args.radiomics_groups,
            split=args.split
        )
        dataloader.load_radiomics(args.radiomics_features)
        
        # Save combined data if requested (for reuse next time)
        if args.save_combined_data:
            save_path = Path(args.save_combined_data)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix == '.parquet':
                df.to_parquet(save_path, index=False)
            else:
                df.to_csv(save_path, index=False)
            
            logger.info(f"Saved combined data to {save_path} for future use")
            logger.info(f"You can use --input-table {save_path} next time for faster loading")
    else:
        # Load from pre-saved table
        if not args.input_table:
            raise ValueError("--input-table is required when not using --use-dataloader")
        logger.info(f"\nLoading data from {args.input_table}...")
        df = load_table(args.input_table)
    
    # Process ROIs
    if args.roi_mode == 'per_roi':
        if 'roi_name' not in df.columns:
            raise ValueError("'roi_name' column not found. Required for per_roi mode.")
        
        unique_rois = df['roi_name'].unique()
        logger.info(f"Found {len(unique_rois)} ROIs: {list(unique_rois)}")
        
        all_results = []
        for roi_name in unique_rois:
            try:
                result = train_roi_model(df, args.target_col, roi_name, args, dataloader=dataloader)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing ROI {roi_name}: {e}", exc_info=True)
                continue
        
    else:  # all_rois
        logger.info("Processing all ROIs merged...")
        result = train_roi_model(df, args.target_col, 'all_rois', args, dataloader=dataloader)
        all_results = [result] if result is not None else []
    
    if len(all_results) == 0:
        logger.error("No results to save!")
        return
    
    # Save results
    logger.info("\nSaving results...")
    
    # Results summary CSV
    results_rows = []
    for result in all_results:
        row = {
            'roi_name': result['roi_name'],
            'subset_features': str(result['subset_features']),
            'mean_score': result['mean_score'],
            'std_score': result['std_score'],
            'k': result['k'],
            'candidate_pool_size': result['candidate_pool_size'],
            'n_combos_evaluated': result['n_combos_evaluated'],
            'ranker': result['ranker'],
            'seed': result['seed']
        }
        
        # Add metrics
        for metric_name in ['macro_f1', 'balanced_accuracy', 'auroc', 'auprc', 'f1']:
            if metric_name in result:
                row[metric_name] = result[metric_name]
        
        results_rows.append(row)
    
    results_df = pd.DataFrame(results_rows)
    results_file = output_dir / 'results_subsets.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved results to {results_file}")
    
    # Best subset JSON
    best_subset_dict = {}
    for result in all_results:
        best_subset_dict[result['roi_name']] = {
            'subset_features': result['subset_features'],
            'mean_score': float(result['mean_score']),
            'std_score': float(result['std_score']),
            'metrics': {k: float(v) for k, v in result['final_metrics'].items() if isinstance(v, (int, float, np.number))},
            'settings': {
                'k': result['k'],
                'candidate_pool_size': result['candidate_pool_size'],
                'ranker': result['ranker'],
                'seed': result['seed']
            }
        }
    
    best_subset_file = output_dir / 'best_subset.json'
    with open(best_subset_file, 'w') as f:
        json.dump(best_subset_dict, f, indent=2)
    logger.info(f"Saved best subsets to {best_subset_file}")
    
    # Save models
    for result in all_results:
        model_file = output_dir / f"final_model_{result['roi_name']}.joblib"
        dump(result['model'], model_file)
        logger.info(f"Saved model to {model_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in all_results:
        print(f"\nROI: {result['roi_name']}")
        print(f"  Best subset size: {result['k']}")
        print(f"  Mean CV score ({args.score}): {result['mean_score']:.4f} ± {result['std_score']:.4f}")
        if 'macro_f1' in result:
            print(f"  Macro F1: {result['macro_f1']:.4f}")
        if 'balanced_accuracy' in result:
            print(f"  Balanced Accuracy: {result['balanced_accuracy']:.4f}")
        print(f"  Features: {result['subset_features']}")
    
    # Overall best
    if len(all_results) > 1:
        best_result = max(all_results, key=lambda x: x['mean_score'])
        print(f"\nOverall best ROI: {best_result['roi_name']}")
        print(f"  Score: {best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()

