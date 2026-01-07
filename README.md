# OA KLGrade Top-K Feature Selection / 骨关节炎KL分级Top-K特征选择

## English

### Project Overview

This project implements an **image-conditioned feature selector** combined with a **KLGrade classifier** for osteoarthritis (OA) severity prediction. The model learns to select patient-specific radiomics features based on 3D medical images, then uses these selected features to predict KLGrade (Kellgren-Lawrence Grade, 0-4).

**Key Innovation**: Instead of using all radiomics features or a fixed subset, the model dynamically selects the most relevant features for each patient based on their medical image, enabling personalized feature selection.

### Architecture

The model consists of three main components:

#### 1. **FeatureSelector** (`models.py`)
- **Backbone**: 3D CNN (either pretrained R3D-18 or a simple 3D CNN)
  - Input: 3D medical image `[B, 1, D, H, W]` (typically `[B, 1, 32, 128, 128]`)
  - Output: Image features `[B, feature_dim]` (typically 512-dim)
- **Gate Head**: MLP that outputs gate logits for each radiomics feature
  - Input: Image features `[B, feature_dim]`
  - Output: Gate logits `[B, n_features]` (e.g., 535 features)
  - Architecture: `Linear(512 → 256) → ReLU → Dropout(0.3) → Linear(256 → n_features)`

#### 2. **KLGradeClassifier** (`models.py`)
- Simple linear classifier (multiclass logistic regression)
- Input: Masked radiomics `[B, n_features]` (only selected features are non-zero)
- Output: Class logits `[B, 5]` (5 classes: KLGrade 0-4)
- Architecture: `Linear(n_features → 5)`

#### 3. **JointModel** (`models.py`)
- Combines FeatureSelector and KLGradeClassifier
- Two-stage gating mechanism:
  - **Warmup Stage**: Soft thresholding with learnable threshold
  - **Hard Top-K Stage**: Hard top-k selection with straight-through estimator

### Loss Function Components

The total loss consists of four components:

```
Total Loss = L_cls + λ_k × L_k + λ_diversity × L_diversity + L_entropy
```

#### 1. **Classification Loss (L_cls)**
- **Type**: CrossEntropyLoss (weighted if `--use-class-weights` is enabled)
- **Purpose**: Minimize classification error for KLGrade prediction
- **Formula**: 
  ```
  L_cls = -Σ y_true × log(softmax(logits))
  ```
- **Weight**: 1.0 (no scaling)

#### 2. **Sparsity Loss (L_k)**
- **Type**: Target-k regularization
- **Purpose**: Encourage the model to select exactly k features
- **Formula**:
  ```
  L_k = mean((Σ p_i - k)²)
  ```
  where `p_i = sigmoid(gate_logits_i)` is the gate probability for feature i
- **Weight**: `λ_k` (default: 0.05, starts at `λ_k_start` = 0.005 during warmup)
- **Schedule**: Linearly interpolated from `λ_k_start` to `λ_k` during warmup epochs

#### 3. **Diversity Loss (L_diversity)**
- **Type**: Pairwise cosine similarity penalty
- **Purpose**: Encourage different patients to select different features (prevent collapse to fixed feature set)
- **Formula**:
  ```
  p_normalized = normalize(p, p=2, dim=1)  # [B, n_features]
  similarity_matrix = p_normalized @ p_normalized.T  # [B, B]
  L_diversity = mean(similarity_matrix[upper_triangle])
  ```
- **Weight**: `λ_diversity` (default: 0.1, configurable via `--lambda-diversity`)
- **Note**: Penalizes high similarity between gate vectors of different patients in the same batch

#### 4. **Entropy Loss (L_entropy)**
- **Type**: Negative entropy regularization
- **Purpose**: Encourage exploration during warmup (prevent early collapse)
- **Formula**:
  ```
  entropy = -mean(Σ [p_i × log(p_i + ε) + (1-p_i) × log(1-p_i + ε)])
  normalized_entropy = entropy / (n_features × log(2))
  L_entropy = -normalized_entropy × 0.05  # Only during warmup
  ```
- **Weight**: 0.05 (only active during warmup epochs)
- **Note**: Maximizes entropy to encourage uniform gate distribution during exploration phase

### Training Strategy

#### Two-Stage Training

1. **Warmup Stage** (epochs 1 to `warmup_epochs`):
   - Uses **soft thresholding**: `mask = ReLU(p - threshold)`
   - Threshold linearly increases from `warmup_thr_start` (0.0) to `warmup_thr_end` (0.5)
   - `λ_k` linearly increases from `λ_k_start` (0.005) to `λ_k` (0.05)
   - Entropy loss is active to encourage exploration

2. **Hard Top-K Stage** (epochs `warmup_epochs+1` to `epochs`):
   - Uses **hard top-k selection**: Select top-k features with highest gate probabilities
   - Straight-through estimator: `mask = hard_mask - p.detach() + p`
   - `λ_k` is fixed at final value (0.05)
   - Entropy loss is disabled

### Data Pipeline

1. **Segmentation** (`nnunet_segmentation_inference.py`):
   - Uses nnU-Net to segment medical images
   - Majority voting across 5 folds
   - Outputs segmentation masks

2. **Radiomics Extraction** (`torchradiomics_from_ROIs.py`):
   - Extracts radiomics features from segmented ROIs
   - Uses PyRadiomics library
   - Outputs long-format CSV: `case_id, roi_name, feature_name, value`

3. **Feature Selection & Classification** (`train_selector_klgrade.py`):
   - Loads images, radiomics, and KLGrade labels
   - Trains joint model to select features and predict KLGrade
   - Outputs predictions and selected features

### File Structure

```
OA_KLG_TopK/
├── train_selector_klgrade.py    # Main training script
├── models.py                     # Model definitions (FeatureSelector, KLGradeClassifier, JointModel)
├── training_utils.py             # Training utilities (loss computation, metrics)
├── data_loader.py                # Data loading and preprocessing
├── nnunet_segmentation_inference.py  # nnU-Net segmentation inference
├── torchradiomics_from_ROIs.py   # Radiomics feature extraction
├── environment.yml               # Conda environment specification
├── subInfo_train.xlsx            # Training KLGrade labels
├── subInfo_test.xlsx             # Test KLGrade labels
├── output_train/                 # Training radiomics (long format)
│   ├── radiomics_results.csv
│   └── radiomics_results_wide.csv
├── output_test/                  # Test radiomics (long format)
│   ├── radiomics_results.csv
│   └── radiomics_results_wide.csv
└── training_logs/                 # Training outputs
    ├── checkpoints/              # Model checkpoints
    │   ├── best.pth
    │   ├── last.pth
    │   ├── scaler.joblib         # Feature scaler
    │   └── feature_names.json    # Feature name mapping
    ├── logs/
    │   └── metrics.csv            # Training metrics per epoch
    ├── plots/                    # Training plots
    │   ├── loss_curve.png
    │   ├── loss_components.png
    │   ├── metrics_curve.png
    │   ├── gate_stats.png
    │   └── confusion_matrix_*.png
    ├── predictions_train.csv     # Training predictions
    ├── predictions_test.csv      # Test predictions
    ├── selected_features_train.json  # Selected features for each training case
    └── selected_features_test.json   # Selected features for each test case
```

### Usage

#### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate oa_klg_topk
```

#### 2. Training

**Windows:**
```powershell
python train_selector_klgrade.py `
    --images-tr "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" `
    --images-ts "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" `
    --radiomics-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\output_train\radiomics_results.csv" `
    --radiomics-test-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\output_test\radiomics_results.csv" `
    --klgrade-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\subInfo_train.xlsx" `
    --outdir "training_logs" `
    --k 15 `
    --warmup-epochs 30 `
    --epochs 500 `
    --early-stopping-patience 100 `
    --batch-size 8 `
    --lr 1e-4 `
    --use-class-weights `
    --lambda-diversity 0.2
```

**Linux:**
```bash
python train_selector_klgrade.py \
    --images-tr "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
    --images-ts "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTs" \
    --radiomics-train-csv "/home/yaxi/OA_KLG_TopK/output_train/radiomics_results.csv" \
    --radiomics-test-csv "/home/yaxi/OA_KLG_TopK/output_test/radiomics_results.csv" \
    --klgrade-train-csv "/home/yaxi/OA_KLG_TopK/subInfo_train.xlsx" \
    --outdir "training_logs_k30" \
    --k 30 \
    --warmup-epochs 100 \
    --epochs 500 \
    --early-stopping-patience 300 \
    --batch-size 8 \
    --lr 1e-4 \
    --use-class-weights \
    --lambda-diversity 0.2 \
    --device cuda:1
```

#### 3. Key Hyperparameters

- `--k`: Number of top features to select (default: 15)
- `--warmup-epochs`: Number of warmup epochs with soft gating (default: 20)
- `--lambda-k`: Final weight for sparsity loss (default: 0.05)
- `--lambda-k-start`: Initial weight for sparsity loss during warmup (default: 0.005)
- `--lambda-diversity`: Weight for diversity loss (default: 0.1)
- `--warmup-thr-start`: Initial threshold for soft gating (default: 0.0)
- `--warmup-thr-end`: Final threshold for soft gating (default: 0.5)
- `--use-class-weights`: Enable class weighting for imbalanced data
- `--pretrained`: Use pretrained R3D-18 backbone (requires torchvision)

### Output Files

#### Training Metrics (`logs/metrics.csv`)
- `epoch`: Epoch number
- `stage`: Training stage (warmup or hard-topk)
- `train_loss`: Total training loss
- `train_ce_loss`: Classification loss
- `train_loss_k`: Sparsity loss
- `train_loss_diversity`: Diversity loss
- `train_loss_entropy`: Entropy loss (warmup only)
- `train_acc`: Training accuracy
- `val_loss`: Total validation loss
- `val_acc`: Validation accuracy
- `val_macro_f1`: Macro-averaged F1 score
- `val_qwk`: Quadratic Weighted Kappa
- `mean_p_sum`: Mean sum of gate probabilities (should be close to k)

#### Predictions (`predictions_*.csv`)
- `case_id`: Patient case ID
- `pred_class`: Predicted KLGrade (0-4)
- `prob_0` to `prob_4`: Class probabilities

#### Selected Features (`selected_features_*.json`)
- `case_id`: Patient case ID
- `topk_indices`: Indices of selected features
- `topk_names`: Names of selected features (format: `roi_name:feature_name`)
- `topk_gates`: Gate probabilities for selected features

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Average of per-class recall
- **Macro F1**: Unweighted mean of per-class F1 scores
- **Weighted F1**: Weighted mean of per-class F1 scores (by support)
- **QWK (Quadratic Weighted Kappa)**: Agreement metric for ordinal classification
- **AUC**: Macro-averaged one-vs-rest ROC AUC

### Key Features

1. **Image-Conditioned Selection**: Features are selected based on patient-specific images
2. **Patient-Specific Features**: Different patients can have different feature sets
3. **Diversity Regularization**: Prevents collapse to a fixed feature set
4. **Two-Stage Training**: Soft gating during warmup, hard top-k during main training
5. **Class Imbalance Handling**: Optional class weighting
6. **Comprehensive Metrics**: Multiple evaluation metrics including QWK for ordinal classification

---

## 中文

### 项目概述

本项目实现了一个**基于图像条件的特征选择器**结合**KL分级分类器**，用于骨关节炎（OA）严重程度预测。模型学习根据3D医学图像为每位患者选择特定的影像组学特征，然后使用这些选定的特征预测KLGrade（Kellgren-Lawrence分级，0-4级）。

**核心创新**：模型不是使用所有影像组学特征或固定子集，而是根据每位患者的医学图像动态选择最相关的特征，实现个性化特征选择。

### 架构

模型由三个主要组件组成：

#### 1. **特征选择器 (FeatureSelector)** (`models.py`)
- **骨干网络**：3D CNN（可以是预训练的R3D-18或简单的3D CNN）
  - 输入：3D医学图像 `[B, 1, D, H, W]`（通常为 `[B, 1, 32, 128, 128]`）
  - 输出：图像特征 `[B, feature_dim]`（通常为512维）
- **门控头 (Gate Head)**：多层感知机，为每个影像组学特征输出门控logits
  - 输入：图像特征 `[B, feature_dim]`
  - 输出：门控logits `[B, n_features]`（例如，535个特征）
  - 架构：`Linear(512 → 256) → ReLU → Dropout(0.3) → Linear(256 → n_features)`

#### 2. **KL分级分类器 (KLGradeClassifier)** (`models.py`)
- 简单的线性分类器（多类逻辑回归）
- 输入：掩码后的影像组学 `[B, n_features]`（只有选定的特征非零）
- 输出：类别logits `[B, 5]`（5个类别：KLGrade 0-4）
- 架构：`Linear(n_features → 5)`

#### 3. **联合模型 (JointModel)** (`models.py`)
- 结合特征选择器和KL分级分类器
- 两阶段门控机制：
  - **预热阶段**：使用可学习阈值的软阈值化
  - **硬Top-K阶段**：使用直通估计器的硬top-k选择

### 损失函数组件

总损失由四个组件组成：

```
总损失 = L_cls + λ_k × L_k + λ_diversity × L_diversity + L_entropy
```

#### 1. **分类损失 (L_cls)**
- **类型**：CrossEntropyLoss（如果启用 `--use-class-weights` 则为加权）
- **目的**：最小化KLGrade预测的分类误差
- **公式**：
  ```
  L_cls = -Σ y_true × log(softmax(logits))
  ```
- **权重**：1.0（无缩放）

#### 2. **稀疏性损失 (L_k)**
- **类型**：目标k正则化
- **目的**：鼓励模型恰好选择k个特征
- **公式**：
  ```
  L_k = mean((Σ p_i - k)²)
  ```
  其中 `p_i = sigmoid(gate_logits_i)` 是特征i的门控概率
- **权重**：`λ_k`（默认：0.05，预热期间从 `λ_k_start` = 0.005 开始）
- **调度**：在预热轮次期间从 `λ_k_start` 线性插值到 `λ_k`

#### 3. **多样性损失 (L_diversity)**
- **类型**：成对余弦相似度惩罚
- **目的**：鼓励不同患者选择不同的特征（防止 collapse 到固定特征集）
- **公式**：
  ```
  p_normalized = normalize(p, p=2, dim=1)  # [B, n_features]
  similarity_matrix = p_normalized @ p_normalized.T  # [B, B]
  L_diversity = mean(similarity_matrix[upper_triangle])
  ```
- **权重**：`λ_diversity`（默认：0.1，可通过 `--lambda-diversity` 配置）
- **注意**：惩罚同一批次中不同患者门控向量之间的高相似度

#### 4. **熵损失 (L_entropy)**
- **类型**：负熵正则化
- **目的**：在预热期间鼓励探索（防止早期 collapse）
- **公式**：
  ```
  entropy = -mean(Σ [p_i × log(p_i + ε) + (1-p_i) × log(1-p_i + ε)])
  normalized_entropy = entropy / (n_features × log(2))
  L_entropy = -normalized_entropy × 0.05  # 仅在预热期间
  ```
- **权重**：0.05（仅在预热轮次期间激活）
- **注意**：最大化熵以在探索阶段鼓励均匀的门控分布

### 训练策略

#### 两阶段训练

1. **预热阶段**（第1轮到 `warmup_epochs` 轮）：
   - 使用**软阈值化**：`mask = ReLU(p - threshold)`
   - 阈值从 `warmup_thr_start` (0.0) 线性增加到 `warmup_thr_end` (0.5)
   - `λ_k` 从 `λ_k_start` (0.005) 线性增加到 `λ_k` (0.05)
   - 熵损失激活以鼓励探索

2. **硬Top-K阶段**（第 `warmup_epochs+1` 轮到 `epochs` 轮）：
   - 使用**硬top-k选择**：选择门控概率最高的k个特征
   - 直通估计器：`mask = hard_mask - p.detach() + p`
   - `λ_k` 固定在最终值 (0.05)
   - 熵损失禁用

### 数据流程

1. **分割** (`nnunet_segmentation_inference.py`)：
   - 使用nnU-Net分割医学图像
   - 5折多数投票
   - 输出分割掩码

2. **影像组学提取** (`torchradiomics_from_ROIs.py`)：
   - 从分割的ROI中提取影像组学特征
   - 使用PyRadiomics库
   - 输出长格式CSV：`case_id, roi_name, feature_name, value`

3. **特征选择与分类** (`train_selector_klgrade.py`)：
   - 加载图像、影像组学和KLGrade标签
   - 训练联合模型以选择特征并预测KLGrade
   - 输出预测和选定的特征

### 文件结构

```
OA_KLG_TopK/
├── train_selector_klgrade.py    # 主训练脚本
├── models.py                     # 模型定义（特征选择器、KL分级分类器、联合模型）
├── training_utils.py             # 训练工具（损失计算、指标）
├── data_loader.py                # 数据加载和预处理
├── nnunet_segmentation_inference.py  # nnU-Net分割推理
├── torchradiomics_from_ROIs.py   # 影像组学特征提取
├── environment.yml               # Conda环境规范
├── subInfo_train.xlsx            # 训练KLGrade标签
├── subInfo_test.xlsx             # 测试KLGrade标签
├── output_train/                 # 训练影像组学（长格式）
│   ├── radiomics_results.csv
│   └── radiomics_results_wide.csv
├── output_test/                  # 测试影像组学（长格式）
│   ├── radiomics_results.csv
│   └── radiomics_results_wide.csv
└── training_logs/                # 训练输出
    ├── checkpoints/              # 模型检查点
    │   ├── best.pth
    │   ├── last.pth
    │   ├── scaler.joblib         # 特征缩放器
    │   └── feature_names.json    # 特征名称映射
    ├── logs/
    │   └── metrics.csv           # 每轮训练指标
    ├── plots/                    # 训练图表
    │   ├── loss_curve.png
    │   ├── loss_components.png
    │   ├── metrics_curve.png
    │   ├── gate_stats.png
    │   └── confusion_matrix_*.png
    ├── predictions_train.csv     # 训练预测
    ├── predictions_test.csv      # 测试预测
    ├── selected_features_train.json  # 每个训练案例的选定特征
    └── selected_features_test.json   # 每个测试案例的选定特征
```

### 使用方法

#### 1. 环境设置

```bash
conda env create -f environment.yml
conda activate oa_klg_topk
```

#### 2. 训练

**Windows:**
```powershell
python train_selector_klgrade.py `
    --images-tr "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" `
    --images-ts "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" `
    --radiomics-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\output_train\radiomics_results.csv" `
    --radiomics-test-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\output_test\radiomics_results.csv" `
    --klgrade-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\subInfo_train.xlsx" `
    --outdir "training_logs" `
    --k 15 `
    --warmup-epochs 30 `
    --epochs 500 `
    --early-stopping-patience 100 `
    --batch-size 8 `
    --lr 1e-4 `
    --use-class-weights `
    --lambda-diversity 0.2
```

**Linux:**
```bash
python train_selector_klgrade.py \
    --images-tr "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
    --images-ts "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTs" \
    --radiomics-train-csv "/home/yaxi/OA_KLG_TopK/output_train/radiomics_results.csv" \
    --radiomics-test-csv "/home/yaxi/OA_KLG_TopK/output_test/radiomics_results.csv" \
    --klgrade-train-csv "/home/yaxi/OA_KLG_TopK/subInfo_train.xlsx" \
    --outdir "training_logs_k30" \
    --k 30 \
    --warmup-epochs 100 \
    --epochs 500 \
    --early-stopping-patience 300 \
    --batch-size 8 \
    --lr 1e-4 \
    --use-class-weights \
    --lambda-diversity 0.2 \
    --device cuda:1
```

#### 3. 关键超参数

- `--k`：要选择的top特征数量（默认：15）
- `--warmup-epochs`：使用软门控的预热轮次数（默认：20）
- `--lambda-k`：稀疏性损失的最终权重（默认：0.05）
- `--lambda-k-start`：预热期间稀疏性损失的初始权重（默认：0.005）
- `--lambda-diversity`：多样性损失的权重（默认：0.1）
- `--warmup-thr-start`：软门控的初始阈值（默认：0.0）
- `--warmup-thr-end`：软门控的最终阈值（默认：0.5）
- `--use-class-weights`：为不平衡数据启用类别加权
- `--pretrained`：使用预训练的R3D-18骨干网络（需要torchvision）

### 输出文件

#### 训练指标 (`logs/metrics.csv`)
- `epoch`：轮次数
- `stage`：训练阶段（warmup 或 hard-topk）
- `train_loss`：总训练损失
- `train_ce_loss`：分类损失
- `train_loss_k`：稀疏性损失
- `train_loss_diversity`：多样性损失
- `train_loss_entropy`：熵损失（仅预热）
- `train_acc`：训练准确率
- `val_loss`：总验证损失
- `val_acc`：验证准确率
- `val_macro_f1`：宏平均F1分数
- `val_qwk`：二次加权Kappa
- `mean_p_sum`：门控概率的平均和（应接近k）

#### 预测 (`predictions_*.csv`)
- `case_id`：患者案例ID
- `pred_class`：预测的KLGrade (0-4)
- `prob_0` 到 `prob_4`：类别概率

#### 选定特征 (`selected_features_*.json`)
- `case_id`：患者案例ID
- `topk_indices`：选定特征的索引
- `topk_names`：选定特征的名称（格式：`roi_name:feature_name`）
- `topk_gates`：选定特征的门控概率

### 评估指标

- **准确率 (Accuracy)**：总体分类准确率
- **平衡准确率 (Balanced Accuracy)**：每类召回率的平均值
- **宏F1 (Macro F1)**：每类F1分数的未加权平均值
- **加权F1 (Weighted F1)**：每类F1分数的加权平均值（按支持度）
- **QWK (二次加权Kappa)**：用于序数分类的一致性指标
- **AUC**：宏平均的一对多ROC AUC

### 关键特性

1. **基于图像的条件选择**：根据患者特定图像选择特征
2. **患者特定特征**：不同患者可以有不同的特征集
3. **多样性正则化**：防止 collapse 到固定特征集
4. **两阶段训练**：预热期间软门控，主训练期间硬top-k
5. **类别不平衡处理**：可选的类别加权
6. **综合指标**：包括用于序数分类的QWK在内的多个评估指标

