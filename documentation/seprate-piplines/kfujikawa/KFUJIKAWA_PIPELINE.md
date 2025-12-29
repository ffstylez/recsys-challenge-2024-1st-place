# KFujikawa Pipeline Documentation

This document explains the complete neural network-focused pipeline for the news recommendation system, covering preprocessing, training, ablation studies, and ensemble methods.

## Pipeline Overview

The kfujikawa pipeline consists of 4 main phases executed sequentially:

1. **Data Preprocessing** (`v0xxx_preprocess/`)
2. **Model Training** (`v1xxx_training/`)
3. **Ablation Studies** (`v2xxx_ablation/`)
4. **Ensemble Methods** (`v8xxx_ensemble/`)

---

## 1. Data Preprocessing Phase (`v0xxx_preprocess/`)

### Execution Order
Run via: `./run.sh` which executes scripts in sequence

### 1.1 Core Dataset Processing
**Files**: `v0100_articles.py`, `v0200_users.py`, `v0300_impressions.py`
- **Articles**: Processes article metadata, adds temporal features (published_ts, weekday), creates indices
- **Users**: User profile preprocessing and feature extraction
- **Impressions**: Impression log processing and candidate generation

### 1.2 Enhanced Feature Engineering
**Files**: `v0101_*`, `v0102_*`, `v0103_*`, `v0201_*`, `v0301_*`
- **Article Features**: In-view statistics, metadata mappings, history counts
- **User Features**: Interaction patterns within data splits
- **Impression Features**: User impression counts and targeting

**Output**: Cleaned, indexed datasets with rich features in structured parquet format

---

## 2. Model Training Phase (`v1xxx_training/`)

### Training Timeline & Data Splits

**Fold Configuration**:
- **Fold 0**: S1 split (May 18-24 train, May 25-31 validation) → For hyperparameter tuning
- **Fold 2**: S3 split (May 18-31 train) → For final model training

### 2.1 Primary Models

#### FastFormer Base Model (`v1170_111_L8_128d.py`)
- **Architecture**: FastFormer transformer (8 layers, 128 dimensions)
- **Configuration**: `111` = future_imp + future_stats + past_imp enabled
- **Purpose**: Base neural recommendation model

#### Enhanced FastFormer (`v1174_111_L8_128d_smpl3_drophist.py`)
- **Architecture**: Same as v1170 but with optimizations:
  - `smpl3`: Advanced sampling strategy
  - `drophist`: Dropout history mechanism
- **Performance**: Best single model in the pipeline

#### BERT Variant (`v1184_111_PL_bert_L4_256d.py`)
- **Architecture**: BERT-based (4 layers, 256 dimensions)
- **Features**: Pseudo-labeling (PL) for enhanced training

### 2.2 Training Process
```bash
# For each model, run both folds:
python model.py train common.fold=0  # S1 split for hyperparameters
python model.py train common.fold=2  # S3 split for final training
python model.py predict --split validation common.fold=0
python model.py predict --split test common.fold=2
```

**Training Configuration**:
- PyTorch Lightning framework
- Max history: 10 articles per user
- Batch size: 64
- Epochs: 4-5
- GPU training with configurable device allocation

---

## 3. Ablation Studies Phase (`v2xxx_ablation/`)

### Purpose
Scientific validation of each component's contribution to model performance

### 3.1 Feature Ablations
- **`v2001_000_v1174_noleak.py`**: Tests without future information leakage
- **`v2003_111_v1174_no-drophist.py`**: Tests impact of dropout history
- **`v2004_111_v1174_no-imp.py`**: Tests without impression features  
- **`v2006_111_v1174_no-sample.py`**: Tests without sampling strategies

### 3.2 Architecture Ablations
- **`v2005_111_v1174_no-transformer.py`**: Tests without transformer architecture
- **`v2007_111_v1174_no-transformer-same-parameter.py`**: Parameter-controlled comparison

### Insights Generated
Each ablation quantifies the performance contribution of specific components, validating the research approach and informing future model development.

---

## 4. Ensemble Methods Phase (`v8xxx_ensemble/`)

### 4.1 Cross-Pipeline Ensemble (`v8004_015_016_v1170_v1174.py`)

**Combined Models**:
```python
# KAMI Pipeline Models (gradient boosting)
015_train_third: weight = 2    # Neural network component
016_catboost: weight = 1       # CatBoost component

# KFujikawa Pipeline Models (neural networks)  
v1174: weight = 2              # Enhanced FastFormer
v1170: weight = 1              # Base FastFormer
```

**Ensemble Strategy**:
- Weighted average of predictions
- Higher weights for better-performing models (v1174 > v1170)
- Combines neural + gradient boosting approaches

**Final Performance**: AUC = 0.8791

### 4.2 Execution
```bash
./run_ensemble.sh
# Runs ensemble for validation and test splits
```

---

## Configuration System

### File Naming Convention
**Format**: `v{phase}{experiment}_{features}_{architecture}.py`

**Examples**:
- `v1174_111_L8_128d_smpl3_drophist.py`
  - `v1174`: Training experiment 1174
  - `111`: Feature flags (future_imp + future_stats + past_imp)
  - `L8_128d`: 8 layers, 128 dimensions
  - `smpl3_drophist`: Sampling + dropout history optimizations

### Technology Stack
- **PyTorch Lightning**: Training infrastructure
- **OmegaConf**: Configuration management  
- **Polars**: High-performance data processing
- **FastFormer/BERT**: Transformer architectures
- **Poetry**: Dependency management

---

## Pipeline Execution

### Complete Pipeline
```bash
# 1. Preprocessing
cd v0xxx_preprocess && ./run.sh

# 2. Training
cd v1xxx_training && ./run_scratch.sh  

# 3. Ensemble  
cd v8xxx_ensemble && ./run_ensemble.sh
```

### Individual Model Training
```bash
# Train specific model
poetry run python v1174_111_L8_128d_smpl3_drophist.py train common.fold=2

# Generate predictions
poetry run python v1174_111_L8_128d_smpl3_drophist.py predict --split test common.fold=2
```

---

## Key Design Principles

1. **Temporal Awareness**: Respects time-based data splits (S1, S3)
2. **Scientific Rigor**: Comprehensive ablation studies validate each component
3. **Modular Architecture**: Clear separation of preprocessing, training, validation, ensemble
4. **Ensemble Strategy**: Combines multiple model types and pipelines
5. **Reproducibility**: Configuration-driven experiments with version control

## Output Structure
```
outputs/
├── v0xxx_preprocess/           # Processed datasets
├── v1xxx_training/            
│   ├── v1170/fold_{0,2}/      # Base model outputs
│   └── v1174/fold_{0,2}/      # Enhanced model outputs
├── v2xxx_ablation/            # Ablation study results
└── v8xxx_ensemble/            # Final ensemble predictions
```

This pipeline represents a research-grade neural network approach with comprehensive validation and production-ready ensemble methods for news recommendation.