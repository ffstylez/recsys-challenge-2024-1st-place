# Sugawarya Pipeline Documentation

This document explains the final ensemble and submission pipeline that combines predictions from KAMI and KFujikawa components into the winning solution.

## Pipeline Overview

The sugawarya pipeline represents the **final ensemble stage** of the 3-component winning system:

1. **KAMI Pipeline**: Feature engineering + gradient boosting models
2. **KFujikawa Pipeline**: Neural network models with minimal preprocessing  
3. **Sugawarya Pipeline**: Advanced ensemble methods + submission generation

---

## Pipeline Architecture

### Sequential Execution Flow
```bash
# Complete ensemble pipeline
poetry run python src/weighted_mean.py      # Stage 1: Weight optimization
poetry run python src/stacking.py           # Stage 2: Meta-learning  
poetry run python src/make_submission.py    # Stage 3: Final submission
```

### Input Dependencies
**Required Predictions from Base Pipelines**:
```
# KAMI Models  
kami/output/experiments/015_train_third/large067_001/
├── validation_result_first.parquet     # S1 split for weight optimization
└── test_result_third.parquet          # S3 split for final predictions

kami/output/experiments/016_catboost/large067/  
├── validation_result_first.parquet
└── test_result_third.parquet

# KFujikawa Models
kfujikawa/data/kfujikawa/v1xxx_training/v1157_111_fix_past_v2/fold_0/predictions/
└── validation.parquet

kfujikawa/data/kfujikawa/v1xxx_training/v1157_111_fix_past_v2/fold_2/predictions/
└── test.parquet

kfujikawa/data/kfujikawa/v1xxx_training/v1170_111_L8_128d/fold_{0,2}/predictions/
kfujikawa/data/kfujikawa/v1xxx_training/v1184_111_PL_bert_L4_256d/fold_{0,2}/predictions/
```

---

## Stage 1: Weighted Mean Optimization

### Purpose
Find optimal linear combination weights for base model predictions using Bayesian optimization.

### Process
1. **Load Base Predictions**: Read predictions from all 5 base models
2. **Validation Optimization**: Use Optuna to find optimal weights on validation data
3. **Test Application**: Apply optimized weights to test predictions
4. **Output**: Linearly combined predictions with optimal weights

### Key Implementation
```python
# Model configuration
base_models = [
    "kami/015_train_third",              # Neural component (weight ~2)
    "kami/016_catboost",                # CatBoost component (weight ~1)  
    "kfujikawa/v1157_fix_past_v2",      # Fixed past model
    "kfujikawa/v1170_L8_128d",          # Base FastFormer  
    "kfujikawa/v1184_PL_bert_L4_256d"   # BERT + pseudo-labeling
]

# Bayesian optimization with Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective_function, n_trials=200)

# Objective: Impression-level AUC maximization
def objective(trial):
    weights = {model: trial.suggest_float(model, 0, 1) for model in base_models}
    ensemble_pred = sum(pred[model] * weights[model] for model in base_models)
    return compute_impression_auc(ensemble_pred, targets)
```

### Optimization Strategy
- **Metric**: Impression-level AUC (exact competition metric)
- **Sampling**: Every 100th impression for computational efficiency
- **Search Space**: [0,1] weight range per model
- **Algorithm**: TPE (Tree-structured Parzen Estimator) via Optuna

---

## Stage 2: Stacking Meta-Learning

### Purpose
Train a meta-learner (LightGBM) to optimally combine base model predictions using sophisticated feature engineering.

### Advanced Feature Engineering

#### Base Features (5 models × multiple variants)
```python
raw_predictions = ["pred_015", "pred_016", "pred_v1157", "pred_v1170", "pred_v1184"]
```

#### Statistical Features (20 features)
```python
# Within-impression statistics for each base model
for model in base_models:
    impression_stats = [
        f"{model}_mean",     # Mean prediction within impression
        f"{model}_max",      # Max prediction within impression
        f"{model}_min",      # Min prediction within impression  
        f"{model}_std"       # Standard deviation within impression
    ]
```

#### Normalization Features (10 features)
```python
# Normalized predictions within each impression
normalization_features = [
    f"{model}_zscore",                  # (pred - mean) / std
    f"{model}_normed_in_impression"     # (pred - min) / (max - min)
]
```

#### Ranking Features (20 features)
```python
# Ranking within impressions
ranking_features = [
    f"{model}_rank",            # Ascending rank (1 = lowest score)
    f"{model}_rank_desc",       # Descending rank (1 = highest score)  
    f"{model}_normedrank",      # Normalized rank [0,1]
    f"{model}_normedrank_desc"  # Normalized descending rank [0,1]
]
```

#### Interaction Features (280+ features)
```python
# Pairwise interactions between all model combinations
for model1, model2 in itertools.combinations(base_models, 2):
    for feature_type in [raw, rank, normedrank, zscore, normalized]:
        interactions = [
            f"{model1}_{model2}_diff",   # Difference
            f"{model1}_{model2}_ratio",  # Ratio
            f"{model1}_{model2}_max",    # Maximum
            f"{model1}_{model2}_min"     # Minimum
        ]
```

#### Count Features (9 features)
```python
behavioral_features = [
    "impression_count",     # Articles per impression
    "article_count",        # User-article interaction count
    "user_count",          # Total user interactions
    "pred_mean"            # Mean prediction across all models
]
```

**Total Meta-Features**: ~350+ engineered features from 5 base predictions

### Meta-Learner Architecture

#### LightGBM Configuration
```python
lgb_params = {
    "objective": "lambdarank",        # Learning-to-rank for news recommendation
    "metric": "ndcg",                # NDCG optimization
    "ndcg_at": [5, 10, 20],         # Multiple ranking depths
    "learning_rate": 0.1,            # Moderate learning rate
    "feature_fraction": 0.8,         # Feature sampling for robustness
    "bagging_fraction": 0.8,         # Row sampling for robustness
    "bagging_freq": 1,               # Enable bagging
    "seed": 19930820,                # Reproducibility
    "max_bin": 1024,                 # High precision binning
    "verbose": 1
}
```

#### Cross-Validation Strategy
```python
# Group K-Fold to prevent impression leakage
gkf = GroupKFold(n_splits=4)
for fold, (train_idx, valid_idx) in enumerate(gkf.split(data, groups=impression_ids)):
    # Train meta-learner on current fold
    # Validate on held-out impressions
    # Average predictions across all folds
```

#### Training Optimization
- **Data Sampling**: Every 10th impression for training efficiency
- **Early Stopping**: 50 rounds without improvement
- **Ranking Groups**: Properly grouped by impression_id for LambdaRank
- **Ensemble**: Average 4 cross-validation models for final predictions

---

## Stage 3: Final Submission Generation

### Ensemble Combination Strategy
```python
# Simple additive combination of both ensemble methods
final_prediction = stacking_prediction + weighted_mean_prediction
```

**Rationale**: 
- Weighted mean provides stable linear baseline
- Stacking captures complex non-linear patterns
- Addition leverages strengths of both approaches

### Submission Format Conversion

#### Competition Format Requirements
```
Format: impression_id [rank1,rank2,rank3,...]
Example:
    12345 [1,3,2,4,5]    # Article ranks within impression 12345
    12346 [2,1,4,3]      # Article ranks within impression 12346
```

#### Ranking Conversion Process
```python
def create_competition_submission():
    # Group by impression
    grouped_data = final_predictions.group_by("impression_id")
    
    # Convert scores to rankings (1=best, 2=second best, ...)
    for impression_group in grouped_data:
        scores = impression_group["final_prediction"]
        rankings = rank_predictions_by_score(scores)  # np.argsort magic
        
    # Write to competition format
    write_submission_file(impression_ids, rankings_list)
    
    # Create ZIP file for submission
    zip_submission_file("predictions.txt", "v999_final_submission.zip")
```

#### Parallel Processing
```python
# Multiprocessing for fast ranking computation
with multiprocessing.Pool(cpu_count()) as pool:
    rankings = list(pool.imap_unordered(get_rank, impression_data))
```

---

## Configuration & Execution

### Environment Setup
```bash
# Dependency management with Poetry
pip install -U poetry
poetry install

# Required packages
dependencies = [
    "polars",          # High-performance data processing
    "optuna",          # Bayesian optimization
    "lightgbm",        # Meta-learner
    "scikit-learn",    # Cross-validation, metrics
    "multiprocessing", # Parallel processing
]
```

### Debug Mode Support
```bash
# Fast development with reduced data
poetry run python src/weighted_mean.py --debug      # Every 1000th impression
poetry run python src/stacking.py --debug           # Every 1000th impression  
poetry run python src/make_submission.py --debug    # Same sampling
```

### Production Execution
```bash
# Full pipeline for final submission
./run_ensemble.sh

# Which executes:
poetry run python src/weighted_mean.py
poetry run python src/stacking.py  
poetry run python src/make_submission.py
```

---

## Data Flow & Dependencies

### Input Requirements
```
Base Model Predictions (5 models × 2 splits each):
├── KAMI validation/test predictions     # 2 models
├── KFujikawa validation/test predictions # 3 models
└── Competition behavior data            # For joining

Original Datasets (for target labels):
├── behaviors.parquet                    # Impression targets  
└── Competition evaluation format
```

### Output Generation  
```
sugawarya/output/
├── test_weighted_mean.parquet          # Stage 1 output
├── test_stacking.parquet              # Stage 2 output
└── v999_final_submission.zip          # Stage 3 final submission
    └── predictions.txt                # Competition format
```

---

## Performance & Optimization

### Computational Efficiency
- **Memory Management**: Smart sampling reduces memory footprint by 10-100x
- **Multiprocessing**: Parallel ranking computation across CPU cores
- **Polars**: High-performance data processing (faster than pandas)
- **Efficient Joins**: Optimized prediction loading and merging

### Validation Strategy
- **Temporal Splits**: Respects time-based data splits from base models
- **Group K-Fold**: Prevents impression-level data leakage in stacking
- **Competition Metrics**: Direct optimization of impression-level AUC and NDCG
- **Out-of-fold**: Proper cross-validation for meta-learner training

### Hardware Requirements
- **CPU**: High-core count beneficial for multiprocessing
- **Memory**: 896 GB recommended for full-scale processing
- **Storage**: Fast I/O for large parquet file operations

---

## Key Design Principles

1. **Ensemble Diversity**: Combines fundamentally different modeling approaches
2. **Advanced Meta-Learning**: 350+ engineered features for optimal combination
3. **Competition Optimization**: Direct optimization of evaluation metrics
4. **Robust Validation**: Multiple cross-validation strategies
5. **Production Ready**: Efficient, scalable, reproducible pipeline
6. **Hierarchical Ensemble**: Two-stage approach maximizes performance gains

---

## Technical Innovation

### Novel Contributions
1. **Meta-Feature Engineering**: Extensive feature engineering on prediction space
2. **Dual Ensemble Strategy**: Combines linear (weighted) and non-linear (stacking) methods  
3. **Bayesian Weight Optimization**: Principled approach to ensemble weight selection
4. **Production Scalability**: Efficient handling of large-scale recommendation data

### Why This Approach Works
- **Model Complementarity**: KAMI (features) + KFujikawa (neural) capture different patterns
- **Ensemble Sophistication**: Beyond simple averaging to capture complex interactions
- **Direct Metric Optimization**: Optimizes exact competition evaluation metrics
- **Robust Validation**: Multiple validation strategies prevent overfitting

This pipeline represents the state-of-the-art in ensemble learning for large-scale recommendation systems, demonstrating how sophisticated model combination can achieve winning performance on challenging real-world datasets.