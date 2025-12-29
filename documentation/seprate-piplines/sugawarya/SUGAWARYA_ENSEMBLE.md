# Sugawarya Ensemble & Submission Pipeline Documentation

Comprehensive guide to the final ensemble methods and submission generation for the winning news recommendation system.

## Philosophy: Advanced Ensemble Learning

**Sugawarya implements the final stage of the 3-component pipeline:**
- Combines predictions from KAMI (gradient boosting) and KFujikawa (neural networks)
- Uses sophisticated ensemble techniques: weighted averaging and stacking
- Optimizes ensemble weights using Optuna hyperparameter optimization
- Generates final competition submissions with proper ranking

**Role in Overall Pipeline**: While KAMI focuses on feature engineering and KFujikawa on neural modeling, Sugawarya specializes in **ensemble optimization** and **prediction fusion**.

---

## Pipeline Components

### 1. Weighted Mean Ensemble (`weighted_mean.py`)

#### Purpose
Optimizes linear combination weights for base model predictions using Bayesian optimization.

#### Key Features
- **Optuna-based optimization**: Uses Bayesian optimization to find optimal ensemble weights
- **AUC maximization**: Optimizes for impression-level AUC (competition metric)
- **Validation-based tuning**: Finds weights on validation set, applies to test set
- **Memory-efficient sampling**: Uses data sampling to handle large datasets

#### Process Flow
```python
# Phase 1: Weight Optimization on Validation Data
validation_predictions = load_base_models(validation_split)
optimal_weights = optuna_optimization(validation_predictions, target_labels)

# Phase 2: Apply Weights to Test Data  
test_predictions = load_base_models(test_split)
final_predictions = weighted_average(test_predictions, optimal_weights)
```

#### Implementation Details
```python
def optimize_weight_by_optuna(explode_pred_df, pred_cols):
    # Smart sampling for efficiency
    mini_df = explode_pred_df.filter(
        explode_pred_df["impression_id"].is_in(
            explode_pred_df["impression_id"].unique().sort()[::100]  # Every 100th impression
        )
    )
    
    def objective(trial):
        # Suggest weights for each base model
        tmp_params = {}
        for pred_name in pred_cols:
            tmp_params[pred_name] = trial.suggest_float(pred_name, 0, 1)
        
        # Create weighted ensemble prediction
        pred_ensemble = sum(df[pred_name] * weight for pred_name, weight in tmp_params.items())
        
        # Optimize impression-level AUC
        return get_impression_auc_mean(df_with_predictions, target_col="target", pred_col="pred_ensemble")
    
    # Bayesian optimization with 200 trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    return study.best_params
```

#### Models Combined
- **KAMI Models**:
  - `015_train_third` - Neural network component
  - `016_catboost` - Gradient boosting component
- **KFujikawa Models**:
  - `v1157_111_fix_past_v2` - Fixed past model
  - `v1170_111_L8_128d` - Base FastFormer
  - `v1184_111_PL_bert_L4_256d` - BERT with pseudo-labeling

---

### 2. Stacking Ensemble (`stacking.py`)

#### Purpose
Implements meta-learning approach using LightGBM to learn optimal combination of base model predictions.

#### Key Innovation: Feature Engineering on Predictions
Instead of simple averaging, creates rich features from base model outputs:

##### Meta-Features Created
```python
# Base features from raw predictions
base_features = ["pred_kami_015", "pred_kami_016", "pred_kfujikawa_1157", ...]

# Statistical aggregation within impressions
impression_stats = [
    f"{pred}_mean",      # Mean prediction in impression
    f"{pred}_max",       # Maximum prediction in impression  
    f"{pred}_min",       # Minimum prediction in impression
    f"{pred}_std"        # Standard deviation in impression
]

# Normalization features
normalization = [
    f"{pred}_zscore",                    # Z-score within impression
    f"{pred}_normed_in_impression"       # Min-max normalization within impression
]

# Ranking features  
ranking = [
    f"{pred}_rank",                      # Rank within impression (ascending)
    f"{pred}_rank_desc",                 # Rank within impression (descending)
    f"{pred}_normedrank",                # Normalized rank [0,1]
    f"{pred}_normedrank_desc"            # Normalized descending rank [0,1]
]

# Interaction features between model pairs
for model1, model2 in combinations(base_models, 2):
    interactions = [
        f"{model1}_{model2}_diff",       # Prediction difference
        f"{model1}_{model2}_ratio",      # Prediction ratio
        f"{model1}_{model2}_max",        # Maximum of pair
        f"{model1}_{model2}_min"         # Minimum of pair
    ]
```

#### Stacking Architecture
```python
# Meta-learner configuration
lgb_params = {
    "objective": "lambdarank",           # Learning-to-rank objective
    "metric": "ndcg",                   # NDCG optimization  
    "ndcg_at": [5, 10, 20],            # Multiple ranking depths
    "learning_rate": 0.1,
    "feature_fraction": 0.8,            # Feature sampling
    "bagging_fraction": 0.8,            # Row sampling
    "seed": 19930820,                   # Reproducibility
}

# Cross-validation setup
gkf = GroupKFold(n_splits=4)            # Group by impression_id
# Prevents data leakage across impressions
```

#### Training Process
1. **Feature Engineering**: Transform base predictions into 100+ meta-features
2. **Cross-Validation**: 4-fold GroupKFold ensuring impression-level splits
3. **Data Sampling**: Use every 10th impression for training efficiency
4. **LambdaRank Training**: Optimize for ranking quality (NDCG)
5. **Ensemble Prediction**: Average predictions across CV folds

---

### 3. Final Submission Generation (`make_submission.py`)

#### Purpose
Combines stacking and weighted mean predictions, converts to competition format.

#### Final Ensemble Strategy
```python
# Combine both ensemble methods
final_prediction = stacking_prediction + weighted_mean_prediction
```

#### Submission Format Generation
```python
def create_submission():
    # Group predictions by impression
    grouped_predictions = group_by_impression(final_predictions)
    
    # Rank articles within each impression  
    ranked_predictions = []
    for impression in grouped_predictions:
        # Convert scores to rankings (1=best, 2=second best, etc.)
        rankings = rank_predictions_by_score(impression.scores)
        ranked_predictions.append(rankings)
    
    # Competition format: impression_id [rank1,rank2,rank3,...]
    write_submission_file(impression_ids, ranked_predictions)
```

#### Multiprocessing Optimization
```python
# Parallel ranking computation
with multiprocessing.Pool(cpu_count()) as pool:
    rankings = pool.imap_unordered(get_rank, impression_data)
```

---

## Technical Implementation

### Data Flow Architecture
```
Base Models (KAMI + KFujikawa)
          ↓
    [weighted_mean.py]
          ↓
   Validation: Optuna Weight Optimization  
   Test: Apply Optimal Weights
          ↓
    [stacking.py]  
          ↓
   Feature Engineering on Predictions
   LightGBM Meta-Learner Training
          ↓
    [make_submission.py]
          ↓
   Combine Stacking + Weighted Mean
   Convert to Competition Rankings
          ↓
     Final Submission ZIP
```

### Technology Stack
- **Optuna**: Bayesian hyperparameter optimization for ensemble weights
- **LightGBM**: LambdaRank meta-learner for stacking
- **Polars**: High-performance data processing and joins
- **Multiprocessing**: Parallel ranking computation
- **GroupKFold**: Impression-aware cross-validation

### Memory Management
```python
# Smart sampling strategies
debug_sampling = impression_ids[::1000]     # Every 1000th impression in debug
training_sampling = impression_ids[::10]    # Every 10th impression for stacking training  
optuna_sampling = impression_ids[::100]     # Every 100th impression for weight optimization
```

---

## Ensemble Techniques Comparison

| Method | Approach | Optimization | Complexity | Interpretability |
|--------|----------|--------------|------------|------------------|
| **Weighted Mean** | Linear combination | Bayesian optimization | Low | High |
| **Stacking** | Meta-learning | Cross-validation + LambdaRank | High | Medium |
| **Final Ensemble** | Additive combination | None (simple sum) | Medium | Medium |

---

## Performance Validation

### Metrics Used
- **Impression-level AUC**: Primary optimization target for weighted averaging
- **NDCG@[5,10,20]**: Ranking quality metrics for stacking
- **Cross-validation**: 4-fold validation with impression-level grouping

### Ensemble Benefits
1. **Diversity**: Combines different model types (neural nets + gradient boosting)
2. **Complementary Strengths**: Neural models capture complex patterns, gradient boosting handles tabular features
3. **Robustness**: Multiple ensemble methods reduce overfitting risk
4. **Competition Optimization**: Direct optimization of competition metrics

---

## Execution Pipeline

### Complete Ensemble Pipeline
```bash
# 1. Weighted Mean Optimization
poetry run python src/weighted_mean.py

# 2. Stacking Meta-Learning  
poetry run python src/stacking.py

# 3. Final Submission Generation
poetry run python src/make_submission.py
```

### Debug Mode
```bash
# Fast development with data sampling
poetry run python src/weighted_mean.py --debug
poetry run python src/stacking.py --debug  
poetry run python src/make_submission.py --debug
```

---

## Output Structure

```
sugawarya/output/
├── test_weighted_mean.parquet      # Optuna-optimized weighted predictions
├── test_stacking.parquet          # LightGBM meta-learner predictions  
└── v999_final_submission.zip      # Competition-ready submission file
    └── predictions.txt            # Format: impression_id [rank1,rank2,...]
```

---

## Key Design Principles

1. **Hierarchical Ensemble**: Two-stage ensemble (weighted + stacking) for maximum performance
2. **Metric Optimization**: Direct optimization of competition metrics (AUC, NDCG)
3. **Cross-Validation**: Rigorous validation with impression-level grouping
4. **Computational Efficiency**: Smart sampling and multiprocessing for scalability
5. **Reproducibility**: Fixed random seeds and deterministic procedures

---

## Ensemble Insights

### Why This Approach Works
1. **Model Diversity**: KAMI and KFujikawa use fundamentally different approaches
2. **Feature Complementarity**: Hand-crafted features vs. learned representations
3. **Advanced Meta-Learning**: Stacking learns complex combination patterns
4. **Multiple Validation**: Both Bayesian optimization and cross-validation
5. **Competition-Specific**: Optimized for exact evaluation metrics used

### Performance Impact
- **Weighted Mean**: Provides stable baseline ensemble
- **Stacking**: Captures non-linear interactions between base models
- **Final Combination**: Additive benefit from both ensemble types
- **Result**: State-of-the-art performance on RecSys Challenge 2024

This ensemble strategy demonstrates that sophisticated model combination techniques can extract significant additional performance from diverse base models, representing the final crucial step in achieving winning results.