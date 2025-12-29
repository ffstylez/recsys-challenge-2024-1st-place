# Comprehensive Pipeline Summary: "Enhancing News Recommendation with Transformers and Ensemble Learning"

## 1. Data Splitting Strategy

### Research Paper Approach
The team implemented a sophisticated **3-stage temporal splitting strategy** (S1, S2, S3) instead of using the full available dataset (April 27 - June 7):

- **S1 Split**: May 18-24 (train) + May 25-31 (validation) � **Hyperparameter optimization only**
- **S2 Split**: May 25-31 (7 days) � **Final training with recent trends**  
- **S3 Split**: May 18-31 (14 days) � **Final training with more data volume**

**Core Hypothesis**: Separating hyperparameter tuning from final training maintains parameter stability while adapting to recent trends.

### Implementation in Codebase

#### KAMI Pipeline Implementation
**Location**: `kami/preprocess/dataset067/run.py`
```yaml
# Dataset067 uses S3 strategy (May 18-31)
date_range: "May 18-31, 2023"  # 14-day training period
split_strategy: "S3"           # Balances recency with data volume
```

#### KFujikawa Pipeline Implementation  
**Location**: `kfujikawa/src/exp/v1xxx_training/`
```python
# Fold configuration for temporal splits
fold_mapping = {
    "fold_0": "S1",  # May 18-24 train, May 25-31 validation (hyperparameters)
    "fold_2": "S3"   # May 18-31 train (final models)
}

# Training process
python model.py train common.fold=0  # S1 for hyperparameter tuning
python model.py train common.fold=2  # S3 for final training
```

#### Sugawarya Pipeline Implementation
**Location**: `sugawarya/src/utils.py`
```python
# Ensemble uses predictions from S1 (validation) and S3 (test)
def read_preds(pred_dir_path, data_period, pred_type):
    pred_type_mapping = {
        "first": "S1_validation",    # For weight optimization
        "third": "S3_test"          # For final predictions
    }
```

### Strategic Benefits Observed in Code
1. **Temporal Consistency**: All components use same time-based splits
2. **Hyperparameter Stability**: S1 hyperparameters applied to S2/S3 training
3. **Recency Adaptation**: S2/S3 capture trends closer to test period (June 1-7)
4. **Ensemble Optimization**: Validation on S1, final predictions on S3

This splitting strategy appears throughout all three pipeline components, demonstrating its fundamental importance to the winning approach.








## 2. Feature Engineering Strategy

### Research Paper Philosophy
The authors identified that **direct use of article IDs or raw textual content leads to overfitting** in news recommendation due to:
- **Rapid content decay**: Articles become less relevant quickly
- **Temporal memorization**: Models memorize specific words/phrases tied to time periods
- **Pattern brittleness**: Overfitting to short-lived trends

**Core Objectives**:
- Prevent sensitivity to article age
- Avoid overfitting to ephemeral patterns
- Create temporally robust features

### Anti-Overfitting Strategies in Code

#### 1. Time-Based Features
**KAMI Implementation**: `c_article_publish_time_v5`
```python
# Location: kami/features/c_article_publish_time_v5/run.py
time_features = [
    "c_time_sec_diff",    # Seconds since publication
    "c_time_min_diff",    # Minutes since publication  
    "c_time_hour_diff",   # Hours since publication
    "c_time_day_diff"     # Days since publication
]
# Creates temporal context instead of memorizing specific articles
```

#### 2. User Interaction Temporal Features
**KAMI Implementation**: `u_stat_history` + `c_is_viewed`
```python
# Recent viewing patterns with sliding windows
"c_is_viewed_1/2/5/10/20"  # Boolean flags for viewing in last N impressions
"u_history_len"            # Total historical interactions
"u_scroll_percentage_*"    # Behavioral patterns over time
```

#### 3. Similarity Features (Not Raw Content)
**KAMI Implementation**: Multiple `c_*_tfidf_svd_sim` modules
```python
# Vector-based similarity instead of raw text
similarity_pipeline = [
    "TF-IDF vectorization",      # Convert text to vectors
    "SVD dimensionality reduction", # Reduce to 50 components  
    "L2 normalization",          # Normalize embeddings
    "Cosine similarity"          # Compute user-article similarity
]
# Prevents memorization of specific words/phrases
```

### Contrasting Approaches: KAMI vs KFujikawa

#### KAMI: Extensive Anti-Overfitting Engineering
```python
# 29 feature extraction modules creating 100+ features
feature_types = {
    "temporal": ["c_article_publish_time_v5", "y_transition_prob_from_first"],
    "similarity": ["c_title_tfidf_svd_sim", "c_body_tfidf_svd_sim"],
    "behavioral": ["u_stat_history", "c_appear_imp_count_v7"],
    "statistical": ["a_click_ranking", "i_article_stat_v2"]
}
```

#### KFujikawa: Neural Network Feature Learning
```python
# Minimal preprocessing, let transformers learn representations
preprocessing = [
    "basic_temporal_encoding",    # Simple timestamps, weekdays
    "categorical_id_mapping",     # Frequency-based ID encoding
    "sequence_preservation",      # Raw sequences for attention
    "embedding_learning"          # Let neural networks discover patterns
]
```

### Strategic Data Leakage (Controlled Future Information)

**IMPORTANT**: The authors deliberately incorporated **3 types of future information** in their **main competition submission pipeline** (not just experimental). This was a core strategy that contributed to their 1st place performance:

KAMI: Uses them directly as a_total_inviews,
a_total_pageviews, a_total_read_time

KFujikawa: Also uses them but with
log-transformation and sinusoidal embeddings
for neural networks (processed in
v0100_articles.py)


#### 1. Future Article Statistics
**Implementation**: Base article features (`a_base`)
```python
# Articles.parquet contains statistics from entire dataset period
"a_total_inviews"     # Includes test period engagement
"a_total_pageviews"   # Long-term popularity indicators  
"a_total_read_time"   # Reflects general article appeal
```

#### 2. Future Impressions (5-minute & 1-hour windows)
**KAMI Implementation**: `c_appear_imp_count_v7` + 67 future features
```python
# Configuration: future_impression_cols: true (in competition models)
"c_common_count_future_5m"    # Future 5 minutes
"c_common_count_future_1h"    # Future 1 hour  
"c_user_count_future_5m"      # User-specific future 5 minutes
"c_user_count_future_1h"      # User-specific future 1 hour
"c_user_count_future_all"     # All future impressions
```

**KFujikawa Implementation**: "111" configuration enables all future leakage
```python
# All production models use "111" = future_imp + future_stats + past_imp
models_with_leakage = [
    "v1157_111_fix_past_v2",      # 111 = all leakage enabled
    "v1170_111_L8_128d",          # 111 = all leakage enabled
    "v1174_111_L8_128d_smpl3_drophist",  # 111 = all leakage enabled
    "v1184_111_PL_bert_L4_256d"   # 111 = all leakage enabled
]
```

#### 3. Pre-Prediction Impressions (Valid temporal features)
**KAMI Implementation**: Multiple temporal aggregation features
```python
# Only uses impressions before prediction time
"c_appear_imp_count_past_all"     # All historical impressions
"c_read_time_accumulation_past"   # Historical read time
"u_history_len"                   # User's historical activity length
```

### Validation Through Ablation Studies

The team validated their leakage strategy through systematic experiments:

**KFujikawa Ablation Studies**:
```python
"v2001_000_v1174_noleak.py"     # No future leakage (experimental)
"v2002_001_v1174_use_past.py"   # Past impressions only (experimental)  
"v1174_111_L8_128d_*"           # All leakage enabled (PRODUCTION)
```

**Final Ensemble Uses Leakage-Enabled Models**:
```python
# Sugawarya combines predictions from all leak-enabled models
final_ensemble = weighted_mean(leak_models) + stacking(leak_models)
```

### Feature Engineering Impact Analysis

**KAMI Strategy**: Creates robust features + strategic future leakage (67 future features)
**KFujikawa Strategy**: Neural learning + controlled future information ("111" configuration)  
**Sugawarya Strategy**: Ensembles leak-enabled predictions through advanced meta-learning

**Key Finding**: The strategic use of 5-minute and 1-hour future lookahead windows was central to achieving 1st place, not just an experimental technique.










## 3. Transformer Architecture

### Research Paper Innovation
Unlike previous news recommendation transformers that predict single candidates independently, the authors adopted a **RankFormer-style approach** where the model estimates click probabilities for **multiple articles simultaneously within the same impression**. This enables **intra-impression article interactions** through self-attention.

### Dual Embedding Architecture

#### Impression Embeddings (Session Context)
**Purpose**: Summarize overall session/impression context
**Components**:
- Temporal context (time of day, session timing)
- Device information (mobile, desktop)
- Geographic context (region, location)
- User session patterns (time since last session)

**KFujikawa Implementation**: `i_base_feat` module
```python
# Location: kfujikawa/src/ebrec/models/newsrec/dataloader.py
impression_features = [
    "i_device_type",           # Device context
    "i_is_sso_user",          # User account type
    "i_read_time",            # Session engagement
    "i_scroll_percentage",    # Session interaction depth
    "i_gender", "i_age"       # User demographics
]
```

#### Inview Embeddings (Article-Specific)
**Purpose**: Individual article representations within the impression
**Architecture**: Three embedding types per article

##### 1. Quantitative Features → Sinusoidal Embeddings
**Research Approach**: Continuous values encoded using transformer-style positional encoding
```python
# KFujikawa Implementation: Numerical feature processing
quantitative_features = [
    "article_age",                    # Time since publication
    "time_since_publication",         # Temporal context
    "user_article_interaction_counts", # Behavioral metrics
    "total_inviews",                 # Article popularity
    "scroll_percentage_sum"          # Engagement metrics
]

# Sinusoidal encoding for neural network input
def sinusoidal_embedding(values, d_model=128):
    # Similar to transformer positional encoding
    return torch.sin/cos(values / temperature)
```

##### 2. Categorical Features → Learned Embeddings
**Research Approach**: Traditional embedding layers for discrete categories
```python
# KFujikawa Implementation: v0102_article_metadata_id_v2.py
categorical_mappings = [
    "ner_clusters_id",     # Named entity clusters
    "entity_groups_id",    # Entity group categories
    "topics_id",           # Article topics  
    "category_id",         # Main category
    "subcategory_id",      # Sub-category
    "image_ids_id"         # Visual content IDs
]

# Frequency-based ID assignment for efficient embeddings
def create_embeddings(vocab_size, embedding_dim):
    return nn.Embedding(vocab_size, embedding_dim)
```

##### 3. Similarity Features → User-Article Affinity
**Research Approach**: Cosine similarity between article topics and user's click history
```python
# KFujikawa Implementation: User-article topic similarity
def compute_topic_similarity(article_topics, user_history_topics):
    # Calculate cosine similarity between article and user's recent topics
    article_embedding = topic_encoder(article_topics)
    user_profile = mean(topic_encoder(user_history_topics))
    return cosine_similarity(article_embedding, user_profile)
```

### Transformer Processing Pipeline

#### Sequence Construction
```python
# Variable-length sequences: one vector per article in impression
impression_sequence = [
    impression_embedding,     # Session context (shared)
    inview_embedding_1,      # Article 1 representation
    inview_embedding_2,      # Article 2 representation  
    # ... up to 20+ articles per impression
    inview_embedding_N       # Article N representation
]
```

#### Self-Attention Interactions
**KFujikawa Implementation**: FastFormer architecture
```python
# Location: kfujikawa/src/ebrec/models/fastformer/fastformer.py
class FastFormer(nn.Module):
    def __init__(self, layers=8, d_model=128):
        self.attention_layers = nn.ModuleList([
            FastFormerBlock(d_model) for _ in range(layers)
        ])
    
    def forward(self, impression_sequence):
        # Self-attention across all articles in impression
        for layer in self.attention_layers:
            impression_sequence = layer(impression_sequence)
        return impression_sequence  # Updated representations
```

**Key Innovation**: Articles can "see" and influence each other's representations within the same impression, enabling competitive ranking dynamics.

### Dynamic Sampling Strategy

#### Research Problem
- **Scale**: 12 million impressions across May 18-24 period
- **Diversity**: 780K users with varying behavior patterns
- **Computational constraints**: Cannot process all data efficiently

#### Solution: Epoch-Based Dynamic Sampling
**Location**: `kfujikawa/src/exp/v1xxx_training/v1174_111_L8_128d_smpl3_drophist.py`

```python
# Configuration (lines 154-159)
batch_sampler=DataLoaderConfig.BatchSamplerConfig(
    dataset={"_var_": "train_dataset"},
    max_sample_per_user=3,        # KEY: 3 impressions per user
    shuffle=True,                 # Random selection
    drop_last=True,
)

```
**Validation**: `v2006_111_v1174_no-sample.py` sets `max_sample_per_user=10**8` (unlimited) for comparison


**Model Variants in Production**:
```python
# All use same transformer architecture with different optimizations
transformer_models = {
    "v1170_111_L8_128d": "Base FastFormer (8 layers, 128 dim)",
    "v1174_111_L8_128d_smpl3_drophist": "Enhanced sampling + dropout history",  
    "v1184_111_PL_bert_L4_256d": "BERT variant + pseudo-labeling"
}
```

## 4. Gradient Boosting Decision Trees (GBDT)

### Research Paper Strategy
The authors employed **two complementary GBDT algorithms** to leverage different strengths:
- **LightGBM**: Excels at large-scale datasets through parallel learning
- **CatBoost**: Minimizes information loss in categorical data handling

Both models frame recommendation as a **ranking task** using LambdaRank pairwise objective to optimize article ranking within impressions.

### KAMI Pipeline Implementation

#### Model Configuration
**Location**: `kami/experiments/015_train_third/` (LightGBM) and `kami/experiments/016_catboost/` (CatBoost)







##### LightGBM Model (`015_train_third`)
```python
# Configuration: kami/experiments/015_train_third/exp/base.yaml
lgbm_params = {
    "objective": "lambdarank",           # LambdaRank for ranking
    "metric": "ndcg",                   # NDCG@10 optimization
    "ndcg_eval_at": [10],              # Ranking depth
    "num_boost_round": 400,            # 400 boosting rounds
    "learning_rate": 0.1,              # Fixed learning rate
    "early_stopping_rounds": 40,       # Early stopping patience
}
```

##### CatBoost Model (`016_catboost`)
```python
# Configuration: kami/experiments/016_catboost/exp/base.yaml
catboost_params = {
    "objective": "YetiRank",           # CatBoost ranking objective
    "eval_metric": "NDCG:top=10",      # NDCG@10 optimization
    "iterations": 1000,                # 1000 iterations
    "learning_rate": 0.1,              # Fixed learning rate
    "od_wait": 40,                     # Early stopping (40 rounds)
}
```

#### Data Sampling & Feature Integration
**Research**: "randomly sample 20% of user impression data"

**KAMI Implementation**: Uses all 100+ engineered features
```python
# Features fed to GBDT models include:
feature_categories = [
    "article_features",      # a_base, a_click_ranking, etc.
    "candidate_features",    # c_*_tfidf_sim, c_appear_imp_count_v7
    "impression_features",   # i_base_feat, i_article_stat_v2
    "user_features",        # u_stat_history, u_click_article_stat_v2
    "leak_features"         # 67 future impression features
]
```

#### Optimization Strategy
**Key Innovation**: NDCG@10 instead of AUC for faster training
```python
# Both models optimize NDCG@10 (strongly correlated with AUC)
optimization = {
    "training_metric": "ndcg@10",        # Fast optimization
    "validation_metric": "auc",          # Final performance
    "early_stopping": 40,               # Rounds without improvement
    "hyperparameter_tuning": "S1 split", # May 18-24 + May 25-31
    "final_training": "S3 split"        # May 18-31 combined
}
```









### Ensemble Integration
**Final Weights in Sugawarya Pipeline**:
```python
kami_models = [
    "015_train_third": "LightGBM",      # Higher weight (~2)
    "016_catboost": "CatBoost"          # Lower weight (~1)
]
# Combined with 3 neural network models for 5-model ensemble
```

## 5. Ensemble Learning

### Research Paper Strategy
The authors designed a **3-stage ensemble approach** to integrate predictions from multiple models, combining both linear and non-linear ensemble techniques.

### Stage 1: Base Model Predictions
**Input**: Predictions from 5 trained models
- **KAMI**: 2 GBDT models (LightGBM + CatBoost)
- **KFujikawa**: 3 neural models (FastFormer variants + BERT)

### Stage 2: Dual Ensemble Methods

#### Method 1: Optimized Weight Averaging
**Research Approach**: Uses Optuna to determine optimal weights maximizing AUC on validation set

**Sugawarya Implementation**: `weighted_mean.py`
```python
# Bayesian optimization with Optuna (200 trials)
def optimize_weight_by_optuna(explode_pred_df, pred_cols):
    def objective(trial):
        weights = {model: trial.suggest_float(model, 0, 1) for model in pred_cols}
        ensemble_pred = sum(pred[model] * weights[model] for model in pred_cols)
        return get_impression_auc_mean(ensemble_pred, targets)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    return study.best_params

# Models combined:
base_models = [
    "kami/015_train_third",              # LightGBM
    "kami/016_catboost",                # CatBoost  
    "kfujikawa/v1157_fix_past_v2",      # FastFormer variant 1
    "kfujikawa/v1170_L8_128d",          # FastFormer variant 2
    "kfujikawa/v1184_PL_bert_L4_256d"   # BERT + pseudo-labeling
]
```

#### Method 2: Stacking with Feature Engineering
**Research Approach**: Treats stage-1 predictions as new features for LightGBM meta-learner

**Feature Engineering on Predictions**:
```python
# Sugawarya Implementation: stacking.py
def feature_engineering(explode_pred_df, raw_pred_cols):
    # 1. Raw prediction scores from each model
    raw_features = raw_pred_cols  # Direct model outputs
    
    # 2. Statistical features within impressions
    impression_stats = [
        f"{model}_mean/max/min/std"  # Statistics per impression
        for model in raw_pred_cols
    ]
    
    # 3. Relative ranking features
    ranking_features = [
        f"{model}_rank/rank_desc",           # Rankings within impression
        f"{model}_normedrank/normedrank_desc" # Normalized rankings [0,1]
        for model in raw_pred_cols
    ]
    
    # 4. Normalization features
    normalization = [
        f"{model}_zscore",                   # Z-score normalization
        f"{model}_normed_in_impression"      # Min-max normalization
        for model in raw_pred_cols
    ]
    
    # 5. Interaction features between model pairs
    for model1, model2 in itertools.combinations(raw_pred_cols, 2):
        interactions = [
            f"{model1}_{model2}_diff/ratio/max/min"  # Pairwise interactions
        ]
    
    # Total: ~350+ engineered meta-features from 5 base predictions
    return enhanced_features
```

**Meta-Learner Configuration**:
```python
# LightGBM stacking model
lgb_params = {
    "objective": "lambdarank",        # Learning-to-rank
    "metric": "ndcg",                # NDCG optimization
    "ndcg_at": [5, 10, 20],         # Multiple ranking depths
    "learning_rate": 0.1,
    "feature_fraction": 0.8,         # Feature sampling
    "bagging_fraction": 0.8,         # Row sampling
}

# Cross-validation with GroupKFold (4 folds)
gkf = GroupKFold(n_splits=4)  # Prevents impression leakage
```

### Stage 3: Final Prediction Fusion
**Research Approach**: Average the optimized weighted averaging and stacking scores

**Sugawarya Implementation**: `make_submission.py`
```python
# Simple additive combination
def create_final_ensemble():
    weighted_predictions = load_weighted_mean_results()  # Stage 2, Method 1
    stacking_predictions = load_stacking_results()      # Stage 2, Method 2
    
    # Final fusion
    final_prediction = weighted_predictions + stacking_predictions
    
    # Convert to competition ranking format
    ranked_predictions = convert_to_rankings(final_prediction)
    return ranked_predictions
```



