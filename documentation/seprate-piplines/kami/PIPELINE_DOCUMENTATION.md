# KAMI Pipeline Documentation

This document explains the complete machine learning pipeline for the news recommendation system, covering data preparation, feature extraction, training, and validation.

## Pipeline Overview

The pipeline consists of 4 main stages executed in order through `tasks.py`:

1. **Data Preparation** (`create_candidates`)
2. **Feature Extraction** (`create_features`) 
3. **Dataset Creation** (`create_datasets`)
4. **Training & Validation** (`train`)

## 1. Data Preparation

### 1.1 Test Demo Setup
**File**: `preprocess/test_demo/run.py`
**Purpose**: Creates a small subset of test data for fast development/debugging
- Samples random users from full test dataset
- Reduces data size for quick pipeline validation
- Used during development phase, not for final training

### 1.2 Candidate Generation  
**File**: `preprocess/make_candidate/run.py`
**Purpose**: Converts user impression logs into training examples

**The Transformation**:
```
Raw impression: User123 saw [Article A, Article B, Article C], clicked [Article A]
↓
Candidate pairs:
- User123 + Article A → Label=1 (clicked)
- User123 + Article B → Label=0 (not clicked)  
- User123 + Article C → Label=0 (not clicked)
```

**Output**: Creates train/validation datasets where each row is a (user, article, clicked?) triplet ready for machine learning. This transforms the recommendation problem into binary classification: "Will this user click this article?"



## 2. Feature Extraction

The system extracts 5 types of features, each capturing different aspects of the recommendation problem:

### 2.1 Article Features (a_*)
Static properties of articles themselves:

- **`a_base`**: Core article metadata (premium status, category, engagement metrics, sentiment)
- **`a_click_ranking`**: Article click ranking features
- **`a_additional_feature`**: Additional article-level features
- **`a_click_ratio`**: Article click-through ratios
- **`a_click_ratio_multi`**: Multi-faceted click ratio features

### 2.2 Candidate Features (c_*)
User-article interaction features computed for each recommendation candidate:

- **`c_appear_imp_count_v7`**: Article appearance counts in impressions
- **`c_appear_imp_count_read_time_per_inview_v7`**: Read time per impression appearance
- **`c_topics_sim_count_svd`**: Topic similarity using count vectors + SVD
- **`c_title_tfidf_svd_sim`**: Title similarity using TF-IDF + SVD
- **`c_subtitle_tfidf_svd_sim`**: Subtitle similarity features
- **`c_body_tfidf_svd_sim`**: Article body text similarity
- **`c_category_tfidf_sim`**: Category-based similarity
- **`c_subcategory_tfidf_sim`**: Subcategory similarity
- **`c_entity_groups_tfidf_sim`**: Named entity group similarity
- **`c_ner_clusters_tfidf_sim`**: NER cluster-based similarity
- **`c_article_publish_time_v5`**: Time-based features relative to article publication
- **`c_is_already_clicked`**: Whether user previously clicked the article

### 2.3 Impression Features (i_*)
Contextual features about the browsing session:

- **`i_base_feat`**: Session context (device, user demographics, engagement metrics)
- **`i_stat_feat`**: Statistical features of the impression
- **`i_viewtime_diff`**: Time difference features in viewing
- **`i_article_stat_v2`**: Article statistics within the impression

### 2.4 User Features (u_*)
Behavioral profiles aggregated from user history:

- **`u_stat_history`**: Statistical summaries of user behavior (scroll patterns, reading time)
- **`u_click_article_stat_v2`**: Statistics about user's clicked articles

### 2.5 User-Article Features (ua_*)
Combined user-article affinity features:

- **`ua_topics_sim_count_svd_feat`**: User and article embeddings in topic space using count vectors + SVD

### 2.6 Target Features (y_*)
Features related to prediction outcomes:

- **`y_transition_prob_from_first`**: Transition probability features from first interaction

## 3. Dataset Creation

### 3.1 Dataset Assembly
**File**: `preprocess/dataset067/run.py`
**What happens**: Takes all the separate feature tables and glues them together

**The Process**:
```
1. Start with candidate pairs: (user_id, article_id, clicked?)
2. Add article features: Join with a_* tables 
3. Add candidate features: Join with c_* tables
4. Add impression features: Join with i_* tables  
5. Add user features: Join with u_* tables
6. Add user-article features: Join with ua_* tables
7. Add target features: Join with y_* tables
8. Remove unwanted columns
```

**Result**: One big table where each row has:
- Basic info: user_id, article_id, impression_id, label (clicked?)
- 100+ features: All the a_*, c_*, i_*, u_*, ua_*, y_* columns
- Ready for machine learning models

**Dataset067**: Uses May 18-31 date range (S3 split strategy)
**Output**: `train_feat.parquet` and `validation_feat.parquet` - complete datasets ready for training

**Data Splitting Strategy (Dataset067)**:
The team implemented a 2-step training approach to handle temporal data decay:

**Step 1 - Hyperparameter Optimization**:
- **S1 Split**: May 18-24 (train) + May 25-31 (validation) 
- Used to find optimal model hyperparameters
- Standard train/validation split for model selection

**Step 2 - Final Model Training**:
After finding best hyperparameters from S1, retrain final models using:
- **S2 Strategy**: Train only on May 25-31 (7 days) - captures most recent trends
- **S3 Strategy**: Train on full May 18-31 (14 days) - balances recency with data volume

The logic: S1 optimizes hyperparameters, then S2/S3 produce final models using those hyperparameters but with data closer to the test period (June 1-7).5

## 4. Training & Validation

The training process follows the research paper's 2-step approach with temporal data splitting:

### Training Timeline

**Phase 1: Hyperparameter Optimization (S1 Split)**8
1. Use dataset with S1 split: May 18-24 (train) + May 25-31 (validation)
2. Run multiple neural network architectures to find best hyperparameters:
   - NAML, NRMS, NPA, NRMS-DocVec, LSTUR, FastFormer models
3. Select optimal model configurations based on validation performance

**Phase 2: Final Model Training (S2/S3 Splits)**
4. **Neural Networks** (`experiments/015_train_third/`):
   - Train with config `large067_001` using best hyperparameters from Phase 1
   - Dataset067 = S3 strategy (May 18-31 training data)
   - Produces transformer-based recommendation models

5. **Gradient Boosting** (`experiments/016_catboost/`):
   - Train CatBoost models with config `large067`  
   - Uses same S3 temporal split (May 18-31)
   - Processes tabular features extracted in previous steps

**Phase 3: Ensemble**
6. Combine predictions from neural networks + CatBoost
7. Final predictions blend both model types for submission

### Key Models Used
- **Neural**: FastFormer (transformer), NRMS, NAML, NPA, LSTUR
- **Gradient Boosting**: CatBoost on extracted features
- **Strategy**: Hyperparameter tuning on S1 → Final training on S3 → Ensemble

## Configuration System

Each component uses Hydra configuration with:
- **Base config**: `exp/base.yaml` - default parameters
- **Size-specific configs**: `exp/demo.yaml`, `exp/small.yaml`, `exp/large.yaml`
- **Experiment-specific configs**: Custom configurations for specific runs

## Execution

### Full Pipeline
```bash
# Run complete pipeline
invoke run-all

# Debug mode (uses small dataset)
invoke run-all --debug
```

### Individual Stages
```bash
# Individual stages
invoke create-candidates [--debug]
invoke create-features [--debug] 
invoke create-datasets [--debug]
invoke train [--debug]
```

## Key Design Principles

1. **Temporal Awareness**: Data splitting respects temporal order to prevent data leakage
2. **Modular Features**: Features are organized by what they represent (content, context, behavior)
3. **Scalable Architecture**: Supports different data sizes (demo/small/large)
4. **Ensemble Approach**: Combines neural networks and gradient boosting
5. **Hyperparameter Separation**: Optimizes hyperparameters on one split, trains final model on another

## Output Structure

```
output/
├── preprocess/
│   ├── make_candidate/{size_name}/
│   │   ├── train_candidate.parquet
│   │   └── validation_candidate.parquet
│   └── dataset067/{size_name}/
│       ├── train_feat.parquet  
│       └── validation_feat.parquet
├── features/{feature_name}/{size_name}/
│   ├── train_feat.parquet
│   └── validation_feat.parquet
└── experiments/{experiment_name}/
    └── trained_models/
```

This pipeline implements the research paper's approach of using temporal data splitting and ensemble learning with transformers for news recommendation.