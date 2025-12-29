# KFujikawa Feature Extraction Documentation

Comprehensive guide to KFujikawa's minimal preprocessing approach for neural network-based news recommendation.

## Philosophy: Neural Network-Driven Feature Learning

**KFujikawa follows a "minimal preprocessing, maximum learning" philosophy:**
- Provide clean, indexed data to neural networks
- Let transformers learn complex representations automatically  
- Focus on data structure and temporal aggregation
- Avoid hand-crafted feature engineering

**Contrast with KAMI**: While KAMI creates 100+ hand-crafted features for gradient boosting, KFujikawa creates ~10-15 basic features for neural networks.

---

## Preprocessing Pipeline (`v0xxx_preprocess/`)

### 1. Core Data Processing

#### `v0100_articles.py` - Article Data Preparation
**Purpose**: Basic article data cleaning and temporal feature extraction
**Input**: `articles.parquet`
**Output**: Clean article data with basic temporal features
**Features Created**:
- `published_ts` - Unix timestamp of publication
- `published_date` - Publication date  
- `published_weekday` - Day of week (1-7)
- Type conversions for engagement metrics (`total_inviews`, `total_pageviews`, `total_read_time`)
- Null handling and data type optimization

**Methodology**:
```python
lf_articles = lf_articles.with_columns(
    (pl.col("published_time").dt.timestamp() // 10**6).cast(pl.Int32).alias("published_ts"),
    pl.col("published_time").cast(pl.Date).alias("published_date"), 
    pl.col("published_time").dt.weekday().alias("published_weekday"),
    pl.col("total_inviews").fill_null(0).cast(pl.Int32),
    # ... basic cleaning operations
)
```

#### `v0200_users.py` - User Data Processing
**Purpose**: User profile data preparation
**Input**: User demographic and behavioral data
**Output**: Clean user features for neural network input
**Features**: Basic user attributes and behavioral metrics

#### `v0300_impressions.py` - Impression Data Processing  
**Purpose**: Session and impression data preparation
**Input**: Behavioral impression logs
**Output**: Structured impression sequences for neural models
**Features**: Session-level aggregations and temporal patterns

### 2. Enhanced Feature Engineering

#### `v0101_article_inviews_in_split.py` & `v0101_article_inviews_in_split_v2.py` - Temporal Aggregation
**Purpose**: Article engagement patterns within data splits
**Input**: Impression behaviors with temporal splits
**Features Created**:
- `inview_counts` - Frequency of article appearances
- `read_time_sum` - Total accumulated read time
- `scroll_percentage_sum` - Total scroll engagement
- `scroll_zero_counts` - Count of zero-scroll impressions
- Time-windowed aggregations by `inview_elapsed_mins`

**Methodology**:
```python
# Temporal aggregation by article and time window
lf_impressions.group_by(["article_index", "inview_elapsed_mins"]).agg(
    pl.len().alias("inview_counts"),
    pl.sum("read_time").alias("read_time_sum"),
    pl.sum("scroll_percentage").alias("scroll_percentage_sum"),
    (pl.col("scroll_percentage") == 0).sum().alias("scroll_zero_counts"),
)
```

#### `v0102_article_metadata_id_v2.py` - Categorical ID Encoding
**Purpose**: Convert categorical fields to numeric IDs for neural network embedding
**Input**: Article metadata (categories, entities, topics)
**Output**: Numeric ID mappings for embedding layers
**Features**:
- `ner_clusters_id` - Named entity cluster IDs
- `entity_groups_id` - Entity group IDs  
- `topics_id` - Topic IDs
- `category_id/subcategory_id` - Category hierarchy IDs
- `image_ids_id` - Image identifier IDs

**Methodology**:
```python
def _compute_metadata_ids(lf, input_col, output_col, dtype, min_inview_count=10000):
    # Frequency-based categorical encoding
    # High-frequency categories get lower IDs (better for embeddings)
    # Rare categories aggregated into common "other" category
```

#### `v0103_article_history_counts.py` - Historical Context Features
**Purpose**: Article popularity in recent time windows
**Input**: User history and article interactions
**Features**:
- `history_last_24h_counts` - Interactions in last 24 hours
- `history_last_1h_counts` - Interactions in last hour
- `history_counts_in_split` - Total interactions within data split

**Methodology**:
```python
# Sliding window aggregation
history_counts = lf_history.group_by("article_index").agg(
    (pl.col("impression_ts") >= split_start_ts - 24*60*60).sum().alias("history_last_24h_counts"),
    (pl.col("impression_ts") >= split_start_ts - 1*60*60).sum().alias("history_last_1h_counts"),
)
```

#### `v0201_user_inviews_in_split.py` - User Engagement Patterns
**Purpose**: User-level engagement statistics within splits
**Features**: User interaction frequency and engagement patterns
**Methodology**: Temporal aggregation at user level

#### `v0301_imp_counts_per_user.py` - User Session Statistics
**Purpose**: Impression-level user behavioral patterns
**Features**: Session frequency and interaction statistics
**Methodology**: User-impression aggregations

#### `v0302_imp_target_in_split.py` - Target Engineering
**Purpose**: Click prediction targets aligned with data splits
**Features**: Binary click targets with temporal alignment
**Methodology**: Target extraction respecting temporal splits

---

## Neural Network Data Preparation

### Embedding-Ready Features
Unlike KAMI's similarity computations, KFujikawa prepares data for learned embeddings:

```python
# From model dataloaders - Neural input preparation
def prepare_neural_input(self, x):
    # User history sequences
    history_input = repeat_by_list_values_from_matrix(
        input_array=x[self.history_column].to_list(),
        matrix=self.lookup_matrix,  # (num_articles, embedding_dim)
        repeats=repeats,
    )
    
    # Candidate articles  
    candidate_input = self.lookup_matrix[x[self.inview_col].explode().to_list()]
    
    # Let transformer learn representations
    return history_input, candidate_input
```

### Text Processing Philosophy
- **Raw text preservation**: Keep original titles, bodies for neural processing
- **Tokenization**: Convert text to token sequences for transformer input
- **Embedding learning**: Let neural networks learn optimal text representations
- **No TF-IDF/SVD**: Avoid manual feature engineering from text

---

## Feature Categories

### 1. **Temporal Features**
- **Basic time encodings**: Timestamps, weekdays, time differences
- **Window aggregations**: 1-hour, 24-hour interaction counts
- **Split-aware features**: Metrics computed within train/validation splits

### 2. **Categorical Encoding**
- **Frequency-based IDs**: Popular categories get lower IDs for embedding efficiency
- **Hierarchical encoding**: Category/subcategory relationships preserved
- **Cold-start handling**: Rare categories mapped to common "other" buckets

### 3. **Behavioral Aggregations**  
- **Simple counts**: Interaction frequencies, view counts
- **Engagement sums**: Total read time, scroll percentages
- **Session patterns**: User-level and article-level engagement statistics

### 4. **Sequential Data**
- **User histories**: Preserved as sequences for transformer attention
- **Impression lists**: Article sequences within sessions
- **Temporal ordering**: Chronological sequence preservation

---

## Comparison with KAMI

| Feature Type | KFujikawa Approach | KAMI Approach |
|-------------|-------------------|---------------|
| **Text Processing** | Raw text → Tokenization → Neural embeddings | Text → TF-IDF → SVD → Similarity features |
| **User Modeling** | History sequences → Transformer attention | Statistical aggregations + similarity scores |
| **Temporal Features** | Basic time windows (1h, 24h) | Complex temporal patterns + transition probabilities |
| **Categorical Data** | Frequency-based ID encoding → Embeddings | Ordinal encoding + click ratio statistics |
| **Interaction Features** | Simple counts and sums | Complex similarity matrices + rankings |
| **Feature Engineering** | ~15 basic features + raw data | 100+ hand-crafted features |

---

## Technical Implementation

### Data Processing Stack
- **Polars**: High-performance data manipulation
- **Temporal Logic**: Sliding windows and split-aware processing
- **Memory Efficiency**: Chunked processing for large datasets
- **Type Optimization**: Efficient data types for neural network input

### Neural Network Integration
- **PyTorch Lightning**: Training infrastructure
- **Transformer Architecture**: FastFormer, BERT-based models
- **Embedding Layers**: Learned representations for categorical features
- **Attention Mechanisms**: Self-attention for sequence modeling

---

## Output Structure

```
kfujikawa/data/
├── v0xxx_preprocess/
│   ├── articles/           # Processed article data
│   ├── users/             # User profile data  
│   ├── impressions/       # Session/impression data
│   └── enhanced_features/ # Additional temporal/behavioral features
```

**Key Files**:
- Article indices and embeddings
- User sequence data for transformer input
- Temporal aggregation features
- Categorical ID mappings for embedding layers

---

## Design Philosophy

**Minimal Feature Engineering Rationale**:
1. **Neural Network Strength**: Transformers excel at learning complex patterns from raw data
2. **Scalability**: Less manual feature tuning required for new domains/data
3. **Representation Learning**: Attention mechanisms discover optimal feature combinations
4. **End-to-End Learning**: Gradients flow through entire pipeline for optimal representations

**Trade-offs**:
- **Pros**: Lower engineering overhead, automatic feature discovery, better generalization
- **Cons**: Less interpretable, requires larger datasets, more compute for training

This approach demonstrates that state-of-the-art performance can be achieved with minimal feature engineering when leveraging modern neural architectures effectively.