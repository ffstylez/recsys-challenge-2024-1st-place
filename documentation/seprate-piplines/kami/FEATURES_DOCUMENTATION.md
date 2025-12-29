# KAMI Pipeline Feature Documentation

Comprehensive guide to all 29 feature extraction modules in the KAMI news recommendation pipeline.

## Overview

The KAMI pipeline extracts 6 categories of features, each identified by a prefix that indicates the feature scope:

- **`a_*`** - Article Features: Static article-level properties
- **`c_*`** - Candidate Features: User-article interaction and similarity features  
- **`i_*`** - Impression Features: Session/impression-level aggregated features
- **`u_*`** - User Features: User behavior and profile features
- **`ua_*`** - User-Article Features: User-article embedding features
- **`y_*`** - Target Features: Sequential patterns and transition features

---

## 1. Article Features (`a_*`)

### `a_base` - Core Article Metadata
**What it does**: Extracts fundamental article properties and engagement metrics
**Input**: `articles.parquet`
**Features**:
- `a_premium` - Boolean flag for premium content
- `a_category_article_type` - Ordinal-encoded article type (News, Sports, etc.)
- `a_total_inviews/pageviews/read_time` - Historical engagement statistics
- `a_sentiment_score` - Article sentiment score [-1, 1]
- `a_ordinal_sentiment_label` - Encoded sentiment: Negative(0), Neutral(1), Positive(2)
**Calculation**: Direct extraction + ordinal encoding for categorical variables

### `a_additional_feature` - Article Engagement Ratios
**What it does**: Computes derived engagement efficiency metrics
**Input**: `articles.parquet`  
**Features**:
- `a_inviews_per_pageviews` - View-to-impression conversion rate
- `a_read_time_per_pageviews/inviews` - Reading engagement intensity
- `a_subcategory_len/image_ids_len` - Content complexity measures
**Calculation**: Mathematical ratios (e.g., read_time / pageviews) and list length counting

### `a_click_ranking` - Article Popularity Ranking
**What it does**: Ranks articles by historical click frequency
**Input**: `history.parquet`
**Features**:
- `a_click_rank` - Global ranking by total clicks (1 = most clicked)
- `a_click_count` - Total historical click count across all users
**Calculation**: Global click aggregation → ranking assignment

### `a_click_ratio` - Category Click Distribution
**What it does**: Analyzes click distribution across article categories
**Input**: `history.parquet` + `articles.parquet`
**Features**: `a_category_click_ratio` - Proportion of total clicks in each category
**Calculation**: (Category clicks / Total clicks) per category

### `a_click_ratio_multi` - Multi-Category Click Statistics
**What it does**: Statistical analysis for complex multi-valued categories
**Input**: `history.parquet` + `articles.parquet`
**Features**: Click ratio statistics (mean/std/max/min) for `subcategory` and `entity_groups`
**Calculation**: Explode multi-valued fields → calculate click ratios → aggregate statistics per article

---

## 2. Candidate Features (`c_*`)

### `c_appear_imp_count_v7` - Impression Appearance Frequency
**What it does**: Tracks how often articles appear in impressions with temporal windows
**Input**: `behaviors.parquet`
**Features**: Count features with time windows (past/future 5m, 1h, all-time) and ranking:
- `c_common/user_count_past_5m/1h/all` - Appearance counts in different time windows
- `c_common/user_count_*_ratio` - Normalized ratios within impression
- `c_common/user_count_*_rank_ascending/descending` - Rankings within impression
**Calculation**: Rolling time-based counting + within-impression normalization and ranking

### `c_appear_imp_count_read_time_per_inview_v7` - Read Time Accumulation
**What it does**: Similar to appearance counts but tracks accumulated read time
**Input**: `behaviors.parquet`
**Features**: Same structure as `c_appear_imp_count_v7` but for `read_time_per_inview`
**Calculation**: Time-windowed read time aggregation + ranking and ratio features

### `c_article_imp_rank` - Article Ranking Within Impressions
**What it does**: Ranks articles within each impression by engagement metrics
**Input**: `candidate.parquet` + `articles.parquet`
**Features**: Rank and ratio features for:
- `total_inviews/pageviews/read_time` rankings within impression
- `sentiment_score` ranking within impression
**Calculation**: Within-impression ranking (1 = highest) + proportion calculation

### `c_article_publish_time_v5` - Article Freshness Features
**What it does**: Temporal features based on time since publication
**Input**: `behaviors.parquet` + `articles.parquet`
**Features**:
- `c_time_sec/min/hour/day_diff` - Time differences in various units
- `c_time_*_diff_rn` - Ranking of freshness within impression
**Calculation**: `impression_time - published_time` → convert to different units → rank within impression

### Content Similarity Features (`c_*_tfidf_*_sim`, `c_*_svd_sim`)

#### `c_title_tfidf_svd_sim` - Title-Based User-Article Similarity
**What it does**: Measures similarity between user reading history and candidate article titles
**Input**: `articles.parquet` + `history.parquet`
**Features**:
- `c_title_tfidf_svd_sim` - Cosine similarity score [0, 1]
- `c_title_tfidf_svd_rn` - Ranking of similarity within impression
**Calculation**:
1. TF-IDF vectorization of article titles
2. SVD dimensionality reduction (50 components)
3. User profile = mean of historical article embeddings (L2 normalized)
4. Cosine similarity between user profile and candidate article
5. Ranking within user's impression

#### Similar Features:
- `c_body_tfidf_svd_sim` - Body text similarity
- `c_subtitle_tfidf_svd_sim` - Subtitle similarity
- `c_category_tfidf_sim` - Category-based similarity
- `c_subcategory_tfidf_sim` - Subcategory similarity
- `c_entity_groups_tfidf_sim` - Named entity similarity
- `c_ner_clusters_tfidf_sim` - NER cluster similarity

#### `c_title_count_svd_sim` - Count Vector Similarity
**What it does**: Uses count vectorization instead of TF-IDF for title similarity
**Method**: Count vectors → SVD → user profiling → similarity computation

#### `c_topics_sim_count_svd` - Topic-Based Similarity
**What it does**: Similarity based on article topic fields
**Method**: Count vectorization of topics → SVD → cosine similarity

#### `c_multi_sim_count_svd` - Multi-Field Content Similarity
**What it does**: Combines multiple content fields for comprehensive similarity
**Fields**: Category + subcategory + NER + entity_groups
**Method**: Concatenate fields → count vectorization → SVD → similarity

#### `c_ner_clusters_sim_count_svd` - NER Cluster Similarity
**What it does**: Specialized similarity based on named entity recognition clusters
**Method**: Count vectorization of NER clusters → SVD → similarity scoring

### `c_is_already_clicked` - User Familiarity
**What it does**: Indicates if user has previously clicked the article
**Input**: `history.parquet` + `candidate.parquet`
**Features**: `c_is_already_clicked` - Boolean flag (1 = previously clicked)
**Calculation**: Set membership check: `article_id in user_click_history`

### `c_is_viewed` - Recent Viewing History
**What it does**: Tracks recent viewing patterns with sliding windows
**Input**: `behaviors.parquet`
**Features**: `c_is_viewed_1/2/5/10/20` - Boolean flags for viewing in last N impressions
**Calculation**: Sliding window lookback with user-specific tracking

---

## 3. Impression Features (`i_*`)

### `i_base_feat` - Core Impression Features
**What it does**: Extracts fundamental session and user context features
**Input**: `behaviors.parquet`
**Features**:
- `i_read_time/scroll_percentage` - Direct engagement metrics
- `i_device_type` - User's device (mobile, desktop, etc.)
- `i_is_sso_user/is_subscriber` - User account properties
- `i_gender/age/postcode` - User demographics
- `i_num_article_ids_inview` - Count of articles in impression
**Calculation**: Direct extraction + list length for article count

### `i_article_stat_v2` - Article Statistics Within Impressions
**What it does**: Statistical aggregation of article properties within each impression
**Input**: `behaviors.parquet` + `articles.parquet`
**Features**: Mean and standard deviation of:
- `i_time_min_diff_mean/std` - Time difference statistics
- `i_total_inviews/pageviews/read_time_mean/std` - Engagement statistics
- `i_sentiment_score_mean/std` - Sentiment statistics
**Calculation**: Join candidate articles with article metadata → group by impression → calculate statistics

### `i_stat_feat` - Additional Impression Statistics
**What it does**: Extended statistical features for impression characterization
**Method**: Similar aggregation approach for additional metrics

### `i_viewtime_diff` - Temporal Viewing Patterns
**What it does**: Analyzes time-based viewing behavior within sessions
**Method**: Time difference calculations and pattern extraction

---

## 4. User Features (`u_*`)

### `u_stat_history` - User Behavioral Profile
**What it does**: Comprehensive statistical profile of user's historical behavior
**Input**: `history.parquet`
**Features**:
- `u_history_len` - Total number of historical interactions
- Scroll percentage statistics: `u_scroll_percentage_fixed_min/max/mean/sum/skew/std`
- Read time statistics: `u_read_time_fixed_min/max/mean/sum/skew/std`
**Calculation**: Explode user history → group by user → calculate statistical aggregations

### `u_click_article_stat_v2` - User Article Preferences
**What it does**: Statistical analysis of user's clicked article characteristics  
**Input**: `history.parquet` + `articles.parquet`
**Features**: User-level statistics of clicked articles' properties
**Calculation**: Join user history with article features → group by user → statistical aggregation

---

## 5. User-Article Features (`ua_*`)

### `ua_topics_sim_count_svd_feat` - Topic-Based Embeddings
**What it does**: Creates topic-space embeddings for both users and articles
**Input**: `articles.parquet` + `history.parquet`
**Features**:
- `topic_user_emb_{0-9}` - 10-dimensional user embeddings in topic space
- `topic_article_emb_{0-9}` - 10-dimensional article embeddings in topic space
**Calculation**:
1. CountVectorizer on concatenated article topics → SVD (10 components)
2. Article embeddings = SVD-transformed topic vectors (L2 normalized)
3. User embeddings = mean of user's historical article embeddings (L2 normalized)
**Output**: Separate `user_feat.parquet` and `article_feat.parquet` files

---

## 6. Target Features (`y_*`)

### `y_transition_prob_from_first` - Sequential Click Patterns
**What it does**: Models sequential article transitions to predict next click
**Input**: `history.parquet`
**Features**:
- `y_transition_prob_from_first` - Probability of transitioning from user's last clicked article
- `y_transition_count_from_first` - Raw count of observed transitions
**Calculation**:
1. Build article-to-article transition matrix from user click sequences
2. For each user, identify last clicked article in history
3. Calculate transition probability from last article to each candidate
4. Handle cold start with global transition probabilities

---

## Feature Engineering Techniques

### Common Patterns:
1. **Temporal Windows**: 5-minute, 1-hour, and all-time aggregations
2. **Within-Group Normalization**: Rankings and ratios within impressions/users
3. **Text Processing**: TF-IDF → SVD → L2 normalization → cosine similarity
4. **Statistical Aggregation**: min/max/mean/sum/std/skew across multiple dimensions
5. **Sequential Modeling**: User behavior tracking and transition analysis

### Technical Implementation:
- **Polars**: High-performance data processing
- **Scikit-learn**: TF-IDF, CountVectorizer, SVD implementations
- **Temporal Logic**: Rolling windows and cumulative statistics
- **Similarity Computing**: Cosine similarity with normalized embeddings
- **Memory Management**: Efficient chunked processing for large datasets

---

## Output Structure
Each feature extraction module outputs:
```
features/{feature_name}/{size}/
├── train_feat.parquet    # Training set features
└── validation_feat.parquet  # Validation set features
```

Features are later combined in the dataset creation phase to form the final training dataset for machine learning models.

This comprehensive feature engineering pipeline creates 100+ features covering content similarity, user behavior, temporal patterns, and interaction dynamics essential for state-of-the-art news recommendation systems.