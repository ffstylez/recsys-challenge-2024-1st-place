import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import pickle
import os
from pathlib import Path

# --- CONFIGURATION ---
# We use relative paths to find the data folder based on your repository structure
# Assuming we are in src/exp/v0xxx_preprocess
BASE_DATA_DIR = Path(__file__).resolve().parents[3] / "data" 
HISTORY_PATH = BASE_DATA_DIR / "history.parquet"
ARTICLES_PATH = BASE_DATA_DIR / "articles.parquet"
OUTPUT_PATH = BASE_DATA_DIR / "user_jsd_scores.pkl"

def calculate_jsd():
    print(f"Loading data from: {BASE_DATA_DIR}")
    
    # 1. Load Articles & Map to Categories
    if not ARTICLES_PATH.exists():
        print(f"ERROR: Could not find {ARTICLES_PATH}")
        return

    df_articles = pd.read_parquet(ARTICLES_PATH)
    # Map Article ID -> Category Index
    all_categories = sorted(df_articles['category_str'].unique())
    cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    article_to_cat_idx = {}
    
    for _, row in df_articles.iterrows():
        cat = row['category_str']
        if cat in cat_to_idx:
            article_to_cat_idx[row['article_id']] = cat_to_idx[cat]

    # 2. Load History
    if not HISTORY_PATH.exists():
        print(f"ERROR: Could not find {HISTORY_PATH}")
        return
        
    df_history = pd.read_parquet(HISTORY_PATH)
    print("Calculating Global Distribution (Average User)...")
    
    # Calculate global category counts
    global_counts = np.zeros(len(all_categories))
    
    # Iterate through history to sum counts (using fixed article IDs)
    # This might take a moment
    for history_list in df_history['article_id_fixed']:
        for aid in history_list:
            if aid in article_to_cat_idx:
                idx = article_to_cat_idx[aid]
                global_counts[idx] += 1
                
    # Normalize to probabilities
    if global_counts.sum() > 0:
        global_dist = global_counts / global_counts.sum()
    else:
        global_dist = np.ones(len(all_categories)) / len(all_categories)
        
    print(f"Global Distribution calculated over {len(all_categories)} categories.")

    # 3. Calculate JSD per User
    print("Calculating JSD for each user...")
    user_jsd_map = {}
    
    for _, row in df_history.iterrows():
        user_id = row['user_id']
        history = row['article_id_fixed']
        
        user_counts = np.zeros(len(all_categories))
        valid_clicks = 0
        
        for aid in history:
            if aid in article_to_cat_idx:
                user_counts[article_to_cat_idx[aid]] += 1
                valid_clicks += 1
        
        if valid_clicks > 0:
            user_dist = user_counts / valid_clicks
            # Calculate JSD (square it to get the divergence metric)
            score = jensenshannon(user_dist, global_dist) ** 2
            user_jsd_map[user_id] = float(score)
        else:
            user_jsd_map[user_id] = 0.0

    # 4. Save
    print(f"Saving scores for {len(user_jsd_map)} users to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(user_jsd_map, f)
    print("DONE! JSD Scores ready.")

if __name__ == "__main__":
    calculate_jsd()
