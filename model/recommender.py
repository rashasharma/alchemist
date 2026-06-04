import os
import re
import pickle
import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

# Resolve absolute paths dynamically relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Load cleaned dataset
csv_path = os.path.join(PROJECT_DIR, 'data', 'cleaned_perfumes.csv')
print(" Loading cleaned perfumes dataset...")
df = pd.read_csv(csv_path)

# Ensure columns are filled appropriately and match types
df['Name'] = df['Name'].fillna('')
df['Brand'] = df['Brand'].fillna('')
df['Gender'] = df['Gender'].fillna('')
df['Rating_Value'] = pd.to_numeric(df['Rating_Value'], errors='coerce').fillna(0.0)
df['Rating_Count'] = pd.to_numeric(df['Rating_Count'], errors='coerce').fillna(0)
df['Perfumers'] = df['Perfumers'].fillna('')
df['Main_Accords'] = df['Main_Accords'].fillna('')

def parse_accords(accord_str):
    try:
        if not accord_str:
            return []
        accord_str = str(accord_str).strip()
        if accord_str.startswith('['):
            return ast.literal_eval(accord_str)
        else:
            return [a.strip().lower() for a in accord_str.split(',') if a.strip()]
    except:
        return []

# Load serialized vectorizer and TF-IDF matrix
vectorizer_path = os.path.join(BASE_DIR, 'serialized', 'vectorizer.pkl')
matrix_path = os.path.join(BASE_DIR, 'serialized', 'tfidf_matrix.pkl')

print(" Loading pre-computed TF-IDF model artifacts...")
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
with open(matrix_path, 'rb') as f:
    tfidf_matrix = pickle.load(f)

# Pre-calculate quality scores for popularity-weighted recommendations
rating_counts = df['Rating_Count'].values
max_log_count = np.log1p(rating_counts.max()) if rating_counts.max() > 0 else 1
norm_popularity = np.log1p(rating_counts) / max_log_count

# Normalized rating (0 to 1)
norm_rating = df['Rating_Value'].values / 5.0

# Quality score is the average of popularity and rating (0 to 1)
quality_scores = 0.5 * norm_popularity + 0.5 * norm_rating

print(" Recommendation Engine Loaded!")

# Abbreviation mapping dictionary
SYNONYMS = {
    "ysl": "yves saint laurent",
    "d&g": "dolce & gabbana",
    "tf": "tom ford",
    "mfk": "maison francis kurkdjian",
    "mhk": "maison francis kurkdjian",
    "jpg": "jean paul gaultier",
    "dior": "christian dior"
}

def get_recommendations(user_input, top_n=5, gender_filter=None, brand_filter=None):
    """
    Finds recommendations based on a perfume search or a raw ingredient.
    Supports token-intersection search, abbreviation mapping, duplicate skipping, 
    popularity weighting, and category filters.
    """
    if not user_input:
        return {
            "search_type": "error",
            "message": "Please enter a perfume name or ingredient.",
            "recommendations": []
        }
    
    query = str(user_input).strip().lower()
    
    # Replace synonyms
    words = query.split()
    mapped_words = [SYNONYMS.get(w, w) for w in words]
    query_clean = " ".join(mapped_words)
    
    # Split query into tokens for robust intersection search
    tokens = query_clean.split()
    if not tokens:
        return {
            "search_type": "error",
            "message": "Empty search query.",
            "recommendations": []
        }
        
    # Check if tokens intersect with Brand or Name
    mask = pd.Series(True, index=df.index)
    for token in tokens:
        escaped_token = re.escape(token)
        mask &= (df['Brand'].str.lower().str.contains(escaped_token, na=False) |
                 df['Name'].str.lower().str.contains(escaped_token, na=False))
        
    matches = df[mask]
    
    target_vector = None
    search_type = "perfume"
    matched_perfume = None
    candidates = []
    
    if not matches.empty:
        # Sort matches by rating count to put the most popular match first
        matches_sorted = matches.sort_values(by='Rating_Count', ascending=False)
        
        # If there are multiple matches, we can capture the candidates for the user
        if len(matches) > 1:
            search_type = "multiple_matches"
            # Limit candidates to top 15
            for idx, row in matches_sorted.head(15).iterrows():
                candidates.append({
                    "id": int(idx),
                    "name": row['Name'],
                    "brand": row['Brand'],
                    "rating": float(row['Rating_Value']),
                    "reviews": int(row['Rating_Count']),
                    "gender": row['Gender']
                })
        
        # Pick the best matched perfume as the primary seed
        best_match_idx = matches_sorted.index[0]
        target_vector = tfidf_matrix[best_match_idx]
        matched_row = df.iloc[best_match_idx]
        matched_perfume = {
            "id": int(best_match_idx),
            "name": matched_row['Name'],
            "brand": matched_row['Brand'],
            "rating": float(matched_row['Rating_Value']),
            "reviews": int(matched_row['Rating_Count']),
            "gender": matched_row['Gender'],
            "perfumers": matched_row['Perfumers'],
            "page_url": matched_row['Page_URL'],
            "notes": matched_row['Notes'],
            "accords": parse_accords(matched_row['Main_Accords'])
        }
        print(f"Matched perfume: {matched_perfume['name']} by {matched_perfume['brand']}")
        
    else:
        # Treat query as a raw ingredient search
        print(f"Treating '{query}' as raw ingredient")
        search_type = "ingredient"
        target_vector = vectorizer.transform([query_clean])

    # Establish seed accords for comparison
    if matched_perfume is not None:
        seed_accords = matched_perfume["accords"]
    elif search_type == "ingredient":
        seed_accords = tokens
    else:
        seed_accords = []
        
        if target_vector.nnz == 0:
            return {
                "search_type": "error",
                "message": f"The word '{user_input}' is not in our scent vocabulary.",
                "recommendations": []
            }
            
    # Calculate similarity scores
    cosine_scores = linear_kernel(target_vector, tfidf_matrix).flatten()
    
    # Apply popularity-weighted quality boosting
    # Boost by 15% quality score if similarity is above a threshold
    sim_threshold = 0.02
    boosted_scores = np.where(
        cosine_scores > sim_threshold,
        0.85 * cosine_scores + 0.15 * quality_scores,
        cosine_scores
    )
    
    # Skip self-recommendation if search_type is a single matched perfume
    excluded_indices = set()
    if matched_perfume is not None:
        excluded_indices.add(matched_perfume['id'])
        
        # Also let's exclude duplicate perfumes with the exact same name and brand to prevent duplicate leaks!
        duplicate_mask = (df['Name'].str.lower() == matched_perfume['name'].lower()) & \
                         (df['Brand'].str.lower() == matched_perfume['brand'].lower())
        for idx in df[duplicate_mask].index:
            excluded_indices.add(idx)

    # Filter recommendations before sorting based on UI requested filters
    valid_mask = np.ones(len(df), dtype=bool)
    
    # Apply gender filter if requested
    if gender_filter:
        gender_filter = gender_filter.lower()
        if gender_filter == "women":
            # Show "for women" and "for women and men"
            valid_mask &= df['Gender'].str.lower().isin(['for women', 'for women and men'])
        elif gender_filter == "men":
            # Show "for men" and "for women and men"
            valid_mask &= df['Gender'].str.lower().isin(['for men', 'for women and men'])
        elif gender_filter == "unisex":
            # Show "for women and men"
            valid_mask &= df['Gender'].str.lower().isin(['for women and men'])
            
    # Apply brand filter if requested
    if brand_filter:
        valid_mask &= (df['Brand'].str.lower() == brand_filter.lower())
        
    # Get sorted list of recommendations
    sorted_indices = np.argsort(boosted_scores)[::-1]
    
    recommendation_results = []
    for idx in sorted_indices:
        if len(recommendation_results) >= top_n:
            break
        if idx in excluded_indices:
            continue
        if not valid_mask[idx]:
            continue
            
        row = df.iloc[idx]
        rec_accords = parse_accords(row['Main_Accords'])
        shared = [a for a in rec_accords if a in seed_accords]
        unique = [a for a in rec_accords if a not in seed_accords]
        
        recommendation_results.append({
            "name": row['Name'],
            "brand": row['Brand'],
            "page_url": row['Page_URL'],
            "gender": row['Gender'],
            "rating": float(row['Rating_Value']),
            "reviews": int(row['Rating_Count']),
            "perfumers": row['Perfumers'],
            "similarity": float(cosine_scores[idx]),
            "notes": row['Notes'],
            "accords": rec_accords,
            "shared_accords": shared,
            "unique_accords": unique
        })
        
    return {
        "search_type": search_type,
        "matched_perfume": matched_perfume,
        "candidates": candidates,
        "recommendations": recommendation_results
    }

def get_suggestions(query, limit=10):
    """
    Returns search suggestions for autocomplete, sorted by popularity.
    """
    if not query:
        return []
    
    query_clean = str(query).strip().lower()
    
    # Replace synonyms in query
    words = query_clean.split()
    mapped_words = [SYNONYMS.get(w, w) for w in words]
    query_clean = " ".join(mapped_words)
    
    tokens = query_clean.split()
    if not tokens:
        return []
        
    # Check if tokens intersect with Brand or Name
    mask = pd.Series(True, index=df.index)
    for token in tokens:
        escaped_token = re.escape(token)
        mask &= (df['Brand'].str.lower().str.contains(escaped_token, na=False) |
                 df['Name'].str.lower().str.contains(escaped_token, na=False))
        
    matches = df[mask]
    if matches.empty:
        return []
        
    # Sort matches by popularity (Rating_Count) so most famous perfumes are suggested first
    matches_sorted = matches.sort_values(by='Rating_Count', ascending=False)
    
    suggestions = []
    # Drop duplicates in case there are identical name/brand rows in the database
    seen = set()
    for _, row in matches_sorted.iterrows():
        if len(suggestions) >= limit:
            break
        key = (row['Name'].lower().strip(), row['Brand'].lower().strip())
        if key in seen:
            continue
        seen.add(key)
        
        suggestions.append({
            "name": row['Name'],
            "brand": row['Brand'],
            "gender": row['Gender'],
            "rating": float(row['Rating_Value']),
            "reviews": int(row['Rating_Count'])
        })
        
    return suggestions