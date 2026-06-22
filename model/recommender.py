import os
import re
import pickle
import ast
import difflib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

csv_path = os.path.join(PROJECT_DIR, 'data', 'cleaned_perfumes.csv')
df = pd.read_csv(csv_path)

df['Name'] = df['Name'].fillna('')
df['Brand'] = df['Brand'].fillna('')
df['Gender'] = df['Gender'].fillna('for women and men')
df['Rating_Value'] = pd.to_numeric(df['Rating_Value'], errors='coerce').fillna(0.0)
df['Rating_Count'] = pd.to_numeric(df['Rating_Count'], errors='coerce').fillna(0).astype(int)
df['Perfumers'] = df['Perfumers'].fillna('[]')
df['Main_Accords'] = df['Main_Accords'].fillna('')
df['Description'] = df['Description'].fillna('')

FAMILY_MAP = {
    'floral': 'floral', 'fruity': 'fruity', 'woody': 'woody', 'citrus': 'citrus',
    'amber': 'amber', 'oriental': 'amber', 'vanilla': 'vanilla', 'musk': 'musky',
    'musky': 'musky', 'aromatic': 'aromatic', 'green': 'green', 'spicy': 'warm spicy',
    'leather': 'leather', 'aquatic': 'aquatic', 'marine': 'marine', 'chypre': 'mossy',
    'gourmand': 'sweet', 'sweet': 'sweet', 'powdery': 'powdery', 'earthy': 'earthy',
    'aldehydic': 'aldehydic', 'fougere': 'aromatic', 'animalic': 'animalic',
    'smoky': 'smoky', 'balsamic': 'balsamic'
}

NOTE_MAP = {
    'bergamot': 'citrus', 'lemon': 'citrus', 'orange': 'citrus', 'mandarin': 'citrus',
    'grapefruit': 'citrus', 'lime': 'citrus', 'neroli': 'citrus',
    'rose': 'rose', 'jasmine': 'white floral', 'tuberose': 'tuberose', 'lily': 'white floral',
    'vanilla': 'vanilla', 'amber': 'amber', 'musk': 'musky', 'patchouli': 'patchouli',
    'cedar': 'woody', 'sandalwood': 'woody', 'vetiver': 'woody', 'oud': 'oud', 'agarwood': 'oud',
    'lavender': 'lavender', 'iris': 'iris', 'violet': 'violet', 'leather': 'leather',
    'cinnamon': 'cinnamon', 'cardamom': 'warm spicy', 'pepper': 'fresh spicy',
    'caramel': 'caramel', 'honey': 'honey', 'chocolate': 'chocolate', 'coffee': 'coffee',
    'coconut': 'coconut', 'peach': 'fruity', 'apple': 'fruity', 'pear': 'pear'
}

def extract_accords_from_desc(desc):
    if not isinstance(desc, str) or not desc.strip():
        return []
    extracted = []
    m = re.search(r'is\s+an?\s+([^.]+?)\s+fragrance', desc, re.IGNORECASE)
    if m:
        family_text = m.group(1).lower()
        for word, accord in FAMILY_MAP.items():
            if re.search(rf'\b{word}\b', family_text):
                if accord not in extracted:
                    extracted.append(accord)
    for note, accord in NOTE_MAP.items():
        if re.search(rf'\b{note}\b', desc, re.IGNORECASE):
            if accord not in extracted:
                extracted.append(accord)
        if len(extracted) >= 8:
            break
    return extracted

def parse_accords(accord_str, description=''):
    accords = []
    if accord_str:
        accord_str = str(accord_str).strip()
        if accord_str.startswith('['):
            try:
                accords = ast.literal_eval(accord_str)
            except Exception:
                accords = []
        else:
            accords = [a.strip().lower() for a in accord_str.split(',') if a.strip()]
    if description:
        desc_accords = extract_accords_from_desc(description)
        for a in desc_accords:
            if a not in accords:
                accords.append(a)
    return accords

def rank_matches(matches_df, query_str):
    q = query_str.strip().lower()
    names = matches_df['Name'].str.strip().str.lower()
    brands = matches_df['Brand'].str.strip().str.lower()
    full_names = brands + ' ' + names
    rev_names = names + ' ' + brands
    exact_mask = (names == q) | (full_names == q) | (rev_names == q)
    prefix_mask = (names.str.startswith(q)) | (full_names.str.startswith(q))
    ranks = np.where(exact_mask, 0, np.where(prefix_mask, 1, 2))
    res = matches_df.copy()
    res['_rank'] = ranks
    return res.sort_values(by=['_rank', 'Rating_Count'], ascending=[True, False])

vectorizer_path = os.path.join(BASE_DIR, 'serialized', 'vectorizer.pkl')
matrix_path = os.path.join(BASE_DIR, 'serialized', 'tfidf_matrix.pkl')

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(matrix_path, 'rb') as f:
    tfidf_matrix = pickle.load(f)

rating_counts = df['Rating_Count'].values
max_log_count = np.log1p(rating_counts.max()) if rating_counts.max() > 0 else 1
norm_popularity = np.log1p(rating_counts) / max_log_count
norm_rating = df['Rating_Value'].values / 5.0
quality_scores = 0.5 * norm_popularity + 0.5 * norm_rating

search_lookup = []
for idx, row in df.iterrows():
    combined = f"{row['Brand']} {row['Name']}".strip().lower()
    search_lookup.append((combined, int(idx)))

search_names_list = [item[0] for item in search_lookup]

SYNONYMS = {
    'ysl': 'yves saint laurent',
    'd&g': 'dolce gabbana',
    'dg': 'dolce gabbana',
    'tf': 'tom ford',
    'mfk': 'maison francis kurkdjian',
    'mhk': 'maison francis kurkdjian',
    'jpg': 'jean paul gaultier',
    'dior': 'dior',
    'adg': 'acqua di gio',
    'br540': 'baccarat rouge 540'
}

def get_recommendations(user_input, top_n=5, gender_filter=None, brand_filter=None):
    if not user_input:
        return {
            'search_type': 'error',
            'message': 'Please enter a perfume name or ingredient.',
            'recommendations': []
        }

    raw_query = str(user_input).strip().lower()
    words = raw_query.split()
    mapped_words = [SYNONYMS.get(w, w) for w in words]
    query_clean = ' '.join(mapped_words)

    tokens = query_clean.split()
    if not tokens:
        return {
            'search_type': 'error',
            'message': 'Empty search query.',
            'recommendations': []
        }

    mask = pd.Series(True, index=df.index)
    for token in tokens:
        escaped = re.escape(token)
        mask &= (df['Brand'].str.lower().str.contains(escaped, na=False) |
                 df['Name'].str.lower().str.contains(escaped, na=False))

    matches = df[mask]

    if matches.empty and len(query_clean) >= 4:
        close_matches = difflib.get_close_matches(query_clean, search_names_list, n=1, cutoff=0.72)
        if close_matches:
            matched_text = close_matches[0]
            alt_tokens = matched_text.split()
            alt_mask = pd.Series(True, index=df.index)
            for token in alt_tokens:
                escaped = re.escape(token)
                alt_mask &= (df['Brand'].str.lower().str.contains(escaped, na=False) |
                             df['Name'].str.lower().str.contains(escaped, na=False))
            matches = df[alt_mask]

    target_vector = None
    search_type = 'perfume'
    matched_perfume = None
    candidates = []

    if not matches.empty:
        matches_sorted = rank_matches(matches, query_clean)
        if len(matches) > 1:
            search_type = 'multiple_matches'
            for idx, row in matches_sorted.head(15).iterrows():
                candidates.append({
                    'id': int(idx),
                    'name': row['Name'],
                    'brand': row['Brand'],
                    'rating': float(row['Rating_Value']),
                    'reviews': int(row['Rating_Count']),
                    'gender': row['Gender']
                })

        best_match_idx = matches_sorted.index[0]
        target_vector = tfidf_matrix[best_match_idx]
        matched_row = df.iloc[best_match_idx]
        matched_perfume = {
            'id': int(best_match_idx),
            'name': matched_row['Name'],
            'brand': matched_row['Brand'],
            'rating': float(matched_row['Rating_Value']),
            'reviews': int(matched_row['Rating_Count']),
            'gender': matched_row['Gender'],
            'perfumers': matched_row['Perfumers'],
            'page_url': matched_row['Page_URL'],
            'notes': matched_row['Description'],
            'accords': parse_accords(matched_row['Main_Accords'], matched_row['Description'])
        }
    else:
        search_type = 'ingredient'
        target_vector = vectorizer.transform([query_clean])

    if target_vector is None or target_vector.nnz == 0:
        return {
            'search_type': 'error',
            'message': f"The term '{user_input}' is not in our scent vocabulary.",
            'recommendations': []
        }

    seed_accords = matched_perfume['accords'] if matched_perfume else tokens

    cosine_scores = linear_kernel(target_vector, tfidf_matrix).flatten()

    boosted_scores = np.where(
        cosine_scores > 0.02,
        0.85 * cosine_scores + 0.15 * quality_scores,
        cosine_scores
    )

    excluded_indices = set()
    if matched_perfume is not None:
        excluded_indices.add(matched_perfume['id'])
        duplicate_mask = (df['Name'].str.lower() == matched_perfume['name'].lower()) & \
                         (df['Brand'].str.lower() == matched_perfume['brand'].lower())
        for idx in df[duplicate_mask].index:
            excluded_indices.add(idx)

    valid_mask = np.ones(len(df), dtype=bool)

    if gender_filter:
        g = gender_filter.lower()
        if g == 'women':
            valid_mask &= df['Gender'].str.lower().isin(['for women', 'for women and men'])
        elif g == 'men':
            valid_mask &= df['Gender'].str.lower().isin(['for men', 'for women and men'])
        elif g == 'unisex':
            valid_mask &= df['Gender'].str.lower().isin(['for women and men'])

    if brand_filter:
        valid_mask &= (df['Brand'].str.lower() == brand_filter.lower())

    sorted_indices = np.argsort(boosted_scores)[::-1]

    recommendations = []
    for idx in sorted_indices:
        if len(recommendations) >= top_n:
            break
        if idx in excluded_indices or not valid_mask[idx]:
            continue

        row = df.iloc[idx]
        rec_accords = parse_accords(row['Main_Accords'], row['Description'])
        shared = [a for a in rec_accords if a in seed_accords]
        unique = [a for a in rec_accords if a not in seed_accords]

        recommendations.append({
            'name': row['Name'],
            'brand': row['Brand'],
            'page_url': row['Page_URL'],
            'gender': row['Gender'],
            'rating': float(row['Rating_Value']),
            'reviews': int(row['Rating_Count']),
            'perfumers': row['Perfumers'],
            'similarity': float(cosine_scores[idx]),
            'notes': row['Description'],
            'accords': rec_accords,
            'shared_accords': shared,
            'unique_accords': unique
        })

    return {
        'search_type': search_type,
        'matched_perfume': matched_perfume,
        'candidates': candidates,
        'recommendations': recommendations
    }

def get_suggestions(query, limit=10):
    if not query:
        return []

    raw_query = str(query).strip().lower()
    words = raw_query.split()
    mapped_words = [SYNONYMS.get(w, w) for w in words]
    query_clean = ' '.join(mapped_words)

    tokens = query_clean.split()
    if not tokens:
        return []

    mask = pd.Series(True, index=df.index)
    for token in tokens:
        escaped = re.escape(token)
        mask &= (df['Brand'].str.lower().str.contains(escaped, na=False) |
                 df['Name'].str.lower().str.contains(escaped, na=False))

    matches = df[mask]
    if matches.empty:
        return []

    matches_sorted = rank_matches(matches, query_clean)

    suggestions = []
    seen = set()
    for _, row in matches_sorted.iterrows():
        if len(suggestions) >= limit:
            break
        key = (row['Name'].lower().strip(), row['Brand'].lower().strip())
        if key in seen:
            continue
        seen.add(key)
        suggestions.append({
            'name': row['Name'],
            'brand': row['Brand'],
            'gender': row['Gender'],
            'rating': float(row['Rating_Value']),
            'reviews': int(row['Rating_Count'])
        })

    return suggestions
