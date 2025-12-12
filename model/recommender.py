import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

print(" Loading Data...")
try:
    df = pd.read_csv('data/cleaned_perfumes.csv')
except FileNotFoundError:
    df = pd.read_csv('../data/cleaned_perfumes.csv')

print(f" Data Loaded. Total Rows: {len(df)}")

print(" Building Vector Matrix...")
vectorizer = TfidfVectorizer(stop_words='english',min_df=2)
tfdif_matrix = vectorizer.fit_transform(df['Notes'].fillna(''))

def get_recommendations(perfume_name, top_n=5):
    mask = df['Name'].str.contains(perfume_name, case=False, na=False) | \
           df['Brand'].str.contains(perfume_name, case=False, na=False)
    
    matches = df[mask]

    if matches.empty:
        return [f"Error: Perfume '{perfume_name}' not found."]
    idx = matches.index[0]
    target_vector=tfdif_matrix[idx]
    cosine_scores=linear_kernel(target_vector,tfdif_matrix).flatten()

    sim_scores = list(enumerate(cosine_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    perfume_indices = [i[0] for i in sim_scores]
    return df[['Name', 'Brand', 'Image_URL']].iloc[perfume_indices].values.tolist()