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
tfidf_matrix = vectorizer.fit_transform(df['Notes'].fillna(''))

def get_recommendations(user_input, top_n=5):
    if not user_input:
        return["Error no input."]
    query = str(user_input).strip()
    mask = df['Name'].str.contains(query, case=False, na=False) | \
           df['Brand'].str.contains(query, case=False, na=False)
    
    matches = df[mask]
    target_vector=None
    search_type="perfume"

    if not matches.empty:
        idx = matches.index[0]
        target_vector=tfidf_matrix[idx]
        print(f"MATCHED PERFUME:{df.iloc[idx]['Name']}")
    
    else:
        print(f"treating '{query}'as raw ingredient")
        search_type="ingredient"
        target_vector=vectorizer.transform([query])
         
        if target_vector.nnz==0:
            return[f"Error: The word '{query}' isn't in our scent vocabulary."]
    cosine_scores=linear_kernel(target_vector,tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    start_index=1 if search_type=="perfume" else 0
    sim_scores = sim_scores[start_index:top_n+start_index]
    
    perfume_indices = [i[0] for i in sim_scores]
    return df[['Name', 'Brand', 'Image_URL']].iloc[perfume_indices].values.tolist()