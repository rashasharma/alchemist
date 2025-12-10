import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

print("⏳ Loading Data...")
try:
    df = pd.read_csv('data/cleaned_perfumes.csv')
except FileNotFoundError:
    df = pd.read_csv('../data/cleaned_perfumes.csv')

print(f"✅ Data Loaded. Total Rows: {len(df)}")

if len(df) > 20000:
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)
    print(f"⚠️ Data limited to first 20,000 rows for speed.")

print("⏳ Building Vector Matrix...")
vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(df['Notes'].fillna(''))
print("⏳ Calculating Similarity (This might take 10-20 seconds)...")
cosine_sim = linear_kernel(matrix, matrix)

print("✅ Math Complete. The AI is ready.")

def get_recommendations(perfume_name, top_n=5):
    mask = df['Name'].str.contains(perfume_name, case=False, na=False) | \
           df['Brand'].str.contains(perfume_name, case=False, na=False)
    
    matches = df[mask]

    if matches.empty:
        return [f"Error: Perfume '{perfume_name}' not found."]
    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    perfume_indices = [i[0] for i in sim_scores]
    return df[['Name', 'Brand']].iloc[perfume_indices].values.tolist()

if __name__ == "__main__":
    test_perfume = "Musky" 
    print(f"\n🧪 Testing Recommendations for '{test_perfume}':")
    
    results = get_recommendations(test_perfume)
    
    for res in results:
        print(f"- {res[0]} (by {res[1]})")