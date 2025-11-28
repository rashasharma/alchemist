import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    df = pd.read_csv('data/cleaned_perfumes.csv')
    print("✅ Data Loaded Successfully!")
except FileNotFoundError:
    df = pd.read_csv('../data/cleaned_perfumes.csv')
    print("✅ Data Loaded Successfully (via relative path)!")

vectorizer = TfidfVectorizer(stop_words='english')

matrix = vectorizer.fit_transform(df['Notes'].fillna(''))
cosine_sim = cosine_similarity(matrix, matrix)

print("\n--- Matrix Stats ---")
print(f"Perfumes (Rows): {matrix.shape[0]}")
print(f"Unique Notes (Dimensions): {matrix.shape[1]}")

def get_recommendations(perfume_name, top_n=5):
    try:
        idx = df[df['Name'].str.contains(perfume_name, case=False, na=False)].index[0]
    except IndexError:
        return [f"Error: Perfume '{perfume_name}' not found in database."]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    perfume_indices = [i[0] for i in sim_scores]
    return df['Name'].iloc[perfume_indices].tolist()

if __name__ == "__main__":
    search_term = "vanilla"
    print(f"\n🔎 Searching for perfumes containing '{search_term}'...")
    
    matches = df[df['Name'].str.contains(search_term, case=False, na=False)]
    
    if matches.empty:
        print("No matches found. Try a different brand")
    else:
        print(f"Found {len(matches)} matches. Here are the first 5:")
        print(matches['Name'].head(5).tolist())
        valid_name = matches['Name'].iloc[0]
        print(f"\n🧪 Generating recommendations for: '{valid_name}'")
        results = get_recommendations(valid_name)
        for res in results:
            print(f"- {res}")