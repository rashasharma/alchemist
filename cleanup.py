import pandas as pd
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

print("--- Starting Data Cleanup & Pre-computation ---")

try:
    df = pd.read_csv('data/raw_perfumes.csv', encoding='utf-8')
except:
    df = pd.read_csv('data/raw_perfumes.csv', encoding='unicode_escape')

print(f"Original shape: {df.shape}")

def get_brand(url):
    try:
        # Extract brand from URL slug and de-hyphenate
        brand = url.split('/perfume/')[1].split('/')[0]
        return brand.replace('-', ' ').strip()
    except:
        return "Unknown"

def clean_name(row):
    name = str(row['Name'])
    brand = str(row['Brand'])
    
    # Case-insensitive replacement of brand name
    if brand.lower() != "unknown":
        pattern = re.compile(re.escape(brand), re.IGNORECASE)
        name = pattern.sub("", name)
    
    # Case-insensitive removal of gender suffixes in name
    for suffix in ["for women and men", "for women", "for men"]:
        pattern = re.compile(re.escape(suffix), re.IGNORECASE)
        name = pattern.sub("", name)
    
    # Clean up double spaces
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

print("Extracting and de-hyphenating Brands...")
df['Brand'] = df['url'].apply(get_brand)

print("Cleaning Perfume Names...")
df['Name_Clean'] = df.apply(clean_name, axis=1)

# Combine accords (weighted) and description
df['Notes_Combined'] = (df['Main Accords'].fillna('') + " ") * 3 + df['Description'].fillna('')

# Select final columns including the new rich metadata and main accords!
final_df = df[[
    'Name_Clean', 'Brand', 'Notes_Combined', 'url', 
    'Gender', 'Rating Value', 'Rating Count', 'Perfumers', 'Main Accords'
]]
final_df.columns = [
    'Name', 'Brand', 'Notes', 'Page_URL', 
    'Gender', 'Rating_Value', 'Rating_Count', 'Perfumers', 'Main_Accords'
]

# Drop rows where Notes combined is empty
final_df = final_df.dropna(subset=['Notes'])

print("Saving cleaned dataset...")
final_df.to_csv('data/cleaned_perfumes.csv', index=False, encoding='utf-8')
print(f"Saved {len(final_df)} cleaned rows.")

print("Building TF-IDF Vector Matrix...")
vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
tfidf_matrix = vectorizer.fit_transform(final_df['Notes'].fillna(''))
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

print("Serializing Model Artifacts...")
os.makedirs('model/serialized', exist_ok=True)
with open('model/serialized/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model/serialized/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

print("Cleaning and Serialization Complete!")
print("Saved to: data/cleaned_perfumes.csv and model/serialized/")