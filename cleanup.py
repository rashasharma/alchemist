import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    df = pd.read_csv('data/raw_perfumes.csv', encoding='utf-8')
except Exception:
    df = pd.read_csv('data/raw_perfumes.csv', encoding='unicode_escape')

def extract_brand(url):
    try:
        return str(url).split('/perfume/')[1].split('/')[0].replace('-', ' ').strip()
    except Exception:
        return 'Unknown'

df['Brand'] = df['url'].apply(extract_brand)

def clean_perfume_name(row):
    name = str(row['Name']).strip()
    brand = str(row['Brand']).strip()
    suffixes = [
        'for women and men', 'for women', 'for men',
        'for-women-and-men', 'for-women', 'for-men'
    ]
    for s in suffixes:
        if name.lower().endswith(s.lower()):
            name = name[:-len(s)].strip()
            break
    brand_variants = [brand, brand.replace(' ', ''), brand.replace('&', '')]
    for b in brand_variants:
        if b and name.lower().endswith(b.lower()):
            name = name[:-len(b)].strip()
            break
    name = re.sub(r'\s+', ' ', name).strip()
    return name

df['Name_Clean'] = df.apply(clean_perfume_name, axis=1)

df['Rating_Count_Clean'] = (
    df['Rating Count']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)
df['Rating_Count_Clean'] = pd.to_numeric(df['Rating_Count_Clean'], errors='coerce').fillna(0).astype(int)
df['Rating_Value_Clean'] = pd.to_numeric(df['Rating Value'], errors='coerce').fillna(0.0).astype(float)

df['Description'] = df['Description'].fillna('')
df['Main Accords'] = df['Main Accords'].fillna('')
df['Perfumers'] = df['Perfumers'].fillna('[]')
df['Gender'] = df['Gender'].fillna('for women and men')

FAMILY_MAP = {
    'floral': 'floral',
    'fruity': 'fruity',
    'woody': 'woody',
    'citrus': 'citrus',
    'amber': 'amber',
    'oriental': 'amber',
    'vanilla': 'vanilla',
    'musk': 'musky',
    'musky': 'musky',
    'aromatic': 'aromatic',
    'green': 'green',
    'spicy': 'warm spicy',
    'leather': 'leather',
    'aquatic': 'aquatic',
    'marine': 'marine',
    'chypre': 'mossy',
    'gourmand': 'sweet',
    'sweet': 'sweet',
    'powdery': 'powdery',
    'earthy': 'earthy',
    'aldehydic': 'aldehydic',
    'fougere': 'aromatic',
    'animalic': 'animalic',
    'smoky': 'smoky',
    'balsamic': 'balsamic'
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

def extract_accords(row):
    raw_acc = str(row['Main Accords']).strip()
    if raw_acc and raw_acc not in ['[]', 'None', 'nan']:
        return raw_acc
    desc = str(row['Description']).strip()
    if not desc:
        return '[]'
    extracted = []
    m = re.search(r'is\s+an?\s+([^.]+?)\s+fragrance', desc, re.IGNORECASE)
    if m:
        family_text = m.group(1).lower()
        for word, accord in FAMILY_MAP.items():
            if re.search(rf'\b{word}\b', family_text):
                if accord not in extracted:
                    extracted.append(accord)
    if len(extracted) < 4:
        for note, accord in NOTE_MAP.items():
            if re.search(rf'\b{note}\b', desc, re.IGNORECASE):
                if accord not in extracted:
                    extracted.append(accord)
            if len(extracted) >= 6:
                break
    if extracted:
        return str(extracted)
    return '[]'

df['Main Accords'] = df.apply(extract_accords, axis=1)

notes_combined = (df['Main Accords'] + ' ') * 3 + df['Description']

vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
tfidf_matrix = vectorizer.fit_transform(notes_combined)

final_df = pd.DataFrame({
    'Name': df['Name_Clean'],
    'Brand': df['Brand'],
    'Description': df['Description'],
    'Page_URL': df['url'],
    'Gender': df['Gender'],
    'Rating_Value': df['Rating_Value_Clean'],
    'Rating_Count': df['Rating_Count_Clean'],
    'Perfumers': df['Perfumers'],
    'Main_Accords': df['Main Accords']
})

final_df.to_csv('data/cleaned_perfumes.csv', index=False, encoding='utf-8')

os.makedirs('model/serialized', exist_ok=True)

with open('model/serialized/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f, protocol=4)

with open('model/serialized/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f, protocol=4)
