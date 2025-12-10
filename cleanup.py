import pandas as pd

try:
    df = pd.read_csv('data/raw_perfumes.csv', encoding='unicode_escape')
except:
    df = pd.read_csv('data/raw_perfumes.csv', encoding='utf-8')

print("original shape:", df.shape)


def get_brand(url):
    try:
        return url.split('/perfume/')[1].split('/')[0]
    except:
        return "Unknown"

def clean_name(row):
    name = str(row['Name'])
    brand = str(row['Brand'])
    
    name = name.replace(brand, "")
    
    name = name.replace("for women and men", "")
    name = name.replace("for women", "")
    name = name.replace("for men", "")
    
    return name.strip()

print("Extracting Brands...")
df['Brand'] = df['url'].apply(get_brand)

print("Cleaning Names...")
df['Name_Clean'] = df.apply(clean_name, axis=1)

df['Notes'] = df['Main Accords'].fillna('') + " " + df['Description'].fillna('')

final_df = df[['Name_Clean', 'Brand', 'Notes', 'url']]
final_df.columns = ['Name', 'Brand', 'Notes', 'Image_URL'] 

final_df = final_df.dropna(subset=['Notes'])

final_df.to_csv('data/cleaned_perfumes.csv', index=False)

print("\n--------------------------------")
print("Cleaning Complete!")
print(f"Original Rows: {len(df)}")
print(f"Cleaned Rows:  {len(final_df)}")
print(f"Sample Name: {final_df['Name'].iloc[0]}")
print(f"Sample Brand: {final_df['Brand'].iloc[0]}")
print("Saved to: data/cleaned_perfumes.csv")