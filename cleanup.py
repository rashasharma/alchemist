import pandas as pd

df = pd.read_csv('data/raw_perfumes.csv', encoding='unicode_escape')

print("Original Shape (Rows, Columns):", df.shape)
print("\n test")
print(df.head())

df_cleaned = df.dropna(subset=['Notes'])
df_cleaned['Brand'] = df_cleaned['Brand'].str.title()
df_cleaned['Name'] = df_cleaned['Name'].str.title()

df_cleaned.to_csv('data/cleaned_perfumes.csv', index=False)

print(f"Original Perfumes: {len(df)}")
print(f"Cleaned Perfumes:  {len(df_cleaned)}")
print(f"Removed {len(df) - len(df_cleaned)} empty rows.")
print("Saved to: data/cleaned_perfumes.csv")