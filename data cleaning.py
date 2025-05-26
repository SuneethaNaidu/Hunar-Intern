import pandas as pd
# Load the dataset
df = pd.read_csv(r"C:\Users\Anitha\OneDrive\Documents\food_coded-1(1).csv")
print("BEFORE CLEANING")
print(df.info())
print(df.isnull().sum())
# Remove exact duplicate rows
df.drop_duplicates(inplace=True)
# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]
# Convert GPA column to numeric
df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
# Fill missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        # Fill numeric columns with median (no inplace)
        median = df[col].median()
        df[col] = df[col].fillna(median)
    else:
        # Fill categorical columns with mode (no inplace)
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])
# Final check after cleaning
print("AFTER CLEANING")
print(df.info())
print(df.isnull().sum())
# Save cleaned dataset
cleaned_path = "cleaned_food_coded-1.csv"
df.to_csv(cleaned_path, index=False)
print("Cleaned file saved to:", cleaned_path)
