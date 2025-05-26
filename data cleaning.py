import pandas as pd
#load the dataset
df=pd.read_csv("C:\Users\Anitha\OneDrive\Documents\food_coded-1(1).csv")
print("BEFORE CLEANING")
print(df.info())
print(df.isnull().sum())
#remove exact duplicates
df.drop_duplicates(inplace=True)
#remove duplicate columns
df.df.loc[:, -df.columns.duplicated()]
df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
# Step 5: Fill missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        # numeric fill with median
        median = df[col].median()
        df[col].fillna(median, inplace=True)
    else:
        # categorical/text fill with mode
        mode = df[col].mode()
        if not mode.empty:
            df[col].fillna(mode[0], inplace=True)
# Final check
print(" AFTER CLEANING")
print(df.info())
print(df.isnull().sum())

# Step 6: Save cleaned version
cleaned_path = "cleaned_food_coded-1.csv"
df.to_csv(cleaned_path, index=False)
print(cleaned_path)
