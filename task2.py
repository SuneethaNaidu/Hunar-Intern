import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("C:/Chinnu work/house_price_data(1).csv")

print("BEFORE CLEANING")
print(df.info())
print(df.isnull().sum())

# Remove exact duplicate rows
df.drop_duplicates(inplace=True)

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Drop rows with missing values
df.dropna(inplace=True)

# Drop non-numeric and non-useful columns for prediction
df = df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1)

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))
