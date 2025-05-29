import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (7, 5)

# Load & preprocess
df = pd.read_csv("C:/Chinnu work/breast cancer(1).csv")
df.drop(columns=['id'], inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(
    StandardScaler().fit_transform(X), y
)

# Elbow Method (1 to 30)
error_rates = [
    1 - accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test))
    for k in range(1, 30)
]

# Plot Elbow
sns.lineplot(x=range(1, 30), y=error_rates, marker='o', color='plum')
plt.title("🔍 Elbow Method to Find Optimal k")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Error Rate")
plt.xticks(range(1, 30))
plt.show()

# Train final model (example: k=7)
k = 7
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Positive', 'Negative'],
    cmap='Purples'
)
plt.title("🧾 Confusion Matrix")
plt.show()

# Print Metrics
print(f"✅ Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print(f"✅ Precision: {precision_score(y_test, y_pred):.2f}")
print(f"✅ Recall:    {recall_score(y_test, y_pred):.2f}")
print(f"✅ F1 Score:  {f1_score(y_test, y_pred):.2f}")
