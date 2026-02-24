import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Dataset
# Ensure your file is named 'IRIS.csv' and in the same directory
try:
    df = pd.read_csv('IRIS.csv')
    print("Dataset loaded successfully!")
    print("-" * 30)
except FileNotFoundError:
    print("Error: IRIS.csv file not found. Please check the file name.")
    exit()

# Display the first 5 rows to understand the data
print(df.head())
print("-" * 30)

# 2. Exploratory Data Analysis (Optional but recommended for your video)
# Check for null values
print("Null values:\n", df.isnull().sum())
print("-" * 30)

# 3. Data Preprocessing
# Features (X) are the measurements
# Target (y) is the species
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode the target labels (convert 'Iris-setosa', etc. into numbers 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into Training and Testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Model Training
# We use Random Forest, but you can also try LogisticRegression or KNeighborsClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model Training Completed.")
print("-" * 30)

# 5. Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6. Prediction Example (Showcasing how it works)
# Let's create a sample input to test the model manually
sample_data = [[5.1, 3.5, 1.4, 0.2]] # Example measurements for Iris-setosa
prediction = model.predict(sample_data)
predicted_species = le.inverse_transform(prediction)
print(f"\nSample Prediction: The flower is classified as '{predicted_species[0]}'")

# 7. Visualization (Confusion Matrix) - Great for your video/demo
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()