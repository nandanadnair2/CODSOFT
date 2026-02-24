import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For building the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. LOAD THE DATASET
# Make sure the CSV file is in the same folder as your script
df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

# 2. DATA CLEANING
# The dataset has messy data (e.g., Year is '(2019)', Votes is '1,086'). We need to fix this.

# Function to convert Year (remove parentheses)
def clean_year(year_val):
    if isinstance(year_val, str):
        # Extract digits from string like "(2019)"
        return int(year_val.strip('()'))
    return year_val

# Function to convert Duration (remove ' min')
def clean_duration(duration_val):
    if isinstance(duration_val, str):
        return int(duration_val.replace(' min', ''))
    return duration_val

# Apply cleaning functions# Clean Votes: Remove commas and convert to float
df['Year'] = df['Year'].apply(clean_year)
df['Duration'] = df['Duration'].apply(clean_duration)

# Clean Votes: Remove '$', 'M', 'k', commas and convert to float
df['Votes'] = df['Votes'].astype(str)
df['Votes'] = df['Votes'].str.replace('$', '', regex=False)
df['Votes'] = df['Votes'].str.replace('M', '', regex=False)
df['Votes'] = df['Votes'].str.replace('k', '', regex=False)
df['Votes'] = df['Votes'].str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# 3. HANDLING MISSING VALUES
# We cannot predict Rating if the Rating itself is missing. So we drop those rows.
df.dropna(subset=['Rating'], inplace=True)

# Fill missing values in other columns with the mode (most frequent value) or mean
# For simplicity, we fill numerical columns with mean and categorical with 'Unknown'
numerical_cols = ['Year', 'Duration', 'Votes']
categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna('Unknown', inplace=True)

# 4. FEATURE ENGINEERING (Encoding)
# Machine learning models need numbers, not text. We convert text columns to numbers.
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Convert all values to string to ensure consistent encoding
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 5. SPLITTING DATA
# X = Features (Input), y = Target (Output - Rating)
X = df[['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Rating']

# Split into 80% Training data and 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. BUILDING THE MODEL
# Let's try Linear Regression first (simplest), then Random Forest (usually better)
model = LinearRegression()
# You can switch to: model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# 7. PREDICTION & EVALUATION
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}") # Closer to 1.0 is better

# Visualizing Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Movie Ratings')
plt.show()

# Example: Predict rating for a new movie (using the first row of test set as an example)
sample_data = X_test.iloc[0].values.reshape(1, -1)
predicted_rating = model.predict(sample_data)
print(f"Predicted Rating for sample: {predicted_rating[0]}")
