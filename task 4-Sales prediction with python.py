import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD THE DATASET
# Assuming your file is named 'advertising.csv'
# If you are using the raw text provided, ensure it is saved as a CSV file first.
df = pd.read_csv('advertising.csv')

# Display the first few rows to understand the data
print("First 5 rows of the dataset:")
print(df.head())

# 2. EXPLORATORY DATA ANALYSIS (EDA)
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Visualize the relationship between Advertising Budgets and Sales
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.suptitle('Advertising Budget vs Sales', y=1.02)
plt.show()

# Check the correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 3. DATA PREPROCESSING
# Define Features (X) and Target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into Training and Testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL BUILDING
# We will use Linear Regression because we are predicting a continuous number (Sales)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# 5. PREDICTIONS
# Predict on the test set
y_pred = model.predict(X_test)

# 6. EVALUATION
# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Interpretation of R2: Closer to 1.0 means the model explains the data very well.

# 7. VISUALIZING RESULTS (Actual vs Predicted)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2) # The diagonal line
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual Sales vs Predicted Sales')
plt.show()

# Example: Predicting sales for a new specific advertising budget
new_budget = pd.DataFrame({'TV': [150], 'Radio': [25], 'Newspaper': [10]})
predicted_sales = model.predict(new_budget)
print(f"\nPredicted Sales for TV=150, Radio=25, Newspaper=10: {predicted_sales[0]:.2f}")