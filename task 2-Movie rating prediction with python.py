import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

def clean_year(year_val):
    if isinstance(year_val, str):
        return int(year_val.strip('()'))
    return year_val

def clean_duration(duration_val):
    if isinstance(duration_val, str):
        return int(duration_val.replace(' min', ''))
    return duration_val

df['Year'] = df['Year'].apply(clean_year)
df['Duration'] = df['Duration'].apply(clean_duration)

df['Votes'] = df['Votes'].astype(str)
df['Votes'] = df['Votes'].str.replace('$', '', regex=False)
df['Votes'] = df['Votes'].str.replace('M', '', regex=False)
df['Votes'] = df['Votes'].str.replace('k', '', regex=False)
df['Votes'] = df['Votes'].str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

df.dropna(subset=['Rating'], inplace=True)

numerical_cols = ['Year', 'Duration', 'Votes']
categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna('Unknown', inplace=True)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df[['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
sample_data = X_test.iloc[0].values.reshape(1, -1)
predicted_rating = model.predict(sample_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
ax1.set_xlabel('Actual Ratings')
ax1.set_ylabel('Predicted Ratings')
ax1.set_title('Actual vs. Predicted Ratings')

ax2.axis('off')
text_content = (
    f"Model Performance\n"
    f"{'='*25}\n\n"
    f"Mean Squared Error: {mse:.4f}\n"
    f"R-squared Score: {r2:.4f}\n\n"
    f"Sample Prediction:\n"
    f"Predicted Rating: {predicted_rating[0]:.2f}"
)
ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace')

fig.suptitle('IMDb Movie Rating Prediction - Evaluation', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
