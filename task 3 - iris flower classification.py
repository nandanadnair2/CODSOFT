import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    df = pd.read_csv('IRIS.csv')
except FileNotFoundError:
    print("Error: IRIS.csv file not found. Please check the file name.")
    exit()

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)
sample_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample_data)
predicted_species = le.inverse_transform(prediction)

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axs[0, 0])
axs[0, 0].set_title('Confusion Matrix')
axs[0, 0].set_xlabel('Predicted')
axs[0, 0].set_ylabel('Actual')

axs[0, 1].axis('off')
text_content = (
    f"Model Performance\n"
    f"{'='*25}\n\n"
    f"Accuracy: {accuracy * 100:.2f}%\n\n"
    f"Classification Report:\n"
    f"{report}"
)
axs[0, 1].text(0.05, 0.95, text_content, transform=axs[0, 1].transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')

axs[1, 0].axis('off')
data_head_str = "Dataset Head (First 5 Rows):\n" + df.head().to_string()
axs[1, 0].text(0.05, 0.95, data_head_str, transform=axs[1, 0].transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')

axs[1, 1].axis('off')
prediction_str = (
    f"Sample Prediction\n"
    f"{'='*25}\n\n"
    f"Input Data: {sample_data[0]}\n"
    f"Predicted Species: '{predicted_species[0]}'"
)
axs[1, 1].text(0.05, 0.95, prediction_str, transform=axs[1, 1].transAxes, fontsize=12,
               verticalalignment='top')

fig.suptitle('Iris Species Classification - Model Evaluation', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
