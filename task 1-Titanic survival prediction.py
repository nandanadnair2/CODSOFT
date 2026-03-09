import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def preprocess_data(df):
    """
    Standard data preparation function.
    Fills missing values and encodes categorical variables.
    """
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[feature_columns]
    y = df['Survived']

    return X, y

def train_model(X_train, y_train):
    """
    Basic model training function using Random Forest.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    try:
        df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\B.Tech\nandana\Titanic-Dataset.csv')
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    ax2.axis('off')
    text_content = (
        f"Model Performance\n"
        f"{'='*25}\n\n"
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
        f"Classification Report:\n"
        f"{report}"
    )
    ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')

    fig.suptitle('Titanic Survival Model - Evaluation Results', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()
