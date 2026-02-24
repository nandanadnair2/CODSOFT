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
    # Handle missing numerical data with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Handle missing categorical data with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Drop column with excessive missing values
    df.drop('Cabin', axis=1, inplace=True)

    # Map categorical text to integer values
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Select feature columns
    feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[feature_columns]
    y = df['Survived']

    return X, y

def train_model(X_train, y_train):
    """
    Basic model training function using Random Forest.
    """
    # Initialize classifier with standard parameters
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    return model

def main():
    # Main execution pipeline
    try:
        # Load the dataset
        df = pd.read_csv('Titanic-Dataset.csv')
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return

    # Prepare the data
    X, y = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = train_model(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    main()
