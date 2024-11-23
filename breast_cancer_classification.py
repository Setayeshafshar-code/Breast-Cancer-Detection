# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
import joblib


# Constants
DATASET_PATH = './data/Breast_Cancer_Dataset.csv'
PROCESSED_DATASET_PATH = './data/breast_cancer_binary.csv'
SEED = 123

def save_model(file_name, model):
    joblib.dump(model, file_name)

def get_feature_names(dataset_path='./data/Breast_Cancer_Dataset.csv', target_column='diagnosis'):
    """Extract feature names from the dataset."""
    df = pd.read_csv(dataset_path)
    features = df.drop(columns=[target_column]).columns.tolist()
    return features
    
def save_to_file(file_name, content):
    with open(file_name, "w") as file:
        file.write(content)

def describe_data(data):
    return data.describe(include='all')

# Function to preprocess the data
def load_data(file_path):
    """
    Load and preprocess the dataset.
    """
    df = pd.read_csv(file_path)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Map 'M' -> 1, 'B' -> 0
    if 'id' in df.columns:
        df = df.drop(columns=['id'])  # Drop unnecessary 'id' column
    df.to_csv(PROCESSED_DATASET_PATH, index=False)  # Save the processed data
    print(f"Preprocessed data saved to {PROCESSED_DATASET_PATH}")
    print(df.head)
    return df

# Function to visualize data
def get_important_features_of_data(df):
    """
    Generate visualizations for the dataset.
    """
    # Heatmap of correlations
    # Define features and target variable
# Replace 'target' with the name of your target column
    target = 'diagnosis'
    X = df.drop(columns=[target])
    y = df[target]

    # 1. Correlation Analysis
    correlation_matrix = df.corr()
    correlation_with_target = correlation_matrix[target].drop(target)

    # Display correlations
    print("Correlation with Target Variable:")
    print(correlation_with_target)

    # 2. Feature Importance Using RandomForest
    # Determine if classification or regression is needed
    if y.nunique() <= 2:
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    # Train the model
    model.fit(X, y)

    # Extract feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    # 3. Remove Low-Importance Features
    # Set a threshold for importance (e.g., 0.01)
    threshold = 0.01
    important_features = feature_importances[feature_importances['Importance'] >= threshold]['Feature']

    # Reduce the dataset
    X_reduced = X[important_features]

    print("\nFeatures retained after reduction:")
    print(important_features.tolist())
    data_reduced = df[important_features.tolist() + [target]]
    return X_reduced, y


def preprocess_data(data):
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(data)
    return X_scaled
def split_data(df, val_ratio=0.1, test_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets.
    """
    n = len(df)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
   
    np.random.seed(SEED)
    idx = np.random.permutation(n)
   
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
   
    df_train, df_val, df_test = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    return df_train, df_val, df_test

# Function to scale the features
def scale_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

# Function to train a logistic regression model
def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    """
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)
    return model

# Function to train a decision tree classifier
def train_decision_tree(X_train, y_train, max_depth=4):
    """
    Train a decision tree classifier.
    """
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=SEED)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X, y, target_names=['Benign', 'Malignant']):
    """
    Evaluate the model using validation/test data.
    """
    y_pred = model.predict(X)
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=target_names))
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return accuracy

# Function to visualize the decision tree
def visualize_decision_tree(model, feature_names):
    """
    Visualize a decision tree.
    """
    plt.figure(figsize=(25, 10))
    plot_tree(model, feature_names=feature_names, class_names=['Benign', 'Malignant'],
              filled=True, rounded=True, fontsize=10)
    plt.show()
    