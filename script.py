from breast_cancer_classification import *
from flask import Flask, request, jsonify
import joblib


if __name__ == "__main__":
    # Preprocess the data
    df = load_data(DATASET_PATH)
    describe_result = describe_data(df)
    save_to_file("./results/data_description.txt", f"Data Description:\n\n{describe_result}")

    X, y = get_important_features_of_data(df)
    save_to_file("./results/important_features.txt", f"Important Features:\n\n{X.columns.tolist()}")

    X_scaled = preprocess_data(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    save_to_file("./results/train_test_split.txt", f"Train-Test Split Summary:\n\n"
                                         f"X_train shape: {X_train.shape}\n"
                                         f"X_test shape: {X_test.shape}\n"
                                         f"y_train distribution:\n{y_train.value_counts()}\n"
                                         f"y_test distribution:\n{y_test.value_counts()}")

    # Logistic Regression
    print("\nTraining Logistic Regression Model...")
    logistic_model = train_logistic_regression(X_train, y_train)
    save_model("./saved_model/logistic.pkl", logistic_model)
    logistic_eval = evaluate_model(logistic_model, X_test, y_test)
    save_to_file("./results/logistic_regression_results.txt", f"Logistic Regression Evaluation:\n\n{logistic_eval}")

    # Decision Tree Classifier
    print("\nTraining Decision Tree Model...")
    tree_model = train_decision_tree(X_train, y_train)
    save_model("./saved_model/tree.pkl", tree_model)
    tree_eval = evaluate_model(tree_model, X_test, y_test)
    save_to_file("./results/decision_tree_results.txt", f"Decision Tree Evaluation:\n\n{tree_eval}")

    print("\nAll results have been saved to text files.")

    