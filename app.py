from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from breast_cancer_classification import get_feature_names, get_important_features_of_data, load_data

# Initialize Flask app
app = Flask(__name__)  # Make sure the app is initialized here

# Load models
logistic_model = joblib.load('./saved_model/logistic.pkl')
tree_model = joblib.load('./saved_model/tree.pkl')
# Extract feature names dynamically from the dataset
feature_names = get_important_features_of_data(load_data('./data/Breast_Cancer_Dataset.csv'))[0].columns.tolist()


@app.route('/')
def home():
    return render_template('form.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = {feature: float(request.form[feature]) for feature in feature_names}
        input_df = pd.DataFrame([input_data])

        # Make predictions
        logistic_prediction = logistic_model.predict(input_df)[0]
        tree_prediction = tree_model.predict(input_df)[0]

        # Prepare response
        response = {
            "logistic_model_prediction": "Malignant" if logistic_prediction == 1 else "Benign",
            "decision_tree_prediction": "Malignant" if tree_prediction == 1 else "Benign",
        }
        return render_template('result.html', input_data=input_data, response=response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)