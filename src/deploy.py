from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the best model and vectorizer
model = joblib.load("../models/xgboost_best_model.pkl")  # Load the best model (XGBoost in this case)
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")  # Load the TF-IDF vectorizer

# Define a route for health check
@app.route('/')
def home():
    return "Model API is running!"

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        
        # Check if 'review' key is in the incoming data
        if 'review' not in data:
            return jsonify({'error': 'No review text provided'}), 400
        
        review_text = data['review']
        
        # Transform the review text using the loaded vectorizer
        review_vector = vectorizer.transform([review_text])
        
        # Make prediction using the model
        prediction = model.predict(review_vector)
        
        # Return the prediction as a JSON response
        result = {'sentiment': int(prediction[0])}  # 0 = Negative, 1 = Positive
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
