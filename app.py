from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained KMeans model (ensure the .pkl file exists in the same directory)
model = joblib.load('kmeans_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.json['data']  # Expects input as a list of features
    data = np.array(data).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(data)
    
    # Return the prediction as a JSON response
    return jsonify({'cluster': int(prediction[0])})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4455)
