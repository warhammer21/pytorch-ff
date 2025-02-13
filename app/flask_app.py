from flask import Flask, request, jsonify, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
import torch
from nn_model import Net  # Import the model class
import os

app = Flask(__name__)

# Swagger UI Setup
SWAGGER_URL = '/swagger-ui'
API_URL = '/swagger.json'  # The URL to your Swagger JSON file
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Flask API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Load the trained model
model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set model to evaluation mode

# Serve the Swagger JSON file from the 'app' directory
@app.route('/swagger.json')
def swagger():
    return send_from_directory(os.getcwd(), 'app/swagger.json')  # Serve from 'app' folder

# Home route
@app.route('/')
def home():
    return "Welcome to the API! Visit /swagger-ui for API documentation."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data from POST request
    features = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)  # Add batch dimension (1 sample)

    # Ensure that the input tensor shape is (1, 3) before passing to the model
    if features.shape[1] != 2:  # Make sure there are exactly 3 features
        return jsonify({'error': 'Input data must have exactly 2 features.'})

    # Make prediction (no need for labels here, just the features)
    with torch.no_grad():
        prediction = model(features)  # Pass only features to the model

    # Return prediction as JSON (no labels involved)
    return jsonify({'prediction': prediction.item()})

    # Return prediction as JSON

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
