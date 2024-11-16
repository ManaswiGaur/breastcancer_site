from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model file
model_path = 'model/best_mlp_model.pkl'
model = joblib.load(model_path)  # Load the trained model once, when the app starts

# Define the route for the homepage
@app.route('/')
def home():
    # Render the main HTML page
    return render_template('index.html')

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data and ensure it has the correct feature names
        features = [
            "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
            "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", 
            "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", 
            "concave_points_worst"
        ]

        data = []
        for feature in features:
            feature_value = request.form.get(feature)
            if feature_value is None or feature_value == '':
                return jsonify({"error": f"{feature} is missing or empty"}), 400
            try:
                data.append(float(feature_value))
            except ValueError:
                return jsonify({"error": f"{feature} is not a valid number"}), 400

        # Convert the data to a numpy array and reshape for prediction
        input_data = np.array(data).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_data)[0]
        result = "Malignant" if prediction == 1 else "Benign"
        return jsonify({"result": result})
    
    except Exception as e:
        # Return error details if any exception occurs
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
