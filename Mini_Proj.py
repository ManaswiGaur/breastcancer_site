from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained ML model
model_path = 'C:\Users\user\Desktop\Breast Cancer\bestmodel.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for the homepage
@app.route('/')
def home():
    # Render the main HTML page
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data from the request
    data = request.form.to_dict()
    try:
        # Convert form inputs to a numpy array
        input_data = np.array([float(value) for value in data.values()]).reshape(1, -1)
        
        # Make a prediction using the ML model
        prediction = model.predict(input_data)[0]
        
        # Define response based on prediction result
        result = "Malignant" if prediction == 1 else "Benign"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
