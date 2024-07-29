from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
print("Flask app created")

try:
    # Load model
    model = joblib.load('model.pkl')
    print("Model loaded successfully")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    print("Rendering home page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded"
    
    print("Handling prediction request")
    try:
        features = [float(x) for x in request.form.values()]
        print(f"Received features: {features}")
        final_features = pd.DataFrame([features], columns=[
            'Length1', 'Length2', 'Length3', 'Height', 'Width',
            'Species_Parkki', 'Species_Perch', 'Species_Pike',
            'Species_Roach', 'Species_Smelt', 'Species_Whitefish'
        ])
        prediction = model.predict(final_features)
        print(f"Prediction: {prediction}")
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        print(f"Error making prediction: {e}")
        return f"Error making prediction: {e}"

if __name__ == "__main__":
    print("Starting Flask app")
    app.run(debug=True)
