from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('svm_model.pkl')  # Load the saved SVM model and label encoder
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    # Prepare the input feature vector
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    prediction_encoded = model.predict(features)      # Predict the crop
    prediction = label_encoder.inverse_transform(prediction_encoded)
    
    return render_template('index.html', prediction_text=f'Recommended Crop: {prediction[0]}')   # Return the prediction to be displayed on the webpage

if __name__ == '__main__':
    app.run(debug=True)
