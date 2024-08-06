from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('../model/food_classifier.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file to the server
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make a prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_labels = list(train_generator.class_indices.keys())  # Assuming you have `train_generator` defined
    predicted_class = class_labels[class_idx]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
