from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import preprocess_data
from crop_recommendation import load_crop_data, recommend_crops

app = Flask(__name__)
model = load_model('models\soil_classification_model.h5')
crop_df = load_crop_data('data\crops.csv')

train_folder = 'data/soil_images/train'
test_folder = 'data/soil_images/test'

train_images, train_labels_encoded, test_images, test_labels_encoded, label_encoder = preprocess_data(train_folder, test_folder)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def index():
    return render_template('index.html', title='Soil Classification & Crop Recommendation')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', title='Soil Classification & Crop Recommendation', error='No file provided')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', title='Soil Classification & Crop Recommendation', error='No file selected')

    if file:
        file_path = os.path.join('uploads', file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        image = preprocess_image(file_path)
        predictions = model.predict(image)
        predicted_label_index = np.argmax(predictions)
        prediction = label_encoder.inverse_transform([predicted_label_index])[0]
        confidence_score = predictions[0][predicted_label_index]
        confidence_threshold = 0.8
        
        
        if confidence_score < confidence_threshold :
                return render_template('result.html', title='Prediction Result', soil_type=None , confidence = confidence_score)
        else:
            confidence_threshold = 0.95
            if confidence_score < confidence_threshold :
                sure = 0
            else:
                sure = 1

            season = request.form.get('season', 'Rabi')  # Default season
            suitable_crops = recommend_crops(prediction, season, crop_df)
            return render_template('result.html', title='Prediction Result', soil_type=prediction, season=season, suitable_crops=suitable_crops, confidence = confidence_score, match = sure)

if __name__ == '__main__':
    app.run(debug=True)
