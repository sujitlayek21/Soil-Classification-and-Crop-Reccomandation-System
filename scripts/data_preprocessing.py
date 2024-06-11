import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_images(image_folder):
    images = []
    labels = []
    for label in os.listdir(image_folder):
        for image_file in os.listdir(os.path.join(image_folder, label)):
            image_path = os.path.join(image_folder, label, image_file)
            image = cv2.imread(image_path)
            #print(image_path)
            image = cv2.resize(image, (128, 128))  # Resize images to a fixed size
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data(train_folder, test_folder):
    train_images, train_labels = load_images(train_folder)
    test_images, test_labels = load_images(test_folder)
    
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    return train_images, train_labels_encoded, test_images, test_labels_encoded, label_encoder
