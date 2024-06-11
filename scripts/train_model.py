import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_preprocessing import preprocess_data

train_folder = 'data/soil_images/train'
test_folder = 'data/soil_images/test'

train_images, train_labels_encoded, test_images, test_labels_encoded, label_encoder = preprocess_data(train_folder, test_folder)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(train_labels_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels_encoded, epochs=10, validation_data=(test_images, test_labels_encoded))

test_loss, test_acc = model.evaluate(test_images, test_labels_encoded)
print(f'Test accuracy: {test_acc}')

model.save('models/soil_classification_model.h5')
