import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('../DATA/skin_disease/weights/skin_model.h5')

class_labels = [" ",'Egzema', 'Melanoma', 'Atopic Dermatitis', 'Basal cell carcinoma', "Melanytic Nevi","Benign Keratosis-like Lesions", "Psorriasis ", "Seborrheic Keratosses", "Tinea Ringworm Candidiasis", "Warts Molluscum"]

test_image_path = r"C:\Users\dest4\Desktop\DATA\skin_disease\IMG_CLASSES\1. Eczema 1677\t-eczema-leg-11.jpg"
test_image = image.load_img(test_image_path, target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Preprocess the test image (rescale to [0, 1])
test_image = test_image / 255.0

# Perform inference
predictions = model.predict(test_image)
print(predictions)
predicted_class = np.argmax(predictions)

# Print the classification label
print(f'Predicted Class: {class_labels[predicted_class]}')