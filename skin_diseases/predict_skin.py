import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('skin_model.h5')

class_labels = ['Egzema', 'Melanoma', 'Atopic Dermatitis', 'Basal cell carcinoma', "Melanytic Nevi","Benign Keratosis-like Lesions", "Psorriasis ", "Seborrheic Keratosses", "Tinea Ringworm Candidiasis", "Warts Molluscum"]

test_image_path = 'IMG_CLASSES\8. Seborrheic Keratoses and other Benign Tumors - 1.8k/v-seborrheic-keratosis-irritated-145.jpg'
test_image = image.load_img(test_image_path, target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Preprocess the test image (rescale to [0, 1])
test_image = test_image / 255.0

# Perform inference
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions)

# Print the classification label
print(f'Predicted Class: {class_labels[predicted_class]}')