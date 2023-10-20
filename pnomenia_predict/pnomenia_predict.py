import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.models import Sequential



model = tf.keras.models.load_model('pnomenia_predict\pnomenia_model.h5',compile=False)
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
test_image_path = 'pnomenia_predict\IM-0001-0001.jpeg'
test_image = image.load_img(test_image_path, target_size=(150,150), color_mode="grayscale")
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Preprocess the test image (rescale to [0, 1])
test_image = test_image / 255.0

# Perform inference
predictions = model.predict(test_image)
print(predictions)