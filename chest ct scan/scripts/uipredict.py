import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

# Load the new model
new_model = tf.keras.models.load_model(r"C:\Users\dest4\Desktop\DATA\chest_ct\weights\chest_cancer_detection_model.h5", compile=False)
new_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

# Define the class labels for the new model
class_labels = ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma']

# Function to perform inference on the selected image
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the test image
        original_image = Image.open(file_path)
        resized_image = original_image.resize((150, 150))
        test_image = image.img_to_array(resized_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        test = image.load_img(file_path, target_size=(150,150))
        test = image.img_to_array(test)
        test = np.expand_dims(test, axis=0)

        # Preprocess the test image (rescale to [0, 1])
        test = test / 255.0
        # Perform inference with the new model
        predictions = new_model.predict(test)
        predicted_class = np.argmax(predictions)

        # Get the corresponding class label
        class_label = class_labels[predicted_class]

        # Display the predicted class label
        result_label.config(text=f"Prediction: {class_label}")

        # Display the selected image
        image_obj = ImageTk.PhotoImage(resized_image)
        image_label.config(image=image_obj)
        image_label.image = image_obj  # Keep a reference to the image

# Create the main window
root = tk.Tk()
root.title("Göğüs Kanseri Teşhisi")

# Set the icon (replace 'icon.png' with your PNG file)
icon_image = tk.PhotoImage(file=r'C:\Users\dest4\Desktop\bulutklinik-V1.2\main\icon.png')  # Specify the path to your PNG file
root.iconphoto(False, icon_image)
root.geometry("240x400")

# Create a label to display the image
image_label = Label(root)
image_label.pack()

# Create a frame to center the button
button_frame = tk.Frame(root)
button_frame.pack()

# Create a button to browse and predict an image
browse_button = Button(button_frame, text="Resim Seç", command=predict_image)
browse_button.pack()

# Create a label to display the prediction result
result_label = Label(root, text="", font=("Helvetica", 14))
result_label.pack()

# Start the main event loop
root.mainloop()
