import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('pnomenia_predict\pnomenia_model.h5', compile=False)
model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

# Define the class labels
predictions_class = ["Normal", "Akciğer İltihabı"]

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

        # Perform inference
        predictions = model.predict(test_image)
        if predictions>0.6:
            print(predictions)
            result_label.config(text="Prediction: Normal")
        else:
            print(predictions)
            result_label.config(text="Prediction: Akciğer İltihabı")


        # Display the selected image
        image_obj = ImageTk.PhotoImage(resized_image)
        image_label.config(image=image_obj)
        image_label.image = image_obj  # Keep a reference to the image

# Create the main window
root = tk.Tk()
root.title("Kanser Teşhisi")

# Set the icon (replace 'icon.png' with your PNG file)
icon_image = tk.PhotoImage(file=r'C:\Users\dest4\Desktop\bulutklinik-V1.2\main\icon.png')  # Specify the path to your PNG file
root.iconphoto(False, icon_image)
root.geometry("240x400")

# Create a label to display the image
image_label = Label(root)
image_label.pack()


# Create a button to browse and predict an image
browse_button = Button(root, text="Resim Seç", command=predict_image)
browse_button.pack()

# Create a label to display the prediction result
result_label = Label(root, text="", font=("Helvetica", 14))
result_label.pack()

# Start the main event loop
root.mainloop()
