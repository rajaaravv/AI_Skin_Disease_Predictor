import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model("models/skin_model.h5")

# Class labels
data_dir = "data/train"
class_names = os.listdir(data_dir)

def predict_skin_disease(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = class_names[class_idx]
    confidence = float(np.max(prediction))*100

    return class_label, confidence

# Example prediction
test_image = "data/test/sample.jpg"  # Replace with your test image
label, conf = predict_skin_disease(test_image)
print(f"Disease: {label}, Confidence: {conf:.2f}%")
