from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

import torch

model = torch.load(r'C:\Users\Arjun Datta PC\OneDrive\Desktop\GRID\resnet50_fruits_model.pth', map_location=torch.device('cpu'))

# Define a function to classify the image
def classify_image(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)

    # Get the class with the highest probability
    class_index = np.argmax(predictions[0])
    class_name = ['class1', 'class2', 'class3'][class_index]

    return class_name

# Define a route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for the classify page
@app.route('/classify', methods=['POST'])
def classify():
    # Get the image from the request
    image = request.files['image']

    # Convert the image to a numpy array
    image_array = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Classify the image
    class_name = classify_image(image)

    # Return the result
    return render_template('result.html', class_name=class_name)

if __name__ == '__main__':
    app.run(debug=True)