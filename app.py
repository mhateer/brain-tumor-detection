from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import imutils

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('brain-tumor.model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Access the uploaded file from the request
    file = request.files['file']
    
    # Read and process the uploaded image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_copy = image.copy()  # Make a copy of the original image
    
    # Preprocess the image
    image = crop_brain_contour(image_copy)
    image = cv2.resize(image, (240, 240))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Make the prediction using the loaded model
    prediction = model.predict(image)
    result = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor Detected'

    if prediction[0][0] > 0.5:
        # Find the contours of the tumor region
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a circle around the tumor region
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.circle(image_copy, (int(x + w/2), int(y + h/2)), int(max(w, h)/2), (0, 255, 0), 2)
    
    # Encode the image with circle as base64 string
    _, img_encoded = cv2.imencode('.png', image_copy)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # Render the result template with the prediction and image
    return render_template('result.html', prediction=result, image=img_base64)

def crop_brain_contour(image):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop the new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    return new_image

if __name__ == '__main__':
    app.run()
