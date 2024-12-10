import os
import json
import requests
import zipfile
import tensorflow.keras as tfkeras
from flask import Flask, request, render_template
import numpy as np
import cv2
import random

app = Flask(__name__)


MODEL_URL = "https://drive.google.com/uc?id=1z-trDBG6hd4I6jRQGoy6NlpRW841zrtx"
DATASET_URL = "https://drive.google.com/uc?id=1GjRh8OblDSch33r7OGLFyo07yUkqGuWM"
DATASET_FOLDER = "riotu-cars-dataset-200"

if not os.path.exists("model.h5"):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open("model.h5", "wb") as f:
        f.write(response.content)

if not os.path.exists(DATASET_FOLDER):
    print("Downloading and extracting dataset...")
    response = requests.get(DATASET_URL)
    with open("dataset.zip", "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall("static")
    os.remove("dataset.zip") 

model = tfkeras.models.load_model("model.h5")

with open("categories.json", "r") as f:
    categories = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if not file:
            return "No file selected", 400

        #now processing the image user uploads as input
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "Invalid image format", 400

        img_resized = cv2.resize(img, (224, 224))
        img_normalized = np.expand_dims(img_resized / 255.0, axis=0)

        #predict the generation
        prediction = model.predict(img_normalized)
        raw_label = categories[np.argmax(prediction)]  
        formatted_label = " ".join(word.capitalize() for word in raw_label.split("_"))

        predicted_folder = os.path.join("static", DATASET_FOLDER, *raw_label.split("_"))
        if os.path.exists(predicted_folder) and os.listdir(predicted_folder):
            sample_image_name = random.choice(os.listdir(predicted_folder))
            sample_image_path = os.path.join(predicted_folder, sample_image_name)
            static_image_path = os.path.relpath(sample_image_path, start="static")
        else:
            static_image_path = None

        print(f"Static image path: {static_image_path}")

        return render_template(
            "result.html",
            predicted_label=formatted_label,
            image_path=static_image_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)
