from flask import Flask, request, jsonify, render_template # type: ignore
from flask_cors import CORS # type: ignore
import numpy as np
import tensorflow as tf
from PIL import Image
import io


app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins":"*"}})

model = tf.keras.models.load_model('predictor_model.h5')
@app.route("/")
def home():
    return render_template("index.html")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((128,128))
    image_array = np.array(image)/255.0
    if image_array.shape[-1] == 4:
        image_array = image_array[:,:,:3]
    image_array = np.expand_dims(image_array,axis=0)
    return image_array

@app.route("/predict",methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error":"no image in file"})
    file = request.files["image"]
    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)
    predicted_class = int(prediction[0][0]>0.5)

    return jsonify({"prediction":predicted_class})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 locally
    app.run(host="0.0.0.0", port=port)