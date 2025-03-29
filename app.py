from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

model = tf.keras.models.load_model('catVdogCNN.h5')

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
if __name__ == "__main__":
    app.run(debug=True)
