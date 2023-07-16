# Standard library imports
import os
from typing import List

# Third-party imports
import numpy as np
import tensorflow as tf
from flask import Flask, Response, request, render_template
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Global variables (consider moving these to a separate config file)
MODEL_PATH = 'models/EfficientnetB0.h5'
SPECIES_PATH = 'class_info/name_of_spesies.txt'

# Load your trained model
custom_objects = {'FixedDropout': tf.keras.layers.Dropout}  # Add the custom object
model = load_model(MODEL_PATH, custom_objects=custom_objects)

# Load the list of species names
with open(SPECIES_PATH, 'r') as f:
    species_list = [line.strip() for line in f]


def load_and_preprocess_image(img_path: str) -> np.array:
    img = load_img(img_path, target_size=(256, 256))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    return train_datagen.flow(x, shuffle=False).next()


def predict_species(img_path: str, model) -> str:
    x = load_and_preprocess_image(img_path)
    preds = model.predict(x)
    pred_class_idx = np.argmax(preds)
    return species_list[pred_class_idx]


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    try:
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        result = predict_species(file_path, model)
        app.logger.info(f'Result: {result}')
        return Response(result, content_type='text/plain')
    except Exception as e:
        app.logger.error(f'An error occurred: {str(e)}')
        return Response(str(e), content_type="text/plain")


if __name__ == '__main__':
    app.run(debug=True)
