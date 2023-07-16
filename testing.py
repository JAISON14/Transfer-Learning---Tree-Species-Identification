# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:05:41 2023

@author: Jaison
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Efficientnetb0 imports
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Model saved with Keras model.save()
MODEL_PATH = 'C:/Users/Jaison/Documents/Workspace/Aidi/1011 AI Project/AI-project-deployment/models/EfficientnetB0.h5'

# Load your trained model
#model = load_model(MODEL_PATH)
# Load the model and make a prediction
custom_objects = {'FixedDropout': tf.keras.layers.Dropout}  # Add the custom object
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# Load the list of species names
species_path="C:/Users/Jaison/Documents/Workspace/Aidi/1011 AI Project/AI-project-deployment/class_info/name_of_spesies.txt"
with open(species_path, 'r') as f:
    species_list = [line.strip() for line in f]

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(256, 256))
    # Load the image
    print("In model_predict function")
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    train_datagen= ImageDataGenerator(rescale=1.0/255)
    x = train_datagen.flow(x, shuffle=False).next()
    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

def upload():
#    if request.method == 'POST':
        # Get the file from post request
#        f = request.files['file']

        # Save the file to ./uploads
#        basepath = os.path.dirname(__file__)
#        file_path = os.path.join(
#            basepath, 'uploads', secure_filename(f.filename))
#        f.save(file_path)
    file_path="C:/Users/Jaison/Documents/Zoom/Class1.jpg"
    # Make prediction
    preds = model_predict(file_path, model)

    # Process your result for human
    # pred_class = preds.argmax(axis=-1)            # Simple argmax
#        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
#        result = str(pred_class[0][0][1])               # Convert to string
    # Get the predicted class name
    pred_class_idx = np.argmax(preds)
    pred_class = species_list[pred_class_idx]
    result = str(pred_class)
    return result
    return None

upload()