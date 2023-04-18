
# %tensorflow_version 2.x
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from pickle import dump
print(tf.__version__)
print(keras.__version__)

model = VGG16()
print(model.summary())

image=load_img('/content/s0006_00351_0_1_0_0_0_01.png', target_size=(224, 224))
plt.imshow(image)

image = tf.keras.preprocessing.image.img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
model.layers.pop()
model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-1].outputs)

# get extracted features
features = model.predict(image)
print(features.shape)

# save to file
dump(features, open('/content/trained.data','wb'))








