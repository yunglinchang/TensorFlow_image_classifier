import tensorflow as tf
import tensorflow_hub as hub

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import json
import numpy as np
import argparse 
from PIL import Image

def format_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    im = np.asarray(im)
    im = format_image(im)
    #add the extra dimension
    im = np.expand_dims(im, axis=0)
    
    ps = model.predict(im)
    
    values, indices = tf.nn.top_k(ps, k=top_k)
    values = values.numpy().tolist()[0]
    indices = indices.numpy().tolist()[0]

    return values, indices

parser = argparse.ArgumentParser(description = 'Deep Neural Network Flower Prediction')
parser.add_argument('image_path')
parser.add_argument('saved_model')
parser.add_argument('--top_k', help='Return the top K most likely classes', type=int, default=3)
parser.add_argument('--category_names', help='Path to a JSON file mapping labels to flower names', default='label_map.json')

args = parser.parse_args()
image_path = args.image_path
model = args.saved_model
top_k = args.top_k
json_file = args.category_names

with open(json_file, 'r') as f:
    class_names = json.load(f)
    
my_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
values, indices = predict(image_path, my_model, top_k)

labels = []
for label in indices:
    labels.append(class_names[str(label+1)])

print('\nResult:')
print('The most possible flower class: ', labels[0])
print('The probability predicting that class: %.3f%%' % (values[0]*100))
print()
