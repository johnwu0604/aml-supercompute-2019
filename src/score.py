from tensorflow.keras.models import load_model
from azureml.core import Model
import requests
import numpy as np
import os
import cv2
import json

# Set search headers and URL
headers = requests.utils.default_headers()
headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'

def init():
	model_dir = Model.get_model_path('nba-player-classifier')
	src_dir = os.path.join(os.getcwd(), 'src')
	global model, face_classifier
	face_classifier = cv2.CascadeClassifier(os.path.join(src_dir, 'haarcascade_frontalface_default.xml'))
	model = load_model(os.path.join(model_dir, 'model.h5'))

def run(raw_data):
	image_dim = 250
	image_url = json.loads(raw_data)['image_url']
	with open('temp.jpg', 'wb') as file:
		download = requests.get(image_url, headers=headers)
		file.write(download.content)
	image = cv2.imread('temp.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	detected_face = face_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
	x = detected_face[0][0]
	y = detected_face[0][1]
	l = detected_face[0][2]
	w = detected_face[0][3]
	image = image[y:y+l, x:x+w]
	image = cv2.resize(image, (image_dim, image_dim))
	image = np.float32(image)
	pred = model.predict([[image]]).tolist()
	return json.dumps({'prediction': pred})
