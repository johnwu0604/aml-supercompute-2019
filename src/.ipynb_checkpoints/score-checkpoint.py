from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import json

def init():
	model_dir = 'outputs'
	global model 
	model = load_model(os.path.join(model_dir, 'model.h5'))

def run(raw_data):
	image_dim = 250
	image_url = json.loads(raw_data)['image_url']
	face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	image = cv2.imread(image_url)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	detected_face = face_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
	x = detected_face[0][0]
	y = detected_face[0][1]
	l = detected_face[0][2]
	w = detected_face[0][3]
	image = image[y:y+l, x:x+w]
	image = cv2.resize(image, (image_dim, image_dim))
	image = np.float32(image)
	pred = model.predict([[image]])
	print(pred)
	return pred


