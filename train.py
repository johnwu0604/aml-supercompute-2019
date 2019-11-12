import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
import argparse

# construct the argument parse and parse the arguments
argparse = argparse.ArgumentParser()
argparse.add_argument("-d", "--data_dir", required=True,
	help="path to input dataset (i.e., directory of images)")
argparse.add_argument("-m", "--model_name", required=True,
	help="path to output model")
argparse.add_argument("-l", "--label_bin", required=True,
	help="path to output label binarizer")
args = vars(argparse.parse_args())

data_dir = args['data_dir']
model_name = args['model_name']
label_bin = args['label_bin']

classes = os.listdir(data_dir)
num_epochs = 25
learning_rate = 1e-3
batch_size = 32
image_dim = (256, 256, 3)

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(data_dir)))
random.seed(42)
random.shuffle(image_paths)

# loop over the input images
for image_path in image_paths:
    try:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_dim[1], image_dim[0]))
        image = img_to_array(image)
        data.append(image)
        
        # extract the class label from the image path and update the labels list
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)
    except:
        print('Discarding image {} due to improper format'.format(image_path))

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, 
                         width_shift_range=0.1, 
                         height_shift_range=0.1, 
                         shear_range=0.2, 
                         zoom_range=0.2,
                         horizontal_flip=True, 
                         fill_mode='nearest')

res_net = ResNet50(weights='imagenet', include_top=False, input_shape=image_dim)
#res_net.trainable = False

global_average_layer = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(classes), activation='softmax')
model = Sequential([
  res_net,
  global_average_layer,
  output_layer
])

optimizer = Adam(lr=learning_rate, decay=learning_rate/num_epochs)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(aug.flow(train_x, train_y, batch_size=batch_size), 
                    steps_per_epoch=len(train_x) // batch_size,
                    epochs=num_epochs,
                    validation_steps=10,
                    verbose=1)

model.save(model_name)

f = open(label_bin, "wb")
f.write(pickle.dumps(lb))
f.close()
