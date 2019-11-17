import os
import argparse
import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from azureml.core.run import Run
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Get the Azure ML run object
run = Run.get_context()

class AmlLogger(Callback):
    ''' A callback class for logging metrics using Azure Machine Learning Python SDK '''

    def on_epoch_end(self, epoch, logs={}):
        run.log('val_accuracy', float(logs.get('val_accuracy')))

    def on_batch_end(self, batch, logs={}):
        run.log('accuracy', float(logs.get('accuracy')))


# Define arguments for training
parser = argparse.ArgumentParser(description='Famous athlete classifier')
parser.add_argument('--data_dir', type=str, default='data', help='Root directory of the data')
parser.add_argument('--image_dim', type=int, default=250, help='Image dimensions')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learnign rate of the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--steps_per_epoch', type=int, default=100, help='Training steps per epoch')
parser.add_argument('--num_epochs', type=int, default=25, help='Training number of epochs')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Training dropout rate on output layer')
parser.add_argument('--activation_function', type=str, default='softmax', help='Training activation function')
parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
args = parser.parse_args()

# Get arguments from parser
data_dir = args.data_dir
image_dim = args.image_dim
batch_size = args.batch_size 
steps_per_epoch = args.steps_per_epoch 
num_epochs = args.num_epochs 
dropout_rate = args.dropout_rate 
activation_function = args.activation_function 
output_dir = args.output_dir 

# Horovod: Initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Define the label classes
classes = ['Lebron_James', 'Stephen_Curry']

# Create data generator to augmnent input images
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             rotation_range=90,
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             shear_range=0.2, 
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=False)

# Create train dataset with generator
train_generator = datagen.flow_from_directory(os.path.join(data_dir, 'train'),
                                              target_size=(image_dim, image_dim),
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              classes=classes)

# Create valid dataset with generator
valid_generator = datagen.flow_from_directory(os.path.join(data_dir, 'valid'),
                                              target_size=(image_dim, image_dim),
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              classes=classes)

# Download base ResNet50 model and set weights to not trainable
base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(image_dim, image_dim, 3))
base_model.trainable = False

# Add pooling, dropout, and classification layers to base model
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(dropout_rate),
    layers.Dense(len(classes), activation=activation_function)
])

# Horovod: add Horovod DistributedOptimizer.
optimizer = Adam(learning_rate=learning_rate*hvd.size(), epsilon=1e-08, clipnorm=1.0)
optimizer = hvd.DistributedOptimizer(optimizer)

# Compile model with optimizer, loss function, and metrics
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'], experimental_run_tf_function=False)

# Horovod: add callbacks
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1)
]

# Horovod: metrics only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(AmlLogger())

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# Train the model
model.fit_generator(train_generator, 
                    steps_per_epoch=steps_per_epoch // hvd.size(),
                    epochs=num_epochs,
                    validation_data=valid_generator, 
                    validation_steps=10,
                    callbacks=callbacks,
                    verbose=verbose)

# Save the output model
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save(os.path.join(output_dir, 'model.h5'))

