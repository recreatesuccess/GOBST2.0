# Importing the Keras libraries and packages
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
import time
from keras.models import load_model

def recall(y_true, y_pred):
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
     recall = true_positives / (possible_positives + K.epsilon())
     return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', precision, recall])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
time_callback = TimeHistory()

classifier.fit_generator(training_set,
                         steps_per_epoch = 500,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 250, callbacks = [time_callback])



classifier.save('/Users/chukaezema/Desktop/GOBST Image Classifier/my_model.h5') 

# Growing the Neural Network

# Initializing the bigger network
classifier1 = Sequential()

# Getting the weights and bias from the previous network
weights = classifier.layers[0].get_weights()[0]
biases = classifier.layers[0].get_weights()[1]

# Settig the weights for the bigger network
classifier1.set_weights(weights)

# Convolution
classifier1.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier1.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier1.add(Flatten())

# Full connection
classifier1.add(Dense(units = 128, activation = 'relu'))
classifier1.add(Dense(units = 1, activation = 'sigmoid'))

classifier1.summary()


# Compiling the CNN
classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
classifier1.fit_generator(training_set,
                         steps_per_epoch = 500,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 250)

# Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/generated_image.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier1.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Good'
else:
    prediction = 'Bad'