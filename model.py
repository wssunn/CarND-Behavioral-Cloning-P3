import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Cropping2D
from keras.layers import BatchNormalization, Lambda

# this function read image_path and return the corresponding image
def read_image(source_path, dataset):
    filename = os.path.split(source_path)[-1]
    image_path = dataset + '/IMG/' + filename
    return image_path

# this function argument the training sets by flipping the image
# so that more data is generated
def process_path(images_path, measurements):
    augmented_images, augmented_measurements = [], []

    for path, measurement in zip(images_path, measurements):
        image = cv2.imread(path)
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement*(-1.0))
    return np.array(augmented_images), np.array(augmented_measurements)

# this function generates pictures, load the pictures on the go
# so that in the training process there is no need to read the entire dataset into memory
def imageLoader(images_path, measurements, batch_size):
    L = len(images_path)

    #this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X,Y = process_path(images_path[batch_start:limit], measurements[batch_start:limit])
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size

# obtain datasets store in folder dataset
datasets = glob.glob('./dataset/*')
images_path = []; measurements = []

# load image_path and measurement, note image itself is not laoded, just the path
# the simulation captures left, centre, right image of the car
# together with a correction factor, it helps the car to stay in the centre
for dataset in datasets:
    csv_path = dataset + '/driving_log.csv'

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            correction = 0.5
            steering_centre = float(line[3])
            steering_left = steering_centre + correction
            steering_right = steering_centre - correction

            center_image = read_image(line[0], dataset)
            left_image = read_image(line[1], dataset)
            right_image = read_image(line[2], dataset)

            images_path.append(center_image); measurements.append(steering_centre)
            images_path.append(left_image); measurements.append(steering_left)
            images_path.append(right_image); measurements.append(steering_right)

# shuffle the data and split them into training and validation dataset
images_path, measurements = shuffle(images_path, measurements)
print('number of images: ', len(images_path))
x_train, x_val, y_train, y_val = train_test_split(images_path, measurements,
                                                test_size=0.2, random_state=0)
training = True # this swtich is used so that during the actual running dropout is no used

def simple_model():
    if training == True:
        drop = 0.25
    else:
        drop = 0
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(BatchNormalization(axis=3)) # data format: 'channel_last' (None, 90, 320, 3)
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(drop))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(drop))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(drop))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(drop))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(drop))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(1))
    return model

epochs = 4
batch_size = 256
train_epoch = np.ceil(len(x_train)/batch_size).astype(int)
val_epoch = np.ceil(len(x_val)/batch_size).astype(int)

train_generator = imageLoader(x_train, y_train, batch_size)
val_generator = imageLoader(x_val, y_val, batch_size)

model = simple_model()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history_object = model.fit_generator(train_generator,steps_per_epoch=train_epoch,
                                     epochs=epochs, verbose=1,
                     validation_data=val_generator, validation_steps=val_epoch)
model.summary()

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
