import csv
import cv2
import numpy as np
from PIL import Image
lines =[]

car_images=[]
steering_angles=[]

with open('data/driving_log.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.5 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            directory = "..." # fill in the path to your training IMG directory
            img_center = np.asarray(Image.open(row[0]))
            img_left = np.asarray(Image.open(row[1]))
            img_right = np.asarray(Image.open(row[2]))

            # add images and angles to data set
            car_images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])

            
X_train = np.array(car_images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(90,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,6,6,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch = 2)
model.save('model.h5')


        
