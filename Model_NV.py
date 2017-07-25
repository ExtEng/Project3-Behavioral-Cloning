import csv
import cv2
import numpy as np
from PIL import Image

## Iniitialize variables
lines =[]
car_images=[]
steering_angles=[]

## Read the csv file and store variables and images

with open('data_m/driving_log.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.4 # Experimented with [0.2, 0.3, 0.4, 0.5]
            
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
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
##Normalize the images 
model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,3)))

##Crop 70 pixels from the top and 25 pixels from the bottom
model.add(Cropping2D(cropping=((70,25), (0,0))))

## First Convolution Layer - 24 filters - (5x5) and stride (2x2)
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
## 2nd Convolution Layer - 36 filters - (5x5) and stride (2x2)
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
## 3rd Convolution Layer - 48 filters 
model.add(Convolution2D(48,5,5,activation="relu"))
## 4th Convolution Layer - 64 filters 
model.add(Convolution2D(64,5,5,activation="relu"))

## Flatten to a layer
model.add(Flatten())
## 1st Fully connected layer - 100 outputs
model.add(Dense(100))
## 2nd Fully connected layer - 50 outputs
model.add(Dense(50))

## 3rd Fully connected layer - 10 outputs 
model.add(Dense(10))

## Output Steering angle
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch = 2)
model.save('model_nV_st4.h5')


        
