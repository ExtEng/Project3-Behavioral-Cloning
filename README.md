# **Behavioral Cloning Project**
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images_Report/center_2017_07_15_19_00_49_825.jpg "Track 1 - Center"
[image2]: ./Images_Report/left_2017_07_15_19_00_49_825.jpg "Track 1 - Left"
[image3]: ./Images_Report/right_2017_07_15_19_00_49_825.jpg "Track 1 - Right"
[image4]: ./Images_Report/center_2017_07_16_13_36_49_825.jpg "Track 2 - Center"
[image5]: ./Images_Report/left_2017_07_16_13_36_49_825.jpg "Track 2 - Left"
[image6]: ./Images_Report/right_2017_07_16_13_36_49_825.jpg "Track 2 - Right"


###  1. Files Submitted

My project includes the following files:
* model_NV.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (modified)
* model_NV_st4.h5 containing a trained convolution neural network 
* run2.mp4 is a video of a successful run though track 1. 
* Readme.md summarizing the results

### 2. Final Code 
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. This 
will run the "NVidia-lite" architecture with a steering coefficient of 0.4 degrees.  The drive script has been modified to change the desired car speed to 20.

```sh
python drive.py model_nV_st4.h5
```

### 3. Model Architecture and Training Strategy

#### Model Architecture
I had initial implemented LeNet model, which even after extensive training (flipped images, double tracks, multi-cameras. etc.) did not prove robust enough. So, I decided to implement NVidia- ish architecture, with some minor differences. Which could drive the car up to 
10 laps without failure on Track 1 and approximately 0.5 of Track 2. 
My Final model consists of the following stages:

1. Starts with Normalization about zero. i.e. {(x /255) - 0.5}, using the keras lamda layer
2. 2D Cropping - Remove parts of the images that we do not care about. 70 pixels from the top & 25 from the bottom
3. 1st Convolution layer - 24 filters (5x5), with (2x2) strides and a ReLU Activation  
4. 2nd Convolution layer - 36 filters (5x5), with (2x2) strides and a ReLU Activation
5. 3rd Convolution layer - 48 filters (5x5), with (2x2) strides and a ReLU Activation 
6. 4th Convolution layer - 64 filters (5x5), with (2x2) strides and a ReLU Activation
7. 1st Fully connected layer - Output - 100 with linear activation 
8. 2nd Fully connected layer - Output - 50 with linear activation
9. 3rd Fully connected layer - Output - 10 with linear activation
10. Final output with linear activation

#### Training 
Then model was trained with the Adam optimizer with a mean square error loss function. Since the Adam Optimizer was used, no learning parameters required tuning.
The model was trained with 80 percent of the captured images and validated the rest (20%), with the data shuffled. Generally, 2 Epochs was sufficient to achieve a low loss error.
After training the Final model with the data from track 1, the Car drove very smoothly through the track, which hinted at overfitting. To prevent memorization, I trained the model with data from both tracks, which added to the robustness.

The model was trained with 
* 2 laps - Track 1 Counter clockwise
* 2 laps - Track 1 Clockwise [1]
* 2 laps - Track 2 Counter clockwise
* 2 laps - Track 2 Clockwise

Note [1] - I preferred this to flipping the image as I did with the first model, because it added more variability in the driving aspect.

For the final data set - I tuned the Steering correction for the left and right camera images from 0.2 to 0.5. Selecting 0.4 as the stable configuration. The Steering correction factor was applied to left (+) and right images (-). 

Sample of Data with 0.4 Steering Correction applied 

Track |Steering Angle - Left | Steering Angle - Center | Steering Angle - Right | Throttle | Brake | Speed
------------ | ------------ | ------------- | ------------ | ------------- | ------------ | -------------
1 | 0.36566524 | -0.03433476 | -0.43433476 | 1 | 0 | 30.19081
2 | -0.2400675 | -0.6400675 | -0.83433476 | 0 | 0 | 28.63895

Corresponding Images from Track 1 

![alt text][image1]
![alt text][image2]
![alt text][image3]

Corresponding Images from Track 2

![alt text][image4]
![alt text][image5]
![alt text][image6]

Initially I had planned more data gathering, such as "recovery from edges", "smoother turns". However, during driving the scenarios stated above I covered a lot "types" of driving, and having achieved a stable model. I noted it for future improvements.


### 4. Model Architecture and Training Strategy

#### 1. Solution Design Approach

My First attempts using the LeNet Architecture, I trained the model with 1 lap of Track 1 (counter clockwise) with only the center normalized image. Which caused the car to veer to the left, which I attributed to my driving through bends (I hug the inside curves). To Prevent this, I trained the model with flipped images and gathered more data where I tried to stay in the center lane. As stated I augmented the data set with horizontally flipped images and reversed the steering angle accordingly.  This improved the driving, but still drove off course. 

To increase the robustness, I used the left and right cameras with 0.2 Steering correction. This improved the driving in the sense that the car stayed closer to the center of the lane. But still failed to stay on the road for a full lap. I played with this parameter up to 0.5, before moving on to a different architecture. 

I used the Nvidia Architecture, But I started with 3 convolution layers and then fully connected layers, and continued augmenting the architecture if it failed in the achieving its task.  Essentially train with multiple steering correction angles and test trough Track 1.  After multiple attempts I finally reached the current architecture listed above.

#### 2. Final Model Architecture

After implementing NVidia-ish architecture the car drove very smooth through the course but had some bugs, occasional it would veer left.  At this point I decided to scrap my old data and train the model with equally balanced data, of 2 lap driving in both directions of the track (total 4 laps). I took out the flipped images and settled for reverse track driving data.  After training the model for 5 epochs with the new data (all 3 images) and steering correction factor of 0.2, the car drove flawlessly. I dropped the Epochs to 2, because after looking at the logs, there seem to have no gain and possibly overfitting. To prevent memorization of the track I trained the model on track 2 data, which looked to have increased the robustness. I ran the car simulator for multiple runs, for approximately up to 10 laps. 

Track 2 data followed the same convention 2 laps per direction for a total of 4 laps. After testing the car model thoroughly, I decided to try and break it by running the car at higher speeds, the max speed I achieved was 20 with a steering correction factor of 4. At higher speeds, the models slightly unstable behaviour (in the steering angle). The car steering essentially oscillates. In the Robotics approach, I would increase my Derivative gain in a PID controller. Which makes me wonder how I can augment the Neural network's steering angle with a PD controller, with the Proportional gain (P) = 0. 

After, completing a couple of runs with the speed at 20, I tried to increase it to 30, which failed to stay on the course. 

My Final submission includes a video of a successful run (2 laps) at 20 labeled run2.mp4

