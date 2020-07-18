# **Behavioral Cloning** 

Overview
---
This repository contains the Behavioral Cloning Project.

In this project, I used deep neural networks and convolutional neural networks to clone driving behavior.
I used image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
here i used nvidia architecture for end to end learning for self driving car
![image](https://i.ibb.co/7NdXLSD/vehicle.png)
#### 2. Attempts to reduce overfitting in the model



The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually, also I used MSE for calculating the error.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road , and finally I flipped the images to generalize my model

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

####  Solution Design Approach

first, I started with reading the data from CSV file and split the data into training 80% and 20% for validation set.

the next step was designing my model , first I started with normalizing and cropping the images
then , I used 5 convolutional layers, flatten layer and 3 fully connected layers and the output is the vehicle control.

then, within the generator I used flipping Images and I used the Images from right and left cameras to augmunt the data and generalize the model

finally I shuffled my data , using MSE to calculate the error and adam as an optimizer

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3

after running the model I got very good results and the car was within the track
check the video of the results : run1.mp4




