from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Lambda,Cropping2D,Convolution2D

import csv
import os
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

    
samples=[]
with open('./data/driving_log.csv') as csvreader:
    reader=csv.reader(csvreader)
    next(reader, None)
    for line in reader:
        samples.append(line)
    
        
    
        
      


train_samples,validation_samples= train_test_split(samples,test_size=0.2)
#print(len(validation_samples))


#this architecture is designed by nvidia team
model=Sequential()
#normalize
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))
#cropping only the part of the street for better results
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))

    
#generators are very good in dealing with large amount of data,using a generator you can pull pieces of the data and process them on the fly only when you need them, which is much more memory-efficient
def generator(samples,batch_size=32):
    no_samples=len(samples)
    while 1:
        shuffle(samples)
        for offset in range (0,no_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            images=[]
            steering=[]
            for batch_sample in batch_samples:
                
                
                
                #use the images from the 3 cameras will help us to make our model generalized
                #center_name= './data/IMG/'+batch_sample[0].split('/')[-1]
                
                
                
                right_name='./data/IMG/'+batch_sample[2].split('/')[-1]
                left_name='./data/IMG/'+batch_sample[1].split('/')[-1]
                
                center=cv2.imread('./data/IMG/'+batch_sample[0].split('/')[-1])    
                center_image=cv2.cvtColor(center, cv2.COLOR_BGR2RGB) 
                
                center_angle=float(batch_sample[3])
                images.append(center_image)
                steering.append(center_angle)
                #flip the data will help us to augmunt the data , and dealing with curves which are on the opposite side of the road
                images.append(cv2.flip(center_image, 1))
                steering.append(center_angle*-1)
                 
                right_image=cv2.cvtColor(cv2.imread(right_name),cv2.COLOR_BGR2RGB)
                right_angle=center_angle-0.2
                images.append(right_image)
                steering.append(right_angle)
                images.append(cv2.flip(right_image, 1))
                steering.append(right_angle*-1)
                
                left_image=cv2.cvtColor(cv2.imread(left_name),cv2.COLOR_BGR2RGB)
                left_angle=center_angle+0.2
                images.append(left_image)
                steering.append(left_angle)
                images.append(cv2.flip(left_image, 1))
                steering.append(left_angle*-1)
                
                
            X_train=np.array(images)
            y_train=np.array(steering)
            yield sklearn.utils.shuffle(X_train,y_train)

train_generator=generator(train_samples)


validation_generator=generator(validation_samples)               
          

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/32, epochs=3, validation_data=validation_generator, validation_steps=len(validation_samples)/32, verbose=1)
model.save('model.h5')            
                
                
        
    
        
        
        
    
    
    
    
             