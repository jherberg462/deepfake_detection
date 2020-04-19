#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# #install dependencies 
# ! pip install --upgrade pip


# In[ ]:


# !pip install numpy --upgrade
# ! pip install pandas --upgrade
# ! pip install boto3 --upgrade
# ! pip install requests --upgrade
# ! pip install scikit-learn --upgrade
# ! pip install tensorflow --upgrade
# ! pip install keras --upgrade
# ! pip install scikit-video --upgrade
# ! pip install scikit-image --upgrade
# !pip install sagemaker --upgrade
# ! pip install opencv-python --upgrade


# In[1]:


import pandas as pd
import numpy as np
import boto3
import cv2 as cv
import os
import time
import requests
import random 
import json
from joblib import dump, load
import math
# import skvideo.io as sk - removed
from sklearn.model_selection import train_test_split
# from skimage.transform import resize - removed - this is very slow
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation
from keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model

#sensitive variables in config.py file that is on .gitignore
from config import key_, secret_, s3_bucket, kaggle_cookie


# In[ ]:


#explore meta.json file
with open('meta.json') as m:
    meta = json.load(m)
video_and_labels = {}
video_label_only = {}
for video in meta:
    video_and_labels[video] = meta[video]
    video_label_only[video] = meta[video]['label']


# In[ ]:


#list videos by if they are real or fake
real_videos = []
fake_videos = []
for video in meta:
    if meta[video]['label'] == 'REAL':
        real_videos.append(video)
    else:
        fake_videos.append(video)
video_set = []
for video in range(len(real_videos)):
    video_set.append(real_videos[video])
    video_set.append(fake_videos[video])
#there are about 4x as many fake videos as there are real videos
#the above will make the training set more balanced


# In[ ]:


random.shuffle(video_set)


# In[ ]:


#load csv files with videos that exist in S3 bucket
video_df = pd.read_csv('video_information.csv')
video_list = video_df['video_names'].to_list()

#there are a small number of videos referenced in the meta.json file that do not appear in my bucket
#check if any of the videos in the video_set list (that our model will train on) are not in the list of videos that are in my bucket and remove if not
for video in video_set:
    if video not in video_list:
        video_set.remove(video)


# In[ ]:


#split dataset into training and testing sets
_, _, train_videos, test_videos = train_test_split(video_set, video_set, test_size=.1, random_state=3)
#the train and test_videos are the video names that will be passed into functions that download and preprocess 
#the data and pass preprocessed data into the model. The functions also look up the y values


# In[ ]:


training_len = len(train_videos)
testing_len = len(test_videos)
print(f'training length: {len(train_videos)}')
print(f'testing length: {len(test_videos)}')


# In[ ]:


#may want to consider changing the array to a video file name/path and incorporate opeing the video in this function
def preprocess_video(video_array, max_size=150, video=None):
    '''
    takes a video array as an input, and looks at every 5th frame (strating from the 3rd), the function will
    return an array of the difference between the frame in question and the 1st and 2nd frame back and forward
    function will crop the video into a box max_size by max_size pixels, reading a random part of each 5th frame
    (the difference between each frame will look at the same location )
    '''
    frame_list = []
    #find how many groups of frames need to be looked at
    num_frames_div_5 = math.floor(len(video_array)/5)
    num_rounds = num_frames_div_5
    round_num = 0
    try:
        num_frames, x_pixel, y_pixel, _ = video_array.shape
        #sometimes download_video_from_s3_bucket will return an array of arrays
        #this is caused when open CV cannot read some frames -- when this happens, the last frames will be 'none'
    except Exception as e:
        print(f'preprocess exception {e} on video: {video}')
        num_frames, = video_array.shape
        x_pixel, y_pixel, _ = video_array[0].shape
    #get the number of pixels we can possibly shift the starting point of the 'first' x and y pixel
    x_extra, y_extra = x_pixel - max_size +1, y_pixel - max_size + 1 
    #add 1 to above due to how np.random.randint excludes the max number passed as a possible output

    for x in np.arange(0, len(video_array)):            
        if x % 5 == 2:
            if round_num < num_rounds:
                #find out how much we will shift the starting x value for the first pixel
                x_shift = np.random.randint(low=x_extra, size=1)[0]
                #if 'high' param is not passed, the 'low' value will serve as the 'high' parameter
                #faster speed if size parameter is passed
                y_shift = np.random.randint(low=y_extra, size=1)[0]
                #get the last pixel on the x axis we will look at
                x_end = max_size + x_shift
                y_end = max_size + y_shift
                #current frame, max_size by max_size pixels, starting at x_shift, y_shift
                frame_sized = video_array[x][ x_shift:x_end, y_shift:y_end,]
                #look at frame in same location as above, 2 frames back
                #todo - update variable names to reflect what they actually are
                #the download from s3 bucket function skipped frames we wont look at, so only need to look back 1 index
                back_3 = video_array[x-1][ x_shift:x_end, y_shift:y_end,]
                #this is actually 4 frames back
                back_5 = video_array[x-2][ x_shift:x_end, y_shift:y_end,]
                try:
                    forward_3 = video_array[x+1][ x_shift:x_end, y_shift:y_end,]
                except:
                    break
                try:
                    forward_5 = video_array[x+2][ x_shift:x_end, y_shift:y_end,]
                except:
                    break
                #get absolute values of the difference between the current frame and the frame 3 frames back
                minus_3 = np.array(abs(frame_sized - back_3))
                minus_5 = np.array(abs(frame_sized - back_5))
                plus_3 = np.array(abs(frame_sized - forward_3))
                plus_5 = np.array(abs(frame_sized - forward_5))
                plus_3 = minus_3
                plus_5 = minus_3
                frame_list.append([minus_3, minus_5, plus_3, plus_5])#, frame_sized])
                round_num += 1
    frame_list = np.array(frame_list)
    #reshape
    ndims = frame_list.shape[1] * frame_list.shape[2] * frame_list.shape[3] * frame_list.shape[4]
    frame_list_ = frame_list.reshape(frame_list.shape[0], ndims)
    return frame_list_


# In[ ]:


#consider returning a list of arrays, eg process x number of videos at a time
def download_video_from_s3_bucket(video_name, aws_key=key_, aws_secret=secret_, bucket=s3_bucket):
    '''
    ##Intended for use when not using Sagemaker##
    takes a video name as input, and returns a downloaded video from s3 bucket in an array
    '''
    s3 = boto3.client('s3',
                      aws_access_key_id=aws_key, 
                      aws_secret_access_key=aws_secret,
                      region_name='us-east-2', #region is hardcoded - this is not a security risk to keep public
                      config= boto3.session.Config(signature_version='s3v4')) #the sig version needs to be s3v4 or the url will error
    video_url = s3.generate_presigned_url('get_object',
                                        Params={"Bucket": bucket,
                                               'Key': video_name},
                                        ExpiresIn=6000)
    video = cv.VideoCapture(video_url)
    #get number of frames
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    #find how many times to iteriate through the below for loop
    #the loop will skip 5 frames, look at the nest 15, then skip 15
    frame_groups = math.floor((frame_count/30) + .5)
    frame_list = []
    for frame in np.arange(0, frame_groups):
        #skip the first 5 frames
        for skip in np.arange(0, 5):
            _ = video.grab()
        #next 10 frames -- save every other frame
        for look_at in np.arange(1, 11):
            _ = video.grab()
            if look_at %2 == 1:
                _, frame_array = video.retrieve()
                frame_list.append(frame_array)
        #skip the next 15 frames
        for skip in np.arange(0, 15):
            _ = video.grab()
        
    video.release()
    frame_array = np.array(frame_list)
    return frame_array


# In[ ]:


def get_video(video, computer=True):
    '''
    takes a video name, and if you are using a computer as input
    calls appropiate function to download video from s3 bucket, depending if you are using a computer or sagemaker
    '''
    if computer==True:
        response = download_video_from_s3_bucket(video)
    #todo - create function to obtain video via sagemaker notebook instance
    #once created, call function below
    else:
        response = 0
    
    x_values = preprocess_video(response)
    y_value = meta[video]['label']
    y_values = []
    for frame in np.arange(0, len(x_values)):
        if y_value == 'FAKE':
            y_values.append(0)
        else:
            y_values.append(1)
    y_values_ = to_categorical(y_values, num_classes=2)
    return x_values, y_values_


# In[ ]:


def generator(video_dictionary, batch_size=1, train=True):
    '''
    takes a dictionary or list of video names, and returns the output from get_video function for one video at a time
    if train is set to false, the list will be randomized initially
    '''
    count = 0
    video_list = []
    for video in video_dictionary:
        video_list.append(video)
    #split dataset into training and testing sets
    _, _, y_train, y_test = train_test_split(video_list, video_list, test_size=.1, random_state=55)
    #if not training, set the video list to the test set, otherwise set it to the training set
    if train == False:
        video_list_ = y_test
    else:
        video_list_ = y_train
    random.shuffle(video_list_)
    while True:
#         x_batch = np.empty(0)
#         y_batch = np.empty(0)
        for x in np.arange(0, batch_size):
            if count == len(video_list_):
                count = 0
                random.shuffle(video_list_)
            x, y = get_video(video_list_[count])
#         yield x_batch, y_batch
        yield x, y


# In[ ]:


class Generator(Sequence): #Generator is capatalized 
    # Class that will allow multiprocessing
    def __init__(self, video_list, y_set=None, batch_size=1):
        #convert the video_list to an array
        self.x, self.y = np.array(video_list), y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.floor(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        #currently only accepts a batch size of 1, update "idx" to "inds" once can accept larger batch size
        batch_x, batch_y = get_video(video_list[idx]) #look into improving get_video, such that it can accept a list
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# You will need to download the model from the following link:
# https://drive.google.com/file/d/19W55lH1Vp5YNOlr_B_OWog37DEsjdh8n/view?usp=sharing

# In[ ]:


#if model exists locally, load it, otherwise create model
try:
    model = load_model("deepfake_model_compare_frames.h5")
except:
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=270000)) #input_dim=x.shape[1] <- hard code the input_dim #1190700
    model.add(Dense(100, activation='relu'))
    model.add(Activation('relu'))
    #output layer
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])


# In[ ]:


#batch size of 1 will call the various functions for one video at a time
batch_size = 1
num_epochs = 1

#create generators for training and testing sets
train_generator = Generator(train_videos, train_videos)
test_generator = Generator(test_videos, test_videos)


try:
    model.fit(x=train_generator, 
              validation_data=test_generator, 
              steps_per_epoch=training_len//batch_size,
              validation_steps=testing_len//batch_size,
              workers=2, 
              use_multiprocessing=True, 
              epochs=num_epochs)
    #save model upon successful completion of running model.fit
    model.save('deepfake_model_compare_frames.h5')
except Exception as e:
    print(e)
    #if there is an exception, want to automatically save the model 
    model.save('compare_frames_model_train_exception.h5') #uncomment in production
    #consider updating this to save to a json file
    print(model.to_json())

    
    


# In[ ]:


model.metrics_names


# In[ ]:


# x, _ = get_video('xpzfhhwkwb.mp4') # fake video
# model.predict(x)


# In[ ]:


# x, _ = get_video('xmkwsnuzyq.mp4')
# model.predict(x) #real video


# In[ ]:


model.summary()


# In[ ]:


model.to_json()


# In[ ]:




