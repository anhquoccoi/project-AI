import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os

import keras.utils

from keras.models import Sequential
from keras.utils.data_utils import Sequence
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau, History
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Lambda, MaxPooling2D, ZeroPadding2D
from keras.models import Input, Model
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
from keras.regularizers import l2
from keras.models import load_model
# TEST VIDEO
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

model = load_model('CuoiKy.h5')
def detect_points(face_img):
    me  = np.array(face_img)/255
    x_test = np.expand_dims(me, axis=0)
    x_test = np.expand_dims(x_test, axis=3)

    y_test = model.predict(x_test)
    label_points = (np.squeeze(y_test)*48)+48 
    
    return label_points
    
# Load haarcascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
dimensions = (96, 96)

# Enter the path to your test image
cap = cv.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    if not ret:
        print('unavailable')
    # cv.imwrite('data.jpg',frame)
    # img = cv.imread('data.jpg')

    default_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    #faces = face_cascade.detectMultiScale(gray_img, 4, 6)

    faces_img = np.copy(gray_img)

    # plt.rcParams["axes.grid"] = False


    all_x_cords = []
    all_y_cords = []

    for i, (x,y,w,h) in enumerate(faces):
        
        h += 10
        w += 10
        x -= 5
        y -= 5
        
        just_face = cv.resize(gray_img[y:y+h,x:x+w], dimensions)
        cv.rectangle(faces_img,(x,y),(x+w,y+h),(255,0,0),1)

        scale_val_x = w/96
        scale_val_y = h/96
        
        label_point = detect_points(just_face)
        all_x_cords.append((label_point[::2]*scale_val_x)+x)
        all_y_cords.append((label_point[1::2]*scale_val_y)+y)

        
        
#        cv.rectangle(default_img,(int(label_point[4]*scale_val_x+x),int(label_point[5]*scale_val_y+y)),(int((label_point[4]+16)*scale_val_x+x),int((label_point[5]+10)*scale_val_y+y)),(0,255,255),2)
#        cv.putText(default_img,'mat phai',(int(label_point[4]*scale_val_x+x-10),int(label_point[5]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.rectangle(default_img,(int(label_point[10]*scale_val_x+x),int(label_point[11]*scale_val_y+y)),(int((label_point[10]+16)*scale_val_x+x),int((label_point[11]+10)*scale_val_y+y)),(0,255,255),2)
#        cv.putText(default_img,'mat trai',(int(label_point[10]*scale_val_x+x-10),int(label_point[11]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.circle(default_img,(int(label_point[20]*scale_val_x+x),int(label_point[21]*scale_val_y+y)),20,(200,100,50),2)
#        cv.putText(default_img,'*',(int(label_point[20]*scale_val_x+x-10),int(label_point[21]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(200,100,50),1)
#        cv.rectangle(default_img,(int(label_point[24]*scale_val_x+x),int(label_point[25]*scale_val_y+y)),(int((label_point[24]+40)*scale_val_x+x),int((label_point[25]+10)*scale_val_y+y)),(209,206,0),2)
#        cv.putText(default_img,'mieng',(int(label_point[24]*scale_val_x+x-10),int(label_point[25]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(209,206,0),1)
#        cv.putText(default_img,'*',(int(label_point[4]*scale_val_x+x-10),int(label_point[5]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'*',(int(label_point[10]*scale_val_x+x-10),int(label_point[11]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'*',(int(label_point[20]*scale_val_x+x-10),int(label_point[21]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(200,100,50),1)
#        cv.putText(default_img,'*',(int(label_point[24]*scale_val_x+x-10),int(label_point[25]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(209,206,0),1)
#        cv.putText(default_img,'*',(int(label_point[0]*scale_val_x+x),int(label_point[0]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'*',(int(label_point[0]*scale_val_x+x-10),int(label_point[0]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'*',(int(label_point[0]*scale_val_x+x-10),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        count = 0
#        for m in range(0,2):
#            for n in range(0,2):
#               cv.putText(default_img,'*',(int(label_point[m]*scale_val_x+x-10),int(label_point[n]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#            count += 1
        cv.putText(default_img,'.',(int(label_point[0]*scale_val_x+x),int(label_point[1]*scale_val_y+y+5)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[1]*scale_val_x+x-10),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[0]*scale_val_x+x-20),int(label_point[2]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'6',(int(label_point[0]*scale_val_x+x-10),int(label_point[4]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'7',(int(label_point[1]*scale_val_x+x-10),int(label_point[2]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'4',(int(label_point[1]*scale_val_x+x-10),int(label_point[4]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[2]*scale_val_x+x-10),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[2]*scale_val_x+x-30),int(label_point[2]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'7',(int(label_point[3]*scale_val_x+x-10),int(label_point[0]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'14',(int(label_point[3]*scale_val_x+x-10),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[3]*scale_val_x+x),int(label_point[2]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'16',(int(label_point[3]*scale_val_x+x-10),int(label_point[3]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'17',(int(label_point[3]*scale_val_x+x-10),int(label_point[4]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'18',(int(label_point[3]*scale_val_x+x-10),int(label_point[5]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x-30),int(label_point[0]*scale_val_y+y-5)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x-3),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x+50),int(label_point[2]*scale_val_y+y+5)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
#        cv.putText(default_img,'12',(int(label_point[4]*scale_val_x+x-5),int(label_point[4]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x-30),int(label_point[0]*scale_val_y+y+30)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x-30),int(label_point[0]*scale_val_y+y+50)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[5]*scale_val_x+x-10),int(label_point[4]*scale_val_y+y+60)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'1',(int(label_point[5]*scale_val_x+x+50),int(label_point[4]*scale_val_y+y+60)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[5]*scale_val_x+x-55),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv.putText(default_img,'.',(int(label_point[6]*scale_val_x+x+10),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
    cv.imshow('RealTime Test',default_img)
    if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
