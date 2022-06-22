from logging import captureWarnings
from tkinter import *
from tkinter.filedialog import Open, SaveAs
import numpy as np
import cv2 as cv
from keras.models import load_model
from keras.models import Input, Model
from matplotlib import pyplot as plt

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

class Main(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Component of Face")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open Camera", command=self.onOpenCam)
        fileMenu.add_command(label="Open Imgin", command=self.onImgin)
        fileMenu.add_command(label="Recognition", command=self.onRecognition)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
        #self.txt = Text(self)
        #self.txt.pack(fill=BOTH, expand=1)
        
  
    def onOpenCam(self):
        # Enter the path to your test image
        cap = cv.VideoCapture(0)
        while 1:
            ret, frame = cap.read()
            if not ret:
                print('unavailable')
            # cv.imwrite('data.jpg',frame)
            # img = cv.imread('data.jpg')

            default_img = frame
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
            
                
                cv.putText(default_img,'.',(int(label_point[0]*scale_val_x+x),int(label_point[1]*scale_val_y+y+5)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[1]*scale_val_x+x-10),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[0]*scale_val_x+x-20),int(label_point[2]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[2]*scale_val_x+x-10),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[2]*scale_val_x+x-30),int(label_point[2]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[3]*scale_val_x+x),int(label_point[2]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x-30),int(label_point[0]*scale_val_y+y-5)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x-3),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x+50),int(label_point[2]*scale_val_y+y+5)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x-30),int(label_point[0]*scale_val_y+y+30)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[4]*scale_val_x+x-30),int(label_point[0]*scale_val_y+y+50)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[5]*scale_val_x+x-10),int(label_point[4]*scale_val_y+y+60)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[5]*scale_val_x+x+60),int(label_point[4]*scale_val_y+y+60)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[5]*scale_val_x+x-55),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                cv.putText(default_img,'.',(int(label_point[6]*scale_val_x+x+10),int(label_point[1]*scale_val_y+y)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            cv.namedWindow('RealTime Test', cv.WINDOW_AUTOSIZE)
            cv.imshow('RealTime Test',default_img)
            k = cv.waitKey(1)
            if k%256 == 27:
               # ESC pressed
                print("Escape hit, closing...")
                break
            if cv.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                break
    def onImgin(self):
        cap = cv.VideoCapture(0)

        cv.namedWindow('IMGIN', cv.WINDOW_AUTOSIZE)

        img_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("failed to grab frame")
                break
            cv.imshow('IMGIN', frame)

            k = cv.waitKey(1)
            if k%256 == 27:
           # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
            # SPACE pressed
                img_name = "{}.jpg".format(img_counter)
                cv.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

        cap.release()

        cv.destroyAllWindows()    
    def onRecognition(self):
        img = cv.imread('0.jpg')
        default_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        #faces = face_cascade.detectMultiScale(gray_img, 4, 6)

        faces_img = np.copy(gray_img)

        plt.rcParams["axes.grid"] = False


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
   
   
            plt.imshow(just_face, cmap='gray')
            plt.plot(label_point[::2], label_point[1::2], 'ro', markersize=5)
            plt.show()
    
    
        plt.imshow(default_img)    
        plt.plot(all_x_cords, all_y_cords, 'wo',  markersize=3)
        plt.show()


root = Tk()
Main(root)
root.geometry("480x480+100+100")
root.mainloop()
