import os
import time
import sys
import queue
import signal
import threading
from subprocess import call
import multiprocessing as mp
from os.path import dirname, join

import cv2
import torch.hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from torchvision import transforms

import Emotion_detection.notebook as emo
import Drowsiness.model as model
from Drowsiness.grad_cam import BackPropagation

# current_dir = os.path.dirname(os.path.realpath('__file__'))
current_dir = '/temp/Drowsiness'
# file = 'alarm.wav' # Alarm sound file

flagd = 0
flage = 0

# Sound player start
timebasedrow= time.time()
timebasedis= time.time()
timerundrow= time.time()
timerundis= time.time()
face_cascade = cv2.CascadeClassifier(current_dir+'/haar_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(current_dir+'/haar_models/haarcascade_eye.xml')
MyModel="BlinkModel.t7"

shape = (24,24)
classes = [
    'Close',
    'Open',
]

eyess=[]
cface=0

ppid = 0  # main module pid

def preprocess(image_path):
    global cface
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path['path'])    
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        ...
    else:
        cface=1
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        roi_color = image[y:y+h, x:x+w]
        """
        Depending on the quality of your camera, this number can vary 
        between 10 and 40, since this is the "sensitivity" to detect the eyes.
        """
        sensi=20
        eyes = eye_cascade.detectMultiScale(face,1.3, sensi) 
        i=0
        for (ex,ey,ew,eh) in eyes:
            (x, y, w, h) = eyes[i]
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye = face[y:y + h, x:x + w]
            eye = cv2.resize(eye, shape)
            eyess.append([transform_test(Image.fromarray(eye).convert('L')), eye, cv2.resize(face, (48,48))])
            i=i+1
    cv2.imwrite(current_dir+'/temp-images/display.jpg',image) 
    

def eye_status(image, name, net):
    img = torch.stack([image[name]])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]

    #print(name,classes[actual_status.data], probs.data[:,0] * 100)
    return classes[actual_status.data]

def func(imag,modl):
    drow(images=[{'path': imag, 'eye': (0,0,0,0)}],model_name=modl)

def drow(images, model_name):
    global flagd
    global flage
    global ppid
    global eyess
    global cface
    global timebasedrow
    global timebasedis
    global timerundrow
    global timerundis
    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join(current_dir+'/model', model_name), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    
    flag =1
    status=""
    for i, image in enumerate(images):
        if(flag):
            preprocess(image)
            flag=0
        if cface==0:
            timebasedrow= time.time()
            timebasedis= time.time()
            timerundrow= time.time()
            timerundis= time.time()
        elif(len(eyess)!=0):
            eye, eye_raw , face = eyess[i]
            image['eye'] = eye
            image['raw'] = eye_raw
            image['face'] = face
            timebasedrow= time.time()
            flagd = 0
            for index, image in enumerate(images):
                status = eye_status(image, 'eye', net)
                if(status =="Close"):
                    if flage == 0:
                        timebasedis = time.time()
                        flage = 1
                    elif((time.time() - timebasedis) > 1.5):
                        print('Distracted')
                        flage = 0
                else:
                    call('pkill -9 aplay &', shell=True)
        else:
            if flagd == 0:
                timebasedrow = time.time()
                flagd = 1
            elif((time.time() - timebasedrow) > 3):
                os.kill(ppid, signal.SIGUSR1)
                flagd = 0

def main():
    global ppid
    ppid = os.getppid()  # get main module pid
    global eyess
    global cface
    cap = cv2.VideoCapture(0)  # the camera object must NOT be defined at global scope
    q = mp.Queue()
    c = mp.Process(target=emo.main, args=[q, ppid])
    c.start()
    while True:
        eyess=[]
        cface=0
        ret, img = cap.read()
        try:
            q.put(img, block=False)
        except queue.Full:
            print('queue is full')
        cv2.imwrite(current_dir+'/temp-images/img.jpg',img) 
        func(current_dir+'/temp-images/img.jpg',MyModel)

''' 
def disp():
    while 1:
        try:
            img = cv2.imread(current_dir+'/temp-images/display.jpg')
            # cv2.imshow(current_dir+'/temp-images/image',img)
            k = cv2.waitKey(30) & 0xff
        except:
            ...
'''
