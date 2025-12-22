import cv2
import torch
import time
from math import sqrt
import numpy as np

import matplotlib.pyplot as plt
from util import detect_corners
from util import computeBrief
from util import briefMatch

time_threhold=4
area_threhold=0.6
frame_count_list=[]
frame_count=0
TTC_list=[]
delta_time=0

#BRIEF_RATIO = 0.8
def get_dis(s1,s2):
    return sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )


def formula_func(kp1,kp2,matches,frame,box):
    total_dis_1=0
    total_dis_2=0
    count=0

    st_1=kp1[matches[0].queryIdx].pt
    st_2=kp2[matches[0].trainIdx].pt

    print('num:',len(matches))
    kp_range=min(5,len(matches))
    for i in range(0,kp_range):
        total_dis_1+=get_dis(kp1[matches[i].queryIdx].pt , st_1)
        total_dis_2+=get_dis(kp2[matches[i].trainIdx].pt , st_2)
        count+=1


    len_1=total_dis_1/count
    len_2=total_dis_2/count

    s=len_1/len_2
    delta_t=delta_time

    TTC=round(delta_t/(s-1),3)
    filename = 'static_fea_dis.txt'

    with open (filename,'a') as file_object:
        if TTC>0 and TTC<50:
            file_object.write(str(TTC)+'\n') 

    '''
    filename = 'static_TTC.txt'
    with open (filename,'a') as file_object:
        #file_object.write(str(TTC)+'\n') 
        if s==1:
            file_object.write(str(-1)+'\n') 
        else:
            file_object.write(str(round(delta_t/(s-1),3))+'\n') 

    if s==1:
        cv2.putText(frame,box[-1]+'static TTC:+inf', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
    else:
        TTC=round(delta_t/(s-1),3)
        cv2.putText(frame,box[-1]+'TTC:'+str(TTC), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
    '''            


def save_plot(TTC):
    global frame_count
    global frame_count_list
    global TTC_list
    TTC_list.append(TTC)
    frame_count_list.append(frame_count)
    
    frame_count+=1

def decide_same2(box,pre_box):
    max_x=max(box[0],pre_box[0])
    max_y=max(box[1],pre_box[1])
    min_x=min(box[2],pre_box[2])
    min_y=min(box[3],pre_box[3])
    if min_y-max_y>0 and min_x-max_x>0:
        same_area=(min_y-max_y)*(min_x-max_x)
        ori_area=(box[2]-box[0])*(box[3]-box[1])
        if (round(same_area/ori_area,3) > area_threhold) and (box[-1]==pre_box[-1]):
            return True
        else:
            return False
    else:
        return False


def danger(frame,box,TTC):
    cv2.putText(frame,box[-1]+' TTC:'+str(TTC), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

def decide_same(a,b,c,d):
    if a+b+c+d<40:
        return True
    else:
        return False

def get_result(frame,pre_frame,box,pre_box,pre_cal_time):
    global delta_time

    pre_img_box=pre_frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    test_img=pre_img_box.copy()

    img_box=frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]

    img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
    pre_img_box = cv2.cvtColor(pre_img_box, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_box, None)
    kp2, des2 = sift.detectAndCompute(pre_img_box, None)
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)


    formula_func(kp1,kp2,matches,frame,box)


def draw_image(frame,pre_frame,box,pre_box,pre_cal_time):
    global delta_time
    pre_len=abs(pre_box[0]-pre_box[2])
    # wit is height
    pre_wit=abs(pre_box[1]-pre_box[3])
    now_len=abs(box[0]-box[2])
    now_wit=abs(box[1]-box[3])
    now_mid=round((now_len+now_wit)/2,3)
    pre_mid=round((pre_len+pre_wit)/2,3)

    s=round(now_mid/pre_mid,3)

    delta_t=delta_time

    if s==1:
        TTC=-1
    else:
        TTC=round(delta_t/(s-1),3)

    '''
    filename = 'static_height.txt'
    with open (filename,'a') as file_object:
        #file_object.write(str(TTC)+'\n') 
        file_object.write(str(now_wit)+'\n') 
    filename = 'static_wit.txt'
    with open (filename,'a') as file_object:
        #file_object.write(str(TTC)+'\n') 
        file_object.write(str(now_len)+'\n') 
    '''
    '''
    filename = 'delta_height.txt'
    with open (filename,'a') as file_object:
        #file_object.write(str(TTC)+'\n') 
        file_object.write(str(now_wit-pre_wit)+'\n') 
    filename = 'delta_width.txt'
    with open (filename,'a') as file_object:
        #file_object.write(str(TTC)+'\n') 
        file_object.write(str(now_len-pre_len)+'\n') 
    '''


    if TTC>0 and TTC<time_threhold:
        cv2.putText(frame,box[-1]+' TTC:'+str(TTC)+' len:'+str(round(now_len,0))+' wit:'+str(round(now_wit,0)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
    else:
        cv2.putText(frame,box[-1]+' TTC:'+str(TTC)+' len:'+str(round(now_len,0))+' wit:'+str(round(now_wit,0)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
        TTC=-1



def function(frame,boxs,pre_boxs,pre_frame,pre_cal_time):
    for box in boxs:
        if box[-1]=='person':
            for pre_box in pre_boxs:
                '''
                a=abs(box[0]-pre_box[0])
                b=abs(box[1]-pre_box[1])
                c=abs(box[2]-pre_box[2])
                d=abs(box[3]-pre_box[3])
                flag=decide_same(a,b,c,d)
                '''
                flag=decide_same2(box,pre_box)
                if flag==True:
                    get_result(frame,pre_frame,box,pre_box,pre_cal_time)
                    break
        else:
            continue
            
def detection(img, boxs):
    for box in boxs:
                cv2.putText(img,box[-1], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

def catch_video(name='my_video', video_index=0):
    global delta_time

    cap = cv2.VideoCapture('./test_1.mp4') # creat camera class


    video_frame_cnt = int(cap.get(7))
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_fps = int(cap.get(5))
    f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')


    if not cap.isOpened():
        # wrong if no camera
        raise Exception('Check if the camera is on.')
    
    model = torch.hub.load('/temp/yolov5', 'custom', path='best.pt', source='local')
    model.eval()

    pre_catch, pre_frame = cap.read()  # read every frame of camera
    pre_results = model(pre_frame)
    pre_boxs = pre_results.pandas().xyxy[0].values
    pre_cal_time=time.time()
    now_time=time.time()

    filename = 'static_fea_dis.txt'
    with open (filename,'r+') as file_object:
        file_object.truncate(0)
    count=0
    while cap.isOpened(): 
        count+=1
        if count==25:
            pre_time=now_time
            now_time=time.time()
            delta_time=now_time-pre_time
            count=0
            start=time.time()       
            catch, frame = cap.read()  # read every frame of camera

            results = model(frame)
            boxs = results.pandas().xyxy[0].values

            yolo_time=time.time()-start

            #detection(frame,boxs)
            start_func=time.time()

            function(frame,boxs,pre_boxs,pre_frame,pre_cal_time)
            pre_cal_time=time.time()
            loss_time=time.time()-start_func

            pre_catch=catch
            pre_frame = frame 

            pre_boxs = boxs

        else:
            start=time.time()  
            catch, frame = cap.read()  # read every frame of camera

        
        
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            # q to exint
            break
        '''
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # click x to exit
            break
        '''
        print("FPS:", 1/(time.time()-start))

    
if __name__ == "__main__":    
    catch_video()
