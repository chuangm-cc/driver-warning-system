import cv2
import torch
import time
from math import sqrt
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from util import detect_corners
from util import computeBrief
from util import briefMatch
import os
import signal

time_threhold=2.5
area_threhold=0.6
frame_count_list=[]
frame_count=0
TTC_list=[]
delta_time=0
test_video='./test_1.mp4'
test_obj='person'
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
ppid = 0

#BRIEF_RATIO = 0.8
def get_dis(s1,s2):
    return sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )


def formula_func(kp1,kp2,matches,frame,box):
    total_dis_1=0
    total_dis_2=0
    count=0

    st_1=kp1[matches[0].queryIdx].pt
    st_2=kp2[matches[0].trainIdx].pt
    #print(st_1[0])
    #print('num:',len(matches))
    kp_range=min(5,len(matches))
    for i in range(0,kp_range):
        total_dis_1+=get_dis(kp1[matches[i].queryIdx].pt , st_1)
        total_dis_2+=get_dis(kp2[matches[i].trainIdx].pt , st_2)
        count+=1
        #print(matches[i].distance)

    len_1=total_dis_1/count
    len_2=total_dis_2/count
    #print(len_1,len_2)

    if len_2==0:
        cv2.putText(frame,box[-1]+'No matching found', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
        return
    s=len_1/len_2
    if s==1:
        return
    delta_t=delta_time
    TTC=round(delta_t/(s-1),3)



    filename = 'static_fea_dis.txt'
    with open (filename,'a') as file_object:
        #if TTC>0 and TTC<50:
        if abs(s-1)<0.1:
            file_object.write(str(s)+'\n') 
    
    if s==1:
        cv2.putText(frame,box[-1]+'static TTC:+inf', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
    else:
        TTC=round(delta_t/(s-1),3)
        cv2.putText(frame,box[-1]+'TTC:'+str(TTC), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
          

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

def detect_corners_harris(image):
    dst = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
    #dst=cv2.dilate(dst,None)
    kp = (np.argwhere(dst > 0.2 * dst.max())).astype(float)
    kp = [cv2.KeyPoint(x[1], x[0], 1) for x in kp]
    return kp

def get_result(frame,pre_frame,box,pre_box,pre_cal_time):
    global delta_time

    pre_img_box=pre_frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    test_img=pre_img_box.copy()
    #cv2.imshow('box', pre_img_box)
    img_box=frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]

    img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
    pre_img_box = cv2.cvtColor(pre_img_box, cv2.COLOR_BGR2GRAY)

    img_box = cv2.GaussianBlur(img_box, (3, 3), 0, 0)
    pre_img_box = cv2.GaussianBlur(pre_img_box, (3, 3), 0, 0)
    #kp=detect_corners(img_box)
    #pre_kp=detect_corners(pre_img_box)
    #print(kp.shape[0])
    #orb=cv2.ORB_create()
    #kp1, des1 = orb.detectAndCompute(img_box, None)
    #kp2, des2 = orb.detectAndCompute(pre_img_box, None)
    #surf= cv2.SURF_create()
    #blur=cv2.blur()
    edge1=cv2.Canny(img_box,threshold1=150,threshold2=200,L2gradient=True)
    edge2=cv2.Canny(pre_img_box,threshold1=150,threshold2=200,L2gradient=True)
    ''' using sift'''
    sift = cv2.SIFT_create()
    #orb = cv2.ORB_create()
    # kp1 = sift.detect(img_box,None)
    # kp2 = sift.detect(pre_img_box,None)

    ''' using harris'''
    kp1 = detect_corners_harris(edge1)
    kp2 = detect_corners_harris(edge2)
    img2 = cv2.drawKeypoints(edge1, keypoints=kp1, outImage=None, color=(255,0,0))
    #cv2.imshow('cam1_time0', img2)
    #print(len(kp1))
    kp1,des1 = sift.compute(img_box,kp1)
    #kp2 = sift.detect(pre_img_box,None)
    kp2,des2 = sift.compute(pre_img_box,kp2)
    #kp1, des1 = sift.detectAndCompute(img_box, None)
    #kp2, des2 = sift.detectAndCompute(pre_img_box, None)
    #print(len(des1))
    if des1 is None or des2 is None:
        return
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    
    print('match:',len(matches))
    print('kp:',len(kp1))
    if len(matches)==0:
        return
    matches = sorted(matches, key=lambda x: x.distance)
    #print(len(matches))
    #for i in range(0,4):
        #print(kp1[matches[i].queryIdx].pt)
        #print(kp2[matches[i].trainIdx].pt)
    #print(matches)
    #kp = sift.detect(img_box, None)
    #pre_kp = sift.detect(pre_img_box, None)
    #pre_locs, pre_desc = sift.compute(pre_img_box, pre_kp)
    #locs,desc = sift.compute(img_box, kp) 
    #print(locs.shape[0])
    #matches = briefMatch(desc, pre_desc, 0.5)

    #kp_0 = kp2[matches[:,1], :]
    #kp_1 = kp1[matches[:,0], :]
    #print(kp_1.shape[0])

    formula_func(kp1,kp2,matches,frame,box)

def get_result_3(frame,pre_frame,box,pre_box,pre_cal_time):
    global delta_time
    global ppid
    ppid = os.getppid()

    pre_img_box=pre_frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    test_img=pre_img_box.copy()
    #cv2.imshow('box', pre_img_box)
    img_box=frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]

    img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
    pre_img_box = cv2.cvtColor(pre_img_box, cv2.COLOR_BGR2GRAY)

    p0=cv2.goodFeaturesToTrack(pre_img_box,mask=None, **feature_params)
    if p0 is None:
        return
    p1,st,err=cv2.calcOpticalFlowPyrLK(pre_img_box,img_box,p0,None,**lk_params)

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    if len(good_old)==0 or len(good_new)==0:
        return
    #print(good_old[0],good_new[0])
    st_0=good_old[0]
    st_1=good_new[0]
    total_dis_0=0
    total_dis_1=0
    count=0
    print(len(good_new),len(good_old))
    kp_range=min(5,len(good_new))
    for i in range(0,kp_range):
        total_dis_0+=get_dis(good_old[i], st_0)
        total_dis_1+=get_dis(good_new[i] , st_1)
        count+=1
        #print(matches[i].distance)
    if total_dis_0==0:
        return
    len_0=total_dis_0
    #/count
    len_1=total_dis_1
    #/count
    s=len_1/len_0
    print(len_0,len_1)
    if s==1:
        return
    delta_t=delta_time

    TTC=round(delta_t/(s-1),3)
    if(TTC<time_threhold):
        os.kill(ppid, signal.SIGALRM)
    filename = 'static_fea_dis.txt'
    with open (filename,'a') as file_object:
        #if TTC>0 and TTC<50:
        #if abs(s-1)<0.1:
        file_object.write(str(TTC)+'\n') 

    cv2.putText(frame,box[-1]+str(TTC), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)



def function(frame,boxs,pre_boxs,pre_frame,pre_cal_time):
    obj_list=['t_sign','tl_green','tl_red','tl_yellow','tl_none']
    for box in boxs:
        #if box[-1] not in obj_list :
        if box[-1]==test_obj:
            for pre_box in pre_boxs:
                flag=decide_same2(box,pre_box)
                if flag==True:
                    get_result_3(frame,pre_frame,box,pre_box,pre_cal_time)
                    break
        else:
            continue
            
def detection(img, boxs):
    for box in boxs:
                cv2.putText(img,box[-1], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

def catch_video(name='my_video', video_index=0):
    global delta_time
    # cv2.namedWindow(name)
    #cap = cv2.VideoCapture(video_index) # creat camera class
    cap = cv2.VideoCapture(test_video) # creat camera class
    #cap = cv2.VideoCapture('./result.mp4') # creat camera class

    video_frame_cnt = int(cap.get(7))
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_fps = int(cap.get(5))
    f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    #videoWriter_1 = cv2.VideoWriter('./result_TTC.mp4', f, video_fps, (video_width, video_height))

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
        if count==1:
            pre_time=now_time
            now_time=time.time()
            delta_time=now_time-pre_time
            #print('delta_t:'+str(delta_time))
            count=0
            start=time.time()       
            catch, frame = cap.read()  # read every frame of camera
            #frame=cv2.imread('static_img.jpg')

            results = model(frame)
            boxs = results.pandas().xyxy[0].values

            #print('time for yolov5:'+str(time.time()-start))
            yolo_time=time.time()-start

            #detection(frame,boxs)
            start_func=time.time()

            function(frame,boxs,pre_boxs,pre_frame,pre_cal_time)
            pre_cal_time=time.time()

            #print('time of max loss:'+str(time.time()-start_func))
            loss_time=time.time()-start_func

            

            #cv2.imshow(name, frame) # show result
            #videoWriter_1.write(frame)
            
            #print(str(time.time()-start))

            pre_catch=catch
            pre_frame = frame 
            #pre_results = model(pre_frame)
            pre_boxs = boxs
            #print('time for whole process:'+str(time.time()-start))
            #print('ratio of max loss caused by different time:'+str(round(loss_time/(time.time()-start),10)))
        else:
            start=time.time()  
            catch, frame = cap.read()  # read every frame of camera
            # cv2.imshow(name, frame) # show result
            #print(str(time.time()-start))
        
        
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

    # release camera
    
    #cap.release()
    #cv2.destroyAllWindows()
    
if __name__ == "__main__":    
    catch_video()
