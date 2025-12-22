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

time_threhold=4
area_threhold=0.6
frame_count_list=[]
frame_count=0
TTC_list=[]
delta_time=0
test_video='./test_1.mp4'
test_obj='car'
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#BRIEF_RATIO = 0.8

def get_dis(s1,s2):
    return sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )


def formula_func(kp1,kp2,matches,frame,box,show_img):
    total_dis_1=0
    total_dis_2=0
    #count=0
    #kp_range=len(matches)
    kp_range=min(5,len(matches))
    for i in range(0,kp_range-1):
        for j in range(i,kp_range):
            total_dis_1+=get_dis(kp1[matches[i].queryIdx].pt , kp1[matches[j].queryIdx].pt)
            total_dis_2+=get_dis(kp2[matches[i].trainIdx].pt , kp2[matches[j].trainIdx].pt)


    len_1=total_dis_1
    len_2=total_dis_2


    if len_2==0:
        cv2.putText(frame,box[-1]+'No matching found', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
        return
    s=len_1/len_2
    if s==1:
        return
    delta_t=delta_time
    TTC=round(delta_t/(s-1),3)
    print('t:',delta_t)
    #print('TTC: ',TTC)
    print(box[0],box[1])
    
    loop_num=kp_range
    for i in range(0,loop_num):
        cv2.circle(show_img,(int(kp1[matches[i].queryIdx].pt[0]),int(kp1[matches[i].queryIdx].pt[1])),2,(0,0,255),-1)
    show_img=cv2.resize(show_img, (0,0) ,fx=4,fy=4)



    filename = 'static_fea_dis.txt'
    with open (filename,'a') as file_object:
        if TTC>-50 and TTC<50:

            file_object.write(str(TTC)+'\n') 
    
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

    pre_img_box=pre_frame[int(pre_box[1]):int(pre_box[3]),int(pre_box[0]):int(pre_box[2])]
    

    img_box=frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    show_img=img_box.copy()

    img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
    pre_img_box = cv2.cvtColor(pre_img_box, cv2.COLOR_BGR2GRAY)

    img_box = cv2.GaussianBlur(img_box, (3, 3), 0, 0)
    pre_img_box = cv2.GaussianBlur(pre_img_box, (3, 3), 0, 0)

    edge1=cv2.Canny(img_box,threshold1=150,threshold2=200,L2gradient=True)
    edge2=cv2.Canny(pre_img_box,threshold1=150,threshold2=200,L2gradient=True)
    ''' using sift'''
    sift = cv2.SIFT_create()


    ''' using harris'''
    kp1 = detect_corners_harris(edge1)
    kp2 = detect_corners_harris(edge2)

    kp1,des1 = sift.compute(img_box,kp1)

    kp2,des2 = sift.compute(pre_img_box,kp2)
    
    if des1 is None or des2 is None:
        return
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    
    print('match:',len(matches))
    print('kp:',len(kp1))
    if len(matches)==0:
        return
    matches = sorted(matches, key=lambda x: x.distance)

    formula_func(kp1,kp2,matches,frame,box,show_img)

def get_result_2(frame,pre_frame,box,pre_box,pre_cal_time):
    global delta_time

    pre_img_box=pre_frame[int(pre_box[1]):int(pre_box[3]),int(pre_box[0]):int(pre_box[2])]
    

    img_box=frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    show_img=img_box.copy()

    img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
    pre_img_box = cv2.cvtColor(pre_img_box, cv2.COLOR_BGR2GRAY)




    sift = cv2.SIFT_create()
    kp1 = sift.detect(img_box,None)
    kp2 = sift.detect(pre_img_box,None)
    kp1,des1 = sift.compute(img_box,kp1)
    kp2,des2 = sift.compute(pre_img_box,kp2)
    
    if des1 is None or des2 is None:
        return
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    
    print('match:',len(matches))
    print('kp:',len(kp1))
    if len(matches)==0:
        return
    matches = sorted(matches, key=lambda x: x.distance)
    formula_func(kp1,kp2,matches,frame,box,show_img)



def get_result_3(frame,pre_frame,box,pre_box,pre_cal_time):
    global delta_time
    #edge_const=10
    wit=int(box[2])-int(box[0])
    hei=int(box[3])-int(box[1])
    const_wit=int(wit/4)
    const_hei=int(hei/4)

    pre_img_box=pre_frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    show_pre_img=pre_img_box.copy()

    img_box=frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    show_img=img_box.copy()


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

    loop_num=len(good_old)
    for i in range(0,loop_num):
        cv2.circle(show_pre_img,(int(good_old[i][0]),int(good_old[i][1])),2,(0,0,255),-1)
    show_pre_img=cv2.resize(show_pre_img, (0,0) ,fx=4,fy=4)

    st_0=good_old[0]
    st_1=good_new[0]
    total_dis_0=0
    total_dis_1=0
    kp_range=len(good_new)
    for i in range(0,kp_range-1):
        for j in range(0,kp_range):
            total_dis_0+=get_dis(good_old[i], good_old[j])
            total_dis_1+=get_dis(good_new[i] , good_new[j])

    if total_dis_0==0:
        return
    len_0=total_dis_0

    len_1=total_dis_1

    s=len_1/len_0
    
    if s==1:
        return
    delta_t=delta_time
    #print(delta_t)
    TTC=round(delta_t/(s-1),3)
    print('match',len(good_new))

    #cv2.waitKey(0)
    filename = 'static_fea_dis.txt'
    with open (filename,'a') as file_object:
        if TTC>0 and TTC <50:
            file_object.write(str(TTC)+'\n') 

    cv2.putText(frame,box[-1]+str(TTC), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

def function(frame,boxs,pre_boxs,pre_frame,pre_cal_time):
    obj_list=['t_sign','tl_green','tl_red','tl_yellow','tl_none']
    sift_time=time.time()
    for box in boxs:
        #if box[-1] not in obj_list :
        if box[-1]==test_obj:
            for pre_box in pre_boxs:
                flag=decide_same2(box,pre_box)
                if flag==True:
                    get_result_2(frame,pre_frame,box,pre_box,pre_cal_time)
                    break
        else:
            continue
    print('time for sift'+str(time.time()-sift_time))

    klt_time=time.time()
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
    print('time for klt'+str(time.time()-klt_time))

    h_time=time.time()
    for box in boxs:
        #if box[-1] not in obj_list :
        if box[-1]==test_obj:
            for pre_box in pre_boxs:
                flag=decide_same2(box,pre_box)
                if flag==True:
                    get_result(frame,pre_frame,box,pre_box,pre_cal_time)
                    break
        else:
            continue
    print('time for GH'+str(time.time()-h_time))
    
            
def detection(img, boxs):
    for box in boxs:
                cv2.putText(img,box[-1], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

def catch_video(name='my_video', video_index=2):
    global delta_time
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


            yolo_time=time.time()
            results = model(frame)
            boxs = results.pandas().xyxy[0].values
            print('time for yolov5:'+str(time.time()-yolo_time))
            

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



def catch_video_2():
    global delta_time
    model = torch.hub.load('/home/cc/test/yolov5', 'custom', path='best.pt', source='local')
    model.eval()

    filename = 'static_fea_dis.txt'
    with open (filename,'r+') as file_object:
        file_object.truncate(0)
    pre_cal_time=0
    pre_frame=cv2.imread('/home/cc/test/yolov5/waymo_od/images/seg_3/0.jpg')
    frame = cv2.imread('/home/cc/test/yolov5/waymo_od/images/seg_3/1.jpg') # read every frame of camera
    pre_results = model(pre_frame)
    pre_boxs = pre_results.pandas().xyxy[0].values
    results = model(frame)
    boxs = results.pandas().xyxy[0].values
    #print('time for yolov5:'+str(time.time()-start))
    function(frame,boxs,pre_boxs,pre_frame,pre_cal_time)
    pre_cal_time=time.time()


    frame=cv2.resize(frame, (0,0) ,fx=0.5,fy=0.5)
    #cv2.imshow('img', frame) # show result
    cv2.waitKey(0)





if __name__ == "__main__":    
    catch_video()
