import cv2
#import torch
#import time
def detection(img, boxs):
    for box in boxs:
                cv2.putText(img,box[-1], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
def catch_video(name='my_video', video_index=2):
    # cv2.namedWindow(name)
    #cap = cv2.VideoCapture(video_index) # creat camera class
    cap = cv2.VideoCapture('./result_TTC_2.mp4') # creat camera class

    '''
    video_frame_cnt = int(cap.get(7))
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_fps = int(cap.get(5))
    f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWriter_1 = cv2.VideoWriter('./result_1.mp4', f, video_fps, (video_width, video_height))
    videoWriter_2 = cv2.VideoWriter('./result_2.mp4', f, video_fps, (video_width, video_height))
    '''


    if not cap.isOpened():
        # wrong if no camera
        raise Exception('Check if the camera is on.')
    #model = torch.hub.load('/home/cc/test/yolov5', 'custom', path='best.pt', source='local')
    #model.eval()
    while cap.isOpened(): 
        #start=time.time()       
        catch, frame = cap.read()  # read every frame of camera

        #videoWriter_1.write(frame)

        #results = model(frame)
        #boxs = results.pandas().xyxy[0].values
        #detection(frame,boxs)
        cv2.imshow(name, frame) # show result
        #videoWriter_2.write(frame)
        #print("FPS:", 1/(time.time()-start))
        
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            # q to exint
            break
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # click x to exit
            break
        
    # release camera
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":    
    catch_video()
