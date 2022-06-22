import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import os
from pathlib import Path
import time
import imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def Flow(frame_list, frame_shift=1, display = False, save_video = False, sampling_rate=10):

    print("Processing Optical Flow...")
    def draw_flow(img, flow, step=16):

        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T

        lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

        return img_bgr


    def draw_hsv(flow):

        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]

        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)

        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    # def normalize(flow_mean):
    #     return (flow_mean - np.min(flow_mean)) / (np.max(flow_mean) - np.min(flow_mean))

    
    # cap = cv2.VideoCapture(video_path)

    # suc, prev = cap.read()
    img_lst = []
    img_lst2 = []
    flow_mean = []
    flow_mean2 = []
    frame_number = 0

    # while suc:
    for frame_number in range(len(frame_list)):
    #     if frame_number%sampling_rate==0:
        if frame_number != len(frame_list)-1:
            current_frame = np.array(frame_list[frame_number])
            next_frame = np.array(frame_list[frame_number+frame_shift])
            # prevgray = cv2.cvtColor(np.array(video[frame_number]), cv2.COLOR_BGR2GRAY)

            # if next_frame is None :
            #     break

            # print(frame_number)
            # img = frame_list.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            # img = np.array(video[frame_number])
            # suc, img = cap.read()

            # if img is None :
            #     break

            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)    
            # start time to calculate FPS
            start = time.time()

            # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = cv2.calcOpticalFlowFarneback(current_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_mean.append(np.mean(np.abs(flow)))
            
            # prevgray = gray

            # End time
            end = time.time()

            # calculate the FPS for current frame detection
            fps = 1 / (end-start)
            # print(f"{fps:.2f} FPS")

            if save_video == True :
                img_lst.append(draw_flow(next_frame, flow))
                img_lst2.append(draw_hsv(flow))

            if display == True :
                cv2.imshow('flow', draw_flow(next_frame, flow))
                cv2.imshow('flow HSV', draw_hsv(flow))

            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break

        # frame_number += 1
    
    flow_mean.append(flow_mean[-1])

    if display == True :
        frames = [i for i in range(0, len(flow_mean))]
        d = {'Frame number':frames,'Flow mean':flow_mean}
        df = pd.DataFrame(d)

        fig = px.bar(df, x='Frame number', y='Flow mean')
        fig.show()

    if save_video == True :
        imageio.mimsave(os.path.join(data_path, "flow.gif"), img_lst)
        imageio.mimsave(os.path.join(data_path, "flow_hsv.gif"), img_lst2)

    # flow_mean = normalize(np.array(flow_mean))
    # flow_mean2 = flow_mean2[:len(frame_list)]
    # print(int(frame_list.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Done.")
    return flow_mean

    # cap.release()
    # cv2.destroyAllWindows()

    # # Video Path
    # data_path = os.path.join(Path(os.getcwd()).parent.absolute(), "Data")
    # video_path = os.path.join(data_path, "cut.mp4")

    # # Run
    # flow = Flow(video_path, display = True)