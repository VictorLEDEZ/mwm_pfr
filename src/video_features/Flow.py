import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import os
from pathlib import Path
import time
import imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

def Flow(frame_list, frame_shift=1, display = False):
    """
    Function that outputs the mean distance score of each frame in an array using the Optical Flow method.

    Input:
           - frame_list         : list -> list of video frames
           - frame_shift        : int -> step between two frames for the sift comparison
           - display            : bool -> display the sift in real time
    Output:
           - flow_mean          : array of score for each frame
    """

    # Display functions
    def draw_flow(img, flow, step=16):      # Arrow flow

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


    def draw_hsv(flow):     # HSV flow

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

    img_lst = []
    img_lst2 = []
    flow_mean = []
    flow_mean2 = []
    frame_number = 0

    for frame_number in tqdm(range(len(frame_list))):
        if frame_number != len(frame_list)-1:
            # Getting current frame and next frame
            current_frame = cv2.resize(np.array(frame_list[frame_number]), (360, 640), interpolation=cv2.INTER_AREA)  # reshape frames on
            next_frame = cv2.resize(np.array(frame_list[frame_number+frame_shift]), (360, 640), interpolation=cv2.INTER_AREA)  # reshape frames on
            
            # Converting to grayscale
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            # Computing optical flow
            flow = cv2.calcOpticalFlowFarneback(current_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_mean.append(np.mean(np.abs(flow)))

            # Display optical flow videos
            if display == True :
                cv2.imshow('flow', draw_flow(next_frame, flow))
                cv2.imshow('flow HSV', draw_hsv(flow))
    
    flow_mean.append(flow_mean[-1])     # Add one frame at the end to have consistant shape between methods

    # Display optical flow bar plot
    if display == True :
        frames = [i for i in range(0, len(flow_mean))]
        d = {'Frame number':frames,'Flow mean':flow_mean}
        df = pd.DataFrame(d)

        fig = px.bar(df, x='Frame number', y='Flow mean')
        fig.show()

    return flow_mean