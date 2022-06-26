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

def Sift(frame_list, frame_shift=1, display = False) :
    """
    Function that outputs the mean distance score of each frame in an array using the SIFT method.
    Each frame is compared to the next one and the distance between the two patterns is computed.

    Input:
           - frame_list         : list -> list of video frames
           - frame_shift        : int -> step between two frames for the sift comparison
           - display            : bool -> display the sift in real time
    Output:
           - sift_mean          : array of score for each frame
    """

    # print("Processing SIFT...")

    # Sift
    sift = cv2.SIFT_create()

    # Feature matching
    bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    
    img_lst = []
    sift_mean = []
    sift_mean2 = []
    frame_number = 0
    frame_list2 = frame_list


    for frame_number in tqdm(range(len(frame_list))):
        if frame_number != len(frame_list)-1:
            current_frame = cv2.resize(np.array(frame_list[frame_number]), (360, 640), interpolation=cv2.INTER_AREA)  # reshape frames on
            next_frame = cv2.resize(np.array(frame_list[frame_number+frame_shift]), (360, 640), interpolation=cv2.INTER_AREA)  # reshape frames on
            
            # Getting current frame and next frame
            # current_frame = np.array(frame_list[frame_number])
            # next_frame = np.array(frame_list[frame_number+frame_shift])

            # Converting to grayscale
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)        

            # Computing sift
            keypoints_1, descriptors_1 = sift.detectAndCompute(current_frame,None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(next_frame,None)

            # Check if keypoints matches in the two frames
            if len(keypoints_1) > 0 and len(keypoints_2) > 0:       # Check if keypoints are not empty
                matches = bf.match(descriptors_1,descriptors_2)
                matches = sorted(matches, key = lambda x:x.distance)
            
                match_step = 10
                # We only consider 1/match_step matches
                if len(matches) > match_step:
                    matches = matches[::len(matches)//match_step]

                matches_dist = [match.distance for match in matches]
                sift_mean.append(np.mean(matches_dist))
            
            else :
                sift_mean.append(sift_mean[-1])     # If keypoints are append the last computed score

            # Display sift video
            if display == True :
                img3 = cv2.drawMatches(current_frame, keypoints_1, next_frame, keypoints_2, matches, next_frame, flags=2)
                img_lst.append(img3)
                
                cv2.putText(img3, f'Frame number: {int(frame_number)}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                cv2.imshow('SIFT', img3)

    sift_mean.append(sift_mean[-1])     # Add one frame at the end to have consistant shape between methods

    # Display sift bar plot
    if display == True :
        frames = [i for i in range(0, len(sift_mean))]
        d = {'Frame number':frames,'SIFT mean':sift_mean}
        df = pd.DataFrame(d)

        fig = px.bar(df, x='Frame number', y='SIFT mean')
        fig.show()

    # print("Done.")
    return sift_mean