import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import os
from pathlib import Path
import time
import imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def Sift(video_path, display = False, save_video = False) :     # Changer gestion de path de save_video

    def normalize(sift_mean):
        return (sift_mean - np.min(sift_mean)) / (np.max(sift_mean) - np.min(sift_mean))

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    #sift
    sift = cv2.SIFT_create()

    #feature matching
    bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    cap = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path)

    if save_video == True :
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(os.path.join(data_path, "sift.mp4"), fourcc, 10.0, (1280,360))

    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    img_lst = []
    frame_number = 0
    frame_shift = 1
    sift_mean = []

    while success:
        current_frame = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        next_frame = cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_number+frame_shift)
        suc, current_frame = cap.read()
        suc, next_frame = cap2.read()

        if next_frame is None :
            break

        start = time.time()

        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)        

        keypoints_1, descriptors_1 = sift.detectAndCompute(current_frame,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(next_frame,None)

        matches = bf.match(descriptors_1,descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[200:250]

        matches_dist = [match.distance for match in matches]
        sift_mean.append(np.mean(matches_dist))
        # print(sift_mean)

        end = time.time()
        totalTime = end - start
        fps = 1 // totalTime
        # print("FPS: ", fps)
        
        if display == True :
            img3 = cv2.drawMatches(current_frame, keypoints_1, next_frame, keypoints_2, matches[200:250], next_frame, flags=2)      # Adapter taille de matches
            img_lst.append(img3)
            
            cv2.putText(img3, f'Frame number: {int(frame_number)}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            # cv2.putText(img3, f'FPS: {int(fps)}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            cv2.imshow('SIFT', img3)

        frame_number += 1

        if save_video == True :
            # Write video
            out.write(img3)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    if save_video == True :
        out.release()
        # Save video
        imageio.mimsave(os.path.join(data_path, "sift.gif"), img_lst)

    if display == True :
        frame_number = [i for i in range(0, len(sift_mean))]
        d = {'Frame number':frame_number,'SIFT mean':sift_mean}
        df = pd.DataFrame(d)

        fig = px.bar(df, x='Frame number', y='SIFT mean')
        fig.show()

    return normalize(np.array(sift_mean))

# # Video Path
# data_path = os.path.join(Path(os.getcwd()).parent.absolute(), "Data")
# video_path = os.path.join(data_path, "cut.mp4")

# # Run
# Sift(video_path, display = True, save_video = False)