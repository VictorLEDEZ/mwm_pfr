import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import os
from pathlib import Path
import time
import imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def Sift(frame_list, frame_shift=1, display = False, save_video = False, sampling_rate=10) :     # Changer gestion de path de save_video

    print("Processing SIFT...")
    def normalize(sift_mean):
        return (sift_mean - np.min(sift_mean)) / (np.max(sift_mean) - np.min(sift_mean))

    # vidcap = cv2.VideoCapture(video_path)
    # success, image = vidcap.read()

    #sift
    sift = cv2.SIFT_create()

    #feature matching
    bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    # cap = cv2.VideoCapture(video_path)
    # cap2 = cv2.VideoCapture(video_path)

    if save_video == True :
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(os.path.join(data_path, "sift.mp4"), fourcc, 10.0, (1280,360))

    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    img_lst = []
    sift_mean = []
    sift_mean2 = []
    frame_number = 0
    frame_list2 = frame_list

    # for frame_number in range(len(frame_list)):
        # if frame_number%sampling_rate==0:
    # print(len(frame_list))
    for frame_number in range(len(frame_list)):
        if frame_number != len(frame_list)-1:
            # print(frame_number)

            current_frame = np.array(frame_list[frame_number])
            next_frame = np.array(frame_list[frame_number+frame_shift])
            # success, current_frame = frame_list.read()
            # success2, next_frame = frame_list2.read()

            # if next_frame is None :
            #     break

            start = time.time()

            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)        

            keypoints_1, descriptors_1 = sift.detectAndCompute(current_frame,None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(next_frame,None)

            matches = bf.match(descriptors_1,descriptors_2)
            matches = sorted(matches, key = lambda x:x.distance)
            
            # print(len(matches))
            match_step = 10

            if len(matches) > match_step:
                matches = matches[::len(matches)//match_step]

            matches_dist = [match.distance for match in matches]
            sift_mean.append(np.mean(matches_dist))
            # print(sift_mean)

            end = time.time()
            totalTime = end - start
            fps = 1 // totalTime
            # print("FPS: ", fps)
                        
            if display == True :
                img3 = cv2.drawMatches(current_frame, keypoints_1, next_frame, keypoints_2, matches, next_frame, flags=2)      # Adapter taille de matches
                img_lst.append(img3)
                
                cv2.putText(img3, f'Frame number: {int(frame_number)}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                # cv2.putText(img3, f'FPS: {int(fps)}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                cv2.imshow('SIFT', img3)

            if save_video == True :
                # Write video
                out.write(img3)
                    # if cv2.waitKey(5) & 0xFF == ord('q'):
                    #     break

    # frame_number += 1

    sift_mean.append(sift_mean[-1])     # Add frame at the end

    if save_video == True :
        out.release()
        # Save video
        imageio.mimsave(os.path.join(data_path, "sift.gif"), img_lst)

    if display == True :
        frames = [i for i in range(0, len(sift_mean))]
        d = {'Frame number':frames,'SIFT mean':sift_mean}
        df = pd.DataFrame(d)

        fig = px.bar(df, x='Frame number', y='SIFT mean')
        fig.show()

    # for i in sift_mean:
    #     for j in range(sampling_rate):
    #         sift_mean2 = np.append(sift_mean2, i)
    # # sift_mean = normalize(np.array(sift_mean))
    # sift_mean2 = sift_mean2[:len(frame_list)]
    # print(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Done.")
    return sift_mean

            # cv2.destroyAllWindows()
            # cap.release()

        # # Video Path
        # data_path = os.path.join(Path(os.getcwd()).parent.absolute(), "Data")
        # video_path = os.path.join(data_path, "cut.mp4")

        # # Run
        # Sift(video_path, display = True, save_video = False)