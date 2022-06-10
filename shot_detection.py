import cv2
import numpy as np
import sys
import pandas as pd
import plotly.express as px
import os
from pathlib import Path
from tqdm import tqdm




# System Arguments

# Argument 1: the path to the videos directory
# Argument 2: Nb slots (int)
# Argument 3: Show visualization (True/False)

def color_hist(file_path,previous_hist):

    ###########################################################################################
    # Function to calculate the color difference histogram between two consecutive frames
    #
    # Input:
    #       - file path: path of the video
    #       - previous_hist: colour histogram of the last frame of the previous video
    # Output:
    #       - list_diff: array of color difference histogram
    #       - fps : frame rate of the video
    #       - last_frame_hist: colour histogram of the last frame of the current video
    ###########################################################################################


    cap = cv2.VideoCapture(file_path)
    fps=int(cap.get(cv2.CAP_PROP_FPS)) #get fps of the video

    list_diff=[]

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==True:
            hist = cv2.calcHist([frame], [0, 1, 2],None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) #color histogram
            frame_diff=np.sum(np.abs(hist-previous_hist)) # hist diff on all channels
            list_diff.append(frame_diff)

            previous_hist=hist #update histogram of previous frame
        else:
            cap.release()
    last_frame_hist=hist #save color histogram of last frame in order to be use as first frame of next video
    return list_diff,fps,last_frame_hist


def compute_videos_hist(dir_path):

    ###########################################################################################
    # Function to calculate the color difference histogram for all the videos in the directory
    #
    # Input:
    #       dir_path: path of the videos directory
    # Output:
    #       total_list_hist: array of color difference histogram for all videos
    #       dict_video_endIndex_fps: dictionary -> key:video last frame index / value: fps
    ###########################################################################################


    total_list_hist=[]                  #color histogram difference for all the videos
    dict_video_endIndex_fps={}          #dict key:video last frame index / value: fps
    last_frame_index=0                  #init for last frame video index
    previous_hist=np.zeros((8,8,8))     #init for first frame diff calculation

    files=[name for name in os.listdir(dir_path) if not name.startswith('.')]   #list video files in directory
    pbar = tqdm(desc='Video',total=len(files),position=0, leave=True)           # create loading bar

    for filename in files:              #loop over all videos in the directory
        pbar.set_description(f'Videos -> {filename}')
        pbar.refresh() # to show immediately the update
        file_path = os.path.join(dir_path, filename)
        list_hist,fps,last_frame_hist=color_hist(file_path,previous_hist)   #compute color histogram difference
        total_list_hist.extend(list_hist)
        last_frame_index+=len(list_hist)                #last frame index of the video
        dict_video_endIndex_fps[last_frame_index]=fps   #save for each video the last frame index and its fps
        previous_hist=last_frame_hist                   #update previous_hist for next video
        pbar.update()
    pbar.close()
    return total_list_hist,dict_video_endIndex_fps

def visualization(total_list_hist,shots):

    ###########################################################################################
    # Function to visualize shots selection depending on color difference histogram
    #
    # Input:
    #       total_list_hist: array of color difference histogram for all videos
    #       shots: array of frame index selected as shots separation
    # Output:
    #       plot figure
    ###########################################################################################


    df=pd.DataFrame() #init dataframe
    x=np.arange(1,len(total_list_hist)) #not taking into acount the first and last frame -> shots by default
    df["x"]=x
    df["hist"]=total_list_hist[1:]
    df["color"]=["shot detection" if i in shots else "frames" for i in x] #color legend to identify histograms selected
                                                                          # as shots
    fig = px.bar(df,x="x", y="hist", color="color")
    fig.update_traces(dict(marker_line_width=0))
    fig.show()

def define_shots(dir_path,nb_shots,range_min=1,show_viz=False):

    ###########################################################################################
    # Main function. Divide color difference histogram into a number of shots passed in inout parameter
    #
    # Input:
    #       dir_path: path of the videos directory
    #       nb_shots: integer for the number of shots to return
    #       range_min: integer -> number of FPS for minimum shot length
    #       show_viz: Boolean -> to visualize shots selection depending on color difference histogram
    # Output:
    #       shots: array of frame index defined as shots separations
    ###########################################################################################


    total_list_hist,dict_video_endIndex_fps=compute_videos_hist(dir_path) #call function to calculate the color
                                                            # difference histogram for all the videos in the directory

    n = len(total_list_hist)                        #number of total frames
    ranked = np.argsort(total_list_hist)            #sort index by color difference histogram value
    largest_indices = ranked[::-1][:n]              #invert order

    shots_index=[0,n]                               #add first and last frame as default shots
    for i,val in enumerate(largest_indices):        #loop over invert order list

        for key in dict_video_endIndex_fps.keys():  #select fps of the corresponding video frame
            if val<=key:
                fps=dict_video_endIndex_fps[key]
                break

        valid=False                                 #flag: possible to add frame in shot list or not

        for j in shots_index:                       #check if a frame at range_min*fps range is not already a shot
            if np.abs(j-val)>range_min*fps:
                valid=True
            else:
                valid=False
                break
        if valid==True:
            shots_index.append(val)

    shots=np.sort(shots_index[:nb_shots+1])         #ordering shots array


    if show_viz==True:                              #show visualization
        visualization(total_list_hist,shots)

    return shots


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print ("""
        Three arguments are required:
            - The first argument should be the path to the videos directory
            - The second argument is an integer for the number of shots to create
            - The third argument is a boolean for showing visualization or not
            """)
        sys.exit(0)

    dir_path=sys.argv[1]
    nb_shots=int(sys.argv[2])

    #range_min=int(sys.argv[3])
    show_viz=bool(sys.argv[3])

    dir_path = os.path.join(Path(os.getcwd()).absolute(), dir_path)

    shots=define_shots(dir_path,nb_shots,1,show_viz)
    print(shots)

