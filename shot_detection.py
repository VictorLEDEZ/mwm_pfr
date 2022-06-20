import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import itertools


def compute_videos_hist(frames_list):
    """
    Function to calculate the color difference histogram for all the videos in the directory

     Input:
           frames_list      : array of arrays -> frames matrix for each video
     Output:
           list_hist_diff   : array of color difference histogram for all videos
    """

    list_hist_diff = []  # color histogram difference for all the videos
    previous_hist = np.zeros((8, 8, 8))  # init for first frame diff calculation

    pbar = tqdm(desc='Color Histogram computation', total=len(frames_list), position=0,
                leave=True)  # create loading bar

    for frame in frames_list:  # loop over all videos in the directory
        pbar.refresh()  # to show immediately the update
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])  # color histogram
        frame_diff = np.sum(np.abs(hist - previous_hist))  # hist diff on all channels
        list_hist_diff.append(frame_diff)

        previous_hist = hist  # update histogram of previous frame

        pbar.update()
    pbar.close()
    return list_hist_diff


def visualization(frames_list, shots):
    """
     Function to visualize shots selection depending on color difference histogram

     Input:
           frames_list  : array of arrays -> frames matrix for each video
           shots        : array of frame index selected as shots separation
     Output:
           plot figure
    """

    df = pd.DataFrame()  # init dataframe
    x = np.arange(1, len(frames_list))  # not taking into account the first and last frame -> shots by default
    df["x"] = x
    df["hist"] = frames_list[1:]
    df["color"] = ["shot detection" if i in shots else "frames" for i in
                   x]  # color legend to identify histograms selected
    # as shots
    fig = px.bar(df, x="x", y="hist", color="color")
    fig.update_traces(dict(marker_line_width=0))
    fig.show()


def define_shots(frames_list, videos_param, nb_shots, range_min=1, show_viz=False):
    """
     Main function. Divide color difference histogram into a number of shots passed in inout parameter

     Input:
            frames_list         : array of arrays -> frames matrix for each video
            videos_param        : dict -> key: file_path / value: [fps,width,height,first_frame_index,last_frame_index]
            nb_shots            : integer for the number of shots to return
            range_min           : integer -> number of FPS for minimum shot length
            show_viz            : Boolean -> to visualize shots selection depending on color difference histogram
     Output:
           shots: array of frame index defined as shots separations
    """
    frames_list = list(itertools.chain(*frames_list))  # flatten list

    list_hist_diff = compute_videos_hist(
        frames_list)  # call function to calculate the color difference histogram for each frame

    n = len(list_hist_diff)  # number of total frames
    ranked = np.argsort(list_hist_diff)  # sort index by color difference histogram value
    largest_indices = ranked[::-1][:n]  # invert order

    shots_index = [0, n]  # add first and last frame as default shots
    for i, val in enumerate(largest_indices):  # loop over invert order list

        for param in videos_param.values():  # select fps of the frame index
            if val <= param[4]:  # check index < or equal to first frame index of a video
                fps = param[0]
                break

        valid = False  # flag: possible to add frame in shot list or not

        for j in shots_index:  # check if a frame at range_min*fps range is not already a shot
            if np.abs(j - val) > range_min * fps:
                valid = True
            else:
                valid = False
                break
        if valid:
            shots_index.append(val)

    shots = np.sort(shots_index[:nb_shots + 1])  # ordering shots array

    if show_viz:  # show visualization
        visualization(list_hist_diff, shots)

    return shots
