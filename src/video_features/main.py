import os
import platform

import cv2
import numpy as np
from tqdm import tqdm
import ffmpeg


def ordering_videos(dir_path):
    """
    Function to order videos by last modification date (in epoch time format).
    useful to keep the videos consistent.

    Input:
            - dir_path      : path of the videos directory
    Output:
            - videos_order  : array of video's file path ordering by last modification date
    """
    videos_order = {}  # dict -> key:filename / value: Datetime in epoch Time
    for file in os.listdir(dir_path):
        if not file.startswith('.'):  # avoid hidden files
            if platform.system() == 'Windows':
                video_datetime = os.path.getctime(dir_path + '/' + file)
            else:
                stat = os.stat(dir_path + '/' + file)
                try:
                    video_datetime = stat.st_birthtime
                except AttributeError:
                    # We're probably on Linux. No easy way to get creation dates here,
                    # so we'll settle for when its content was last modified.
                    video_datetime = stat.st_mtime
            file_path = os.path.join(dir_path, file)  # create file path
            videos_order[file_path] = video_datetime
    videos_order = list(dict(sorted(videos_order.items(),
                        key=lambda item: item[1])).keys())  # list of file path
    # ordering dict by dateTime
    return videos_order


def read_and_save_frames(videos_order):
    """
    Function that reads all videos in videos_order list and save all frames in a array. Return frames list and a
    dictionary with video parameters

    Input:
            - videos_order      : array of video's file path ordering by last modification date
    Output:
            - frames_list       : array of arrays -> frames matrix for each video
            - dict_videos_param : dict -> key: file_path / value: [fps,width,height,first_frame_index,last_frame_index]
    """

    pbar = tqdm(desc='Video', total=len(videos_order),
                position=0, leave=True)  # create loading bar

    frames_list = []  # array of arrays
    dict_videos_param = {}  # dict for parameters of each video
    frame_nb = 0  # number of total frames counter
    first_frame_index = 0
    for file in videos_order:  # loop over all videos in the directory
        video_list = []  # frame list for each video
        filename = file.split("/")[-1]  # extract file name from file path
        pbar.set_description(f'Videos -> {filename}')  # tqdm bar description
        pbar.refresh()  # to show immediately the update
        vidcap = cv2.VideoCapture(file)

        # width of video frame
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height of video frame
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))  # fps of video frame

        success, image = vidcap.read()  # start reading with frame

        while success:  # keep reading until .read() return True
            video_list.append(np.asarray(image))  # append frame to video list
            success, image = vidcap.read()
            frame_nb += 1  # upgrade number of frames counter

        frames_list.append(video_list)  # add list
        last_frame_index = frame_nb  # keep last frame index of video
        pbar.update()
        dict_videos_param[file] = [fps, width, height, first_frame_index,
                                   last_frame_index]  # add video parameters to dict
        # update first frame index for next video
        first_frame_index = last_frame_index + 1

    pbar.close()
    frames_list = np.array(frames_list)
    return frames_list, dict_videos_param


def summary_param(dict_videos_param):
    """
    Function that returns the parameters needed for the output summary video (fps & resolution).

    Input:
            - dict_videos_param : dict -> key: file_path / value: [fps,width,height,first_frame_index,last_frame_index]
    Output:
            - min_fps           : minimal frame per seconds found in all videos
            - min_resolution    : minimal resolution found in all videos
    """

    fps_list = []  # list of fps
    # loop over dictionary values
    for i, param in enumerate(dict_videos_param.values()):
        fps_list.append(param[0])  # append fps to list
        if i == 0:  # init minimal resolution with first video
            min_resolution = (param[1], param[2])
        else:
            # find new minimal resolution
            if param[1] * param[2] < (min_resolution[0] * min_resolution[1]):
                # update minimal resolution value
                min_resolution = (param[1], param[2])
    min_fps = np.min(fps_list)  # select minimal fps in fps list
    return min_fps, min_resolution

def create_clip(summary_video_path,audio_path,clip_filename):

    """
    Function that creates a clip with video and audio

    Input:
            - summary_video_path  : string -> path to mp4 file
            - audio_path          : string -> path to audio file
            - clip_filename       : string -> filename of the output clip
    Output:
            Create mp4 file with audio and video
    """

    input_video = ffmpeg.input(summary_video_path+'.mp4')

    input_audio = ffmpeg.input(audio_path)

    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(clip_filename+'.mp4').run()