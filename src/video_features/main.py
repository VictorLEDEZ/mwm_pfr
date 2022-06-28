import os
import platform
import sys

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm

from video_features.generate_summary import (create_summary,
                                             summary_frames_selection)
from video_features.shot_detection import define_shots


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

def summary_param(videos_order):

    """
    Function that returns the parameters needed for the output summary video (fps & resolution).

    Input:
            - videos_order      : array of video's file path ordering by last modification date
    Output:
            - min_fps           : integer -> minimal frame per seconds found in all videos
            - min_resolution    : tuple of integer -> minimal resolution found in all videos
    """

    fps_list = []
    for idx, file in enumerate(videos_order):
        video_cap = cv2.VideoCapture(file)

        # width of video frame
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height of video frame
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = width * height
        fps = int(round(video_cap.get(cv2.CAP_PROP_FPS)))  # fps of video frame

        fps_list.append(fps)  # append fps to list
        if idx == 0:  # init minimal resolution with first video
            min_resolution = (width, height)
        else:
            # find new minimal resolution
            if width * height < (min_resolution[0] * min_resolution[1]):
                # update minimal resolution value
                min_resolution = (width, height)
    min_fps = np.min(fps_list)  # select minimal fps in fps list
    return min_fps, min_resolution

def read_and_save_frames(videos_order):
    """
    Function that reads all videos in videos_order list and save all frames in a array. Return frames list and a
    dictionary with video parameters

    Input:
            - videos_order      : array of video's file path ordering by last modification date
    Output:
            - frames_list       : array of arrays -> frames matrix for each video
            - min_fps           : integer -> minimal fps found in all videos
            - min_resolution    : tuple of integers -> minimal resolution found in all videos
    """

    min_fps, min_resolution = summary_param(videos_order)

    pbar = tqdm(desc='Video', total=len(videos_order),
                position=0, leave=True)  # create loading bar

    frames_list = []  # array of arrays
    frame_nb = 0  # number of total frames counter
    for file in videos_order:  # loop over all videos in the directory
        video_list = []  # frame list for each video
        filename = file.split("/")[-1]  # extract file name from file path
        pbar.set_description(f'Videos -> {filename}')  # tqdm bar description
        pbar.refresh()  # to show immediately the update
        video_cap = cv2.VideoCapture(file)

        success, frame = video_cap.read()  # start reading with frame

        while success:  # keep reading until .read() return True
            frame_resized = cv2.resize(frame, min_resolution, interpolation=cv2.INTER_AREA)  # reshape frames on
            # the same resolution (minimum resolution)
            video_list.append(np.asarray(frame_resized))  # append frame to video list
            success, frame = video_cap.read()
            frame_nb += 1  # upgrade number of frames counter

        frames_list.append(video_list)  # add list
        pbar.update()
        video_cap.release()

    pbar.close()
    frames_list = np.array(frames_list)
    return frames_list, min_fps, min_resolution


def create_clip(summary_video_path, audio_path, clip_filename):
    """
    Function that creates a clip with video and audio

    Input:
            - summary_video_path  : string -> path to mp4 file
            - audio_path          : string -> path to audio file
            - clip_filename       : string -> filename of the output clip
    Output:
            Create mp4 file with audio and video
    """
    input_video = ffmpeg.input(str(summary_video_path))

    input_audio = ffmpeg.input(str(audio_path))

    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(
        clip_filename+'.mp4').run()

    # ffmpeg.concat(input_video, input_audio, v=1, a=1).output(
    #     clip_filename).run(cmd=r'C:\Users\RamziG5\anaconda3\envs\pfr\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win64-v4.2.2.exe')
