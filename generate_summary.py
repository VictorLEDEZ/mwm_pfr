import numpy as np
import cv2
import os
import sys
import time
from tqdm import tqdm
from shot_detection import ordering_videos


def summary_frames_selection(summary_duration,summary_fps,shot_percentage,dict_shots_order,min_shot_nb):

    """
    Function that generate an array of frame index selected for the summary and output the time in seconds before the most
    important frame in the summary.

    Input:
           - summary_duration   : int -> output time in seconds
           - summary_fps        : int -> number of frames per seconds for summary video in output
           - shot_percentage    : int -> percentage of the shot kept for the summary
           - dict_shots_order   : dict -> dictionnary of shots ordering by their importance.
                                        key: shot number / value: (shot mean, index of max value on the shot, (index first frame,index last frame))
           - min_shot_nb        : int -> minimum of shots that must be included in the summary
    Output:
           - summary_frames_index   : array of frame index selected for the summary
           - time_before_drop       : float -> time in seconds before the most important frame of the summary
    """

    output_frames_nb=summary_duration*summary_fps #convert output duration in seconds into frame number
    nb_shots_used=0                               # count number of shots included in the summary

    while (nb_shots_used<min_shot_nb): # change percentage until the minimum of shots included in the summary is reached
        nb_shots_used=0                     # reset counter
        summary_frames_nb=0                 # number of frames selected counter
        summary_frames_index=[]             # array of frame index selected

        best_shot=True

        for key, value in dict_shots_order.items():
            first_frame_index=value[2][0]   # first frame index of the current shot
            last_frame_index=value[2][1]    # last frame index of the current shot
            if best_shot==True:             # save best frame index of top1 shot
                best_frame = value[1]
            shot_frames_nb = last_frame_index-first_frame_index               # number of frames in shots
            summary_shot_frames_nb=int(shot_frames_nb*shot_percentage/100)  # number of frames to select for the summary

            if summary_shot_frames_nb+summary_frames_nb>output_frames_nb: # if length shot exceeds total frames number
                summary_shot_frames_nb=output_frames_nb-summary_frames_nb # complete by the number of frames missing


            slice_inf = value[1]-int(summary_shot_frames_nb/2)                        # lower bound index of selected frames
            slice_sup = value[1]-int(summary_shot_frames_nb/2)+summary_shot_frames_nb # upper bound index of selected frames

            # conditions for not selecting frames from another shot
            # the number of frames outside the shot range is compensated on the other side of the bound

            if slice_inf < first_frame_index:        # if lower bound index is out of shot range
                offset=first_frame_index-slice_inf
                slice_inf=first_frame_index
                slice_sup+=offset
            elif slice_sup > last_frame_index:       # if upper bound index is out of shot range
                offset=slice_sup-last_frame_index
                slice_sup=last_frame_index
                slice_inf-=offset

            shot_selection=list(range(slice_inf,slice_sup)) # generate a list of index between the two bounds
            summary_frames_nb+=len(shot_selection)          # update number of selected frames
            summary_frames_index.extend(shot_selection)     # update list of frames selected

            nb_shots_used+=1                                # increase shots number visited

            if (output_frames_nb-summary_frames_nb < summary_fps):  # break when we reach expected output video length or < one fps unit
                break

        shot_percentage-=5                                  # update percentage -> decrease by 5%

    summary_frames_index.sort()                                 #sort list
    nb_frames_before_top=summary_frames_index.index(best_frame) #index of best frame in list

    time_before_drop=nb_frames_before_top/summary_fps            #time in seconds before the most important frame

    return summary_frames_index,time_before_drop

def frames_selection_download(list_frame_index,files_list):

    """
    Function that keep frames passed in a list of index

    Input:
           - list_frame_index: frame index array
    Output:
           - selection: array of frames selected
    """
                                               #list of files in datetime order
    pbar = tqdm(desc='Video',total=len(files_list),position=0, leave=True)           #create loading bar

    selection=[]
    count = 0
    for file_path in files_list:              #loop over all videos in the directory
        filename=file_path.split("/")[-1]
        pbar.set_description(f'Videos -> {filename}')
        pbar.refresh() # to show immediately the update
        print(file_path)
        vidcap = cv2.VideoCapture(file_path)
        success,image = vidcap.read()


        while success:
            if count in list_frame_index:
                selection.append(np.asarray(image))
            success,image = vidcap.read()
            count += 1
        pbar.update()
    pbar.close()
    selection = np.array(selection)
    return selection

def create_summary(frames_selected,output_filename,width,height,fps):

    """
    Function that create a mp4 video from frame array

    Input:
           - frames_selected: array of frames selected
           - output_filename: string -> output file name
           - width: -> width size
           - height: int -> height size
           - fps: int -> frames per seconde

    Output:
           return a mp4 video
    """


    out = cv2.VideoWriter(output_filename+'.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
    i=0
    for frame in frames_selected:
        out.write(frame)
        time.sleep(0.7)
        i+=1
    # When everything done, release the video capture and video write objects
    # Cleanup and save video
    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    dir_path=sys.argv[1]
    result_name="test_result"
    summary_fps=29 #take min value in dict_video_endIndex_fps from  compute_videos_hist in shot detection file
    width=1920 #take the min resolution
    height=1080
    summary_duration=30
    shot_percentage=50
    #dict_shots_order -> result from shot_order.py
    files_list= ordering_videos(dir_path)
    list_frame_index,time_before_drop=summary_frames_selection(summary_duration,summary_fps,shot_percentage,dict_shots_order)
    selection=frames_selection_download(list_frame_index,files_list)
    create_summary(selection,result_name,width,height,summary_fps)