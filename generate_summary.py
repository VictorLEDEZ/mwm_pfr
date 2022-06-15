import numpy as np
import cv2
import os
import sys
import time
from tqdm import tqdm
from shot_detection import ordering_videos


def frames_selection(list_frame_index,files_list):

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
    print(selection.shape)
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
    fps=29 #take min value in dict_video_endIndex_fps from  compute_videos_hist in shot detection file
    width=1920
    height=1080
    files_list= ordering_videos(dir_path)
    list_frame_index=[23,24,50,53,54,57,59,60,61,62,63,400,402,403,406,700,702,703,704,1500,1501,1502,1503,1504,1505,2500,2501,2502,3400,3401]
    selection=frames_selection(list_frame_index,files_list)
    create_summary(selection,result_name,width,height,fps)