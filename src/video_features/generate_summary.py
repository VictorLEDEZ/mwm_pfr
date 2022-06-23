import cv2
import time
import itertools
from tqdm import tqdm


def summary_frames_selection(summary_duration, summary_fps, shot_percentage, dict_shots_order, min_shot_nb):
    """
    Function that generate an array of frame index selected for the summary and output the time in seconds before the
    most important frame in the summary.

    Input:
           - summary_duration   : int -> output time in seconds
           - summary_fps        : int -> number of frames per seconds for summary video in output
           - shot_percentage    : int -> percentage of the shot kept for the summary
           - dict_shots_order   : dict -> dictionary of shots ordering by their importance.
                                key: shot number
                                value: (shot mean, index of max value on the shot, (index first frame,index last frame))
           - min_shot_nb        : int -> minimum of shots that must be included in the summary
    Output:
           - summary_frames_index   : array of frame index selected for the summary
           - time_before_drop       : float -> time in seconds before the most important frame of the summary
    """

    output_frames_nb = summary_duration * summary_fps  # convert output duration in seconds into frame number
    nb_shots_used = 0  # count number of shots included in the summary

    while nb_shots_used < min_shot_nb:  # change percentage until the min of shots included in the summary is reached
        nb_shots_used = 0  # reset counter
        summary_frames_nb = 0  # number of frames selected counter
        summary_frames_index = []  # array of frame index selected

        best_shot = True

        for key, value in dict_shots_order.items():
            first_frame_index = value[2][0]  # first frame index of the current shot
            last_frame_index = value[2][1]  # last frame index of the current shot
            if best_shot:  # save best frame index of top1 shot
                best_frame = first_frame_index
                best_shot = False
            shot_frames_nb = last_frame_index - first_frame_index  # number of frames in shots
            summary_shot_frames_nb = int(
                shot_frames_nb * shot_percentage / 100)  # number of frames to select for the summary

            if summary_shot_frames_nb + summary_frames_nb > output_frames_nb:  # if length shot exceeds total frames
                # number
                summary_shot_frames_nb = output_frames_nb - summary_frames_nb  # complete by the number of frames
                # missing

            slice_inf = value[1] - int(summary_shot_frames_nb / 2)  # lower bound index of selected frames
            slice_sup = value[1] - int(
                summary_shot_frames_nb / 2) + summary_shot_frames_nb  # upper bound index of selected frames

            # conditions for not selecting frames from another shot
            # the number of frames outside the shot range is compensated on the other side of the bound

            if slice_inf < first_frame_index:  # if lower bound index is out of shot range
                offset = first_frame_index - slice_inf
                slice_inf = first_frame_index
                slice_sup += offset
            elif slice_sup > last_frame_index:  # if upper bound index is out of shot range
                offset = slice_sup - last_frame_index
                slice_sup = last_frame_index
                slice_inf -= offset

            shot_selection = list(range(slice_inf, slice_sup))  # generate a list of index between the two bounds
            summary_frames_nb += len(shot_selection)  # update number of selected frames
            summary_frames_index.extend(shot_selection)  # update list of frames selected

            nb_shots_used += 1  # increase shots number visited

            if output_frames_nb - summary_frames_nb < summary_fps:  # break when we reach expected output video
                # length or < one fps unit
                break

        shot_percentage -= 5  # update percentage -> decrease by 5%

    summary_frames_index.sort()  # sort list
    nb_frames_before_top = summary_frames_index.index(best_frame)  # index of best frame in list

    time_before_drop = nb_frames_before_top / summary_fps  # time in seconds before the most important frame
    summary_duration = len(summary_frames_index) / summary_fps  # summary time duration in seconds
    return summary_frames_index, time_before_drop, summary_duration


def create_summary(frames_list, summary_frames_index, summary_filename, summary_resolution, summary_fps):
    """
    Function that create a mp4 video from frame array

    Input:
            - frames_list           : array of arrays -> frames matrix for each video
            - summary_frames_index  : array of frames index selected
            - summary_filename      : string -> output file name
            - summary_resolution    : tuple -> resolution of summary video (width, height)
            - summary_fps           : int -> frames per seconds for summary video

    Output:
           return a mp4 video
    """

    frames_list = list(itertools.chain(*frames_list))  # flatten list
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(str(summary_filename) + '.mp4', fourcc, summary_fps, summary_resolution)

    for index, frame in enumerate(tqdm(frames_list)):  # loop over frame list & write frame if index in list selection
        if index in summary_frames_index:
            frame_resized = cv2.resize(frame, summary_resolution, interpolation=cv2.INTER_AREA)  # reshape frames on
            # the same resolution
            out.write(frame_resized)
            time.sleep(0.4)
    # When everything done, release the video capture and video write objects
    # Cleanup and save video
    cv2.destroyAllWindows()
    out.release()
