import warnings
warnings.filterwarnings("ignore")

from video_features.SIFT import Sift
from video_features.Flow import Flow
from video_features.object_det_score import object_det_score
from video_features.shot_detection import define_shots
from video_features.shots_order import shots_order
from tqdm import tqdm 

import os
from pathlib import Path
import numpy as np  # Change obj to np array (to remove)


def score(frame_list, shots, sampling_rate=10):
    def normalize(flow_mean):
        return (flow_mean - np.min(flow_mean)) / (np.max(flow_mean) - np.min(flow_mean))
    # Video Path
    # data_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), "Data")
    # video_path = os.path.join(data_path, "cut.mp4")

    # Run
    sift = []
    flow = []
    obj_score = []
    sift_norm = []
    flow_norm = []
    obj_score_norm = []

    max_sift = 0
    max_flow = 0
    max_object_score = 0

    # sift_video = []
    # flow_video = []
    # obj_score_video = []
    dir = os.getcwd()
    os.chdir("./darkflow-mast")


    active_sift = False

    if active_sift == True:
        for video in tqdm(frame_list):
            # for frame_number in range(len(video)):
            # if frame_number%sampling_rate==0:
            sift_video = Sift(video[::sampling_rate], frame_shift=1, display=False,
                            save_video=False)  # PATENTED ?
            flow_video = Flow(video[::sampling_rate], frame_shift=1,
                            display=False)
            obj, obj_score_video = object_det_score(video[::sampling_rate], gpu=1)

            # sift_video.append(Sift(video[::sampling_rate], frame_shift=1, display = False, save_video = False, sampling_rate=sampling_rate)) # PATENTED ?
            # flow_video.append(Flow(video[::sampling_rate], frame_shift=1, display = False, save_video = False, sampling_rate=sampling_rate))
            # obj_score_video.append(object_det_score(video[::sampling_rate], gpu=1)[1])
            # frame_number += 1

            # sift_norm = [np.zeros(flow[i].shape) for i in range(len(flow))]     # !!!!!!!! #
            sift_tmp = []
            flow_tmp = []
            obj_score_tmp = []

            if max(sift_video) > max_sift:
                max_sift = max(sift_video)
            if max(flow_video) > max_flow:
                max_flow = max(flow_video)
            if max(obj_score_video) > max_object_score:
                max_object_score = max(obj_score_video)

            for i, j, k in zip(sift_video, flow_video, obj_score_video):
                for s in range(sampling_rate):
                    sift_tmp.append(i)
                    flow_tmp.append(j)
                    obj_score_tmp.append(k)

            sift.append(sift_tmp)
            flow.append(flow_tmp)
            obj_score.append(obj_score_tmp)

        # print(np.shape(sift))
        # print(np.shape(sift[0]))

        # Normalization
        for video in range(len(sift)):
            sift_norm.append(np.array(sift[video]) / max_sift)
            flow_norm.append(np.array(flow[video]) / max_flow)
            obj_score_norm.append(np.array(obj_score[video]) / max_object_score)

            # print("FLOW", flow)
            # print("\n")

            # sift2 = [i for i in sift_video for s in range(sampling_rate)]
            # flow2 = [i for i in flow_video for s in range(sampling_rate)]
            # obj_score2 = [i for i in obj_score for s in range(sampling_rate)]
            # print("Flow2", flow2)

        # print(flow.shape)
        # print(sift.shape)
        # print(obj_score.shape)

    elif active_sift == False:
        for video in tqdm(frame_list):
            # for frame_number in range(len(video)):
            # if frame_number%sampling_rate==0:
            flow_video = Flow(video[::sampling_rate], frame_shift=1,
                            display=False)
            obj, obj_score_video = object_det_score(video[::sampling_rate], gpu=1)

            # sift_video.append(Sift(video[::sampling_rate], frame_shift=1, display = False, save_video = False, sampling_rate=sampling_rate)) # PATENTED ?
            # flow_video.append(Flow(video[::sampling_rate], frame_shift=1, display = False, save_video = False, sampling_rate=sampling_rate))
            # obj_score_video.append(object_det_score(video[::sampling_rate], gpu=1)[1])
            # frame_number += 1

            sift_tmp = []
            flow_tmp = []
            obj_score_tmp = []

            if max(flow_video) > max_flow:
                max_flow = max(flow_video)
            if max(obj_score_video) > max_object_score:
                max_object_score = max(obj_score_video)

            for j, k in zip(flow_video, obj_score_video):
                for s in range(sampling_rate):
                    flow_tmp.append(j)
                    obj_score_tmp.append(k)

            flow.append(flow_tmp)
            obj_score.append(obj_score_tmp)

        # print(np.shape(sift))
        # print(np.shape(sift[0]))

        # Normalization
        for video in range(len(flow)):
            flow_norm.append(np.array(flow[video]) / max_flow)
            obj_score_norm.append(np.array(obj_score[video]) / max_object_score)

        sift_norm = [np.zeros(flow_norm[i].shape) for i in range(len(flow))]     # !!!!!!!! #


    os.chdir(dir)

    def agregation(flow=flow_norm, sift=sift_norm, obj=obj_score_norm, flow_coef=0.5, sift_coef=0.5, obj_coef=1):
        agreg = [flow_coef*flow[i] + sift_coef*sift[i] +
                 obj_coef*obj_score[i] for i in range(len(flow))]
        return agreg

    agreg = agregation(flow=flow_norm, sift=sift_norm,
                       obj=obj_score_norm, flow_coef=0.5, sift_coef=0.5, obj_coef=1)

    agreg_sift = [inner for outer in sift_norm for inner in outer]
    agreg_flow = [inner for outer in flow_norm for inner in outer]
    agreg_obj_score = [inner for outer in obj_score_norm for inner in outer]
    agreg_agreg = [inner for outer in agreg for inner in outer]

    # shots = define_shots(video_path, nb_shots=10)
    len_cum = np.cumsum([len(i) for i in flow_norm])
    dict_shots_order, df_order = shots_order(
        shots, agreg_agreg, len_cum, display=True)

    ### To remove ###
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # len_cum = np.cumsum([len(i) for i in flow])
    frame_number = [i for i in range(0, len_cum[-1])]

    fig = make_subplots(rows=4, cols=1, subplot_titles=(
        'Objects', 'Optical Flow', 'SIFT', 'Agregation'))

    fig.append_trace(go.Scatter(
        x=frame_number,
        y=agreg_obj_score,
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=frame_number,
        y=agreg_flow,
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=frame_number,
        y=agreg_sift,
    ), row=3, col=1)

    fig.append_trace(go.Scatter(
        x=frame_number,
        y=agreg_agreg,
    ), row=4, col=1)
    for i in len_cum[:-1]:
        fig.add_vline(x=i, line_dash="dash")
    # title_text="Stacked Subplots"
    fig.update_layout(height=800, width=1000, showlegend=False)
    fig.show()

    return dict_shots_order
