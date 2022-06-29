import warnings
warnings.filterwarnings("ignore")

from video_features.SIFT import Sift
from video_features.Flow import Flow
from video_features.object_det_score import object_det_score
from video_features.shot_detection import define_shots
from video_features.shots_order import shots_order
from tqdm import tqdm 
import pathlib

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import os
from pathlib import Path
import numpy as np


def score(frame_list, shots, sampling_rate=10):
    """
    Function that computes SIFT, Optical Flow and Object Detection.
    Then, calculate the score of each frame for each feature then order the shots by score.

    Input:
           - frame_list         : list -> list of video frames
           - shots              : list -> list of video shot
           - sampling_rate      : int -> rate of frame for computation
    Output:
           - dict_shot_order    : dictionnary of score for each shot
    """
    sift = []
    flow = []
    obj_score = []
    sift_norm = []
    flow_norm = []
    obj_score_norm = []

    max_sift = 0
    max_flow = 0
    max_object_score = 0

    dir = os.getcwd()
    darkflow_path = pathlib.Path(__file__).parent.parent.parent.joinpath("darkflow-mast")
    os.chdir(darkflow_path)

    # Compute SIFT, Optical Flow, Object Detection
    for video in tqdm(frame_list):
        sift_video = Sift(video[::sampling_rate], frame_shift=1, display=False)
        flow_video = Flow(video[::sampling_rate], frame_shift=1,
                        display=False)
        obj, obj_score_video = object_det_score(video[::sampling_rate], gpu=1)

        sift_tmp = []
        flow_tmp = []
        obj_score_tmp = []

        # Get maximum score for normalization
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

    # Normalization
    for video in range(len(sift)):
        sift_norm.append(np.array(sift[video]) / max_sift)
        flow_norm.append(np.array(flow[video]) / max_flow)
        obj_score_norm.append(np.array(obj_score[video]) / max_object_score)

    os.chdir(dir)

    # Agregation of the 3 scores
    def agregation(flow=flow_norm, sift=sift_norm, obj=obj_score_norm, flow_coef=0.5, sift_coef=0.5, obj_coef=1):
        agreg = [flow_coef*flow[i] + sift_coef*sift[i] + obj_coef*obj_score[i] for i in range(len(flow))]
        return agreg

    agreg = agregation(flow=flow_norm, sift=sift_norm,
                       obj=obj_score_norm, flow_coef=0.5, sift_coef=0.5, obj_coef=1)

    agreg_sift = [inner for outer in sift_norm for inner in outer]
    agreg_flow = [inner for outer in flow_norm for inner in outer]
    agreg_obj_score = [inner for outer in obj_score_norm for inner in outer]
    agreg_agreg = [inner for outer in agreg for inner in outer]

    len_cum = np.cumsum([len(i) for i in flow_norm])
    dict_shots_order, df_order = shots_order(
        shots, agreg_agreg, len_cum, display=True)

    # Visualization
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
    fig.update_layout(height=800, width=1000, showlegend=False)
    fig.show()

    return dict_shots_order
