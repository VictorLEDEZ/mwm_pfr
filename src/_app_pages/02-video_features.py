import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import itertools
import time

DARKFLOW_PATH = (Path(__file__).parent.parent.parent / "darkflow").absolute()
sys.path.append(str(DARKFLOW_PATH))

from video_features.SIFT import Sift
from video_features.Flow import Flow
from video_features.object_det_score import object_det_score
from video_features.shot_detection import define_shots, compute_videos_hist
from video_features.shots_order import shots_order
from video_features.main import (create_clip, ordering_videos,
                                 read_and_save_frames, summary_param)
from video_features.generate_summary import (create_summary,
                                             summary_frames_selection)
from music_features.beat_tracking import get_downbeats


@st.cache
def extract_downbeats(audio_path):
    return get_downbeats(audio_path)


@st.cache
def extract_col_hist(frames_list):
    return compute_videos_hist(frames_list)


@st.cache
def extract_frames_list(session_videos):
    return read_and_save_frames(session_videos)


@st.cache
def compute_feature_score(frame_list_key, shots, sampling_rate=10):
    frame_list = st.session_state[frame_list_key]
    def normalize(flow_mean):
        return (flow_mean - np.min(flow_mean)) / (np.max(flow_mean) - np.min(flow_mean))

    sift = []
    flow = []
    obj_score = []
    sift_norm = []
    flow_norm = []
    obj_score_norm = []

    max_sift = 0
    max_flow = 0
    max_object_score = 0

    active_sift = True

    if active_sift:
        for video in frame_list:
            sift_video = Sift(video[::sampling_rate], frame_shift=1, display=False)  # PATENTED ?
            flow_video = Flow(video[::sampling_rate], frame_shift=1,
                            display=False)
            obj, obj_score_video = object_det_score(video[::sampling_rate],
                                                    model_path=str(DARKFLOW_PATH / "cfg/yolo.cfg"),
                                                    weights_path=str(DARKFLOW_PATH / "bin/yolo.weights"),
                                                    config_path=str(DARKFLOW_PATH / "cfg"),
                                                    gpu=1)
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

        for video in range(len(sift)):
            sift_norm.append(np.array(sift[video]) / max_sift)
            flow_norm.append(np.array(flow[video]) / max_flow)
            obj_score_norm.append(np.array(obj_score[video]) / max_object_score)

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

    len_cum = np.cumsum([len(i) for i in flow_norm])
    dict_shots_order, df_order = shots_order(
        shots, agreg_agreg, len_cum, display=False)


    # len_cum = np.cumsum([len(i) for i in flow])
    frame_number = [i for i in range(0, len_cum[-1])]

    plot_data = {"frame_number": frame_number,
                 "len_cum": len_cum,
                 "agreg_obj_score": agreg_obj_score,
                 "agreg_flow": agreg_flow,
                 "agreg_sift": agreg_sift,
                 "agreg_agreg": agreg_agreg}

    return dict_shots_order, df_order, plot_data


def score_plot(plot_data):
    fig = make_subplots(rows=4, cols=1, subplot_titles=(
        'Objects', 'Optical Flow', 'SIFT', 'Agregation'))
    len_cum = plot_data["len_cum"]
    frame_number = plot_data["frame_number"]
    agreg_obj_score = plot_data["agreg_obj_score"]

    agreg_flow = plot_data["agreg_flow"]
    agreg_sift = plot_data["agreg_sift"]
    agreg_agreg = plot_data["agreg_agreg"]

    fig.add_trace(go.Scatter(
        x=frame_number,
        y=agreg_obj_score,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=frame_number,
        y=agreg_flow,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=frame_number,
        y=agreg_sift,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=frame_number,
        y=agreg_agreg,
    ), row=4, col=1)
    for i in len_cum[:-1]:
        fig.add_vline(x=i, line_dash="dash")
    # title_text="Stacked Subplots"
    fig.update_layout(height=600, width=1200, showlegend=False)
    return fig


def display():
    st.title("Video Features")
    st.header("Extracting shots")
    session_videos = st.session_state["ordered_videos"]

    # Shot extraction and color histograms
    # ____________________________________
    if not session_videos:
        st.subheader("Upload some videos in the context page!")
        st.stop()

    col, _ = st.columns(2)
    with col:
        number_of_shots = st.slider('Select the number of shots', 1, 20, 10)
        shot_pct_duration = st.slider('Percentage per shot', 1, 100, 50)

        if st.button("Extract Video Features"):
            stime = time.time()
            with st.spinner("Extracting video frames..."):
                frames_list, summary_fps, summary_resolution = extract_frames_list(session_videos)
                st.metric("Summary FPS", summary_fps)
                st.metric("Summary resolution", str(summary_resolution))
                st.session_state["frames_list"] = frames_list
                st.session_state["summary_fps"] = summary_fps
                st.session_state["summary_resolution"] = summary_resolution

            with st.spinner("Bounding shots..."):
                audio_path = st.session_state["audio_path"]
                downbeats_frequency, downbeat_times = extract_downbeats(audio_path)
                st.session_state["downbeats_frequency"] = downbeats_frequency
                st.session_state["downbeat_times"] = downbeat_times

                if not downbeats_frequency:
                    st.write("Video shots require the downbeats frequency")
                shots = define_shots(frames_list, summary_fps, number_of_shots, shot_pct_duration, downbeats_frequency)
                st.session_state["shots"] = shots

            with st.spinner("Computing color histogram..."):
                col_histogram = st.session_state["col_histogram"]
                if not col_histogram:
                    col_histogram = compute_videos_hist(list(itertools.chain(*frames_list)))
                    st.session_state["col_histogram"] = col_histogram


    shots = st.session_state["shots"]
    col_histogram = st.session_state["col_histogram"]
    if shots.size != 0 and col_histogram:
        viz, txt = st.columns(2)
        df = pd.DataFrame()
        x = np.arange(1, len(col_histogram))
        df["Frames"] = x
        df["hist"] = col_histogram[1:]
        df["color"] = ["shot detection" if i in shots else "frames" for i in x]
        with txt:
            st.subheader('Plot Comment')
            st.write("Shots defined at the following bouderies.", list(shots))
        with viz:
            fig = px.bar(df, x="Frames", y="hist", color="color", title="Frames color histogram")
            fig.update_layout(height=800, width=1200)
            fig.update_traces(dict(marker_line_width=0))
            st.plotly_chart(fig, height=800, width=1200)

    # Video feature extraction
    # ____________________________________
    st.header("Video feature extraction")
    shots = st.session_state["shots"]
    if shots.size == 0:
        st.write("Feature extraction requires defined video shots !")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        sampling_rate = col1.slider('Select frames sampling rate', 1, 30, 10)
        feature_selection = st.multiselect(
            'Video feature selection',
            ['objects', 'flow', 'sift'],
            ['objects', 'flow', 'sift'])

    col1, col2 = st.columns(2)

    with st.spinner("Extracing video features: sift, optical flow and object detection..."):
        stime = time.time()
        frames_list, summary_fps, summary_resolution = extract_frames_list(session_videos)
        st.session_state["frames_list"] = frames_list
        dict_shots_order, df_order, plot_data = compute_feature_score("frames_list", shots, sampling_rate=10)
        fig = score_plot(plot_data)
        st.session_state["ordered_shots"] = dict_shots_order
        col1.plotly_chart(fig, use_container_width=True)
        etime = time.time()
        if not st.session_state["video_ext_timer"]:
            st.session_state["video_ext_timer"] = int(etime - stime)
    if dict_shots_order:
        fig = px.bar(df_order['score'], labels={'index': 'Segment number', 'value': 'Importance'},
                     title='Video segments order')
        fig.update_layout(showlegend=False)
        col2.plotly_chart(fig, use_container_width=True)
        col2.write("Interest score ordered shots:")
        col2.write(dict(itertools.islice(dict_shots_order.items(), 2)))

    with st.spinner("Selecting summary frames"):
        summary_duration = st.session_state["summary_duration"]
        dict_shots_order = st.session_state["ordered_shots"]
        summary_fps = st.session_state["summary_fps"]
        downbeats_frequency = st.session_state["downbeats_frequency"]
        min_shot_nb = len(dict_shots_order)
        summary_frames_index, time_before_drop, new_summary_duration = summary_frames_selection(
            summary_duration, summary_fps, shot_pct_duration, dict_shots_order, min_shot_nb, downbeats_frequency)
        st.session_state["time_before_drop"] = time_before_drop
        st.session_state["summary_frames_index"] = summary_frames_index
        st.session_state["new_summary_duration"] = new_summary_duration
        etime = time.time()
        if not st.session_state["video_ext_timer"]:
            st.session_state["video_ext_timer"] = int(etime - stime)
