import streamlit as st
import time
from music_features.beat_tracking import get_downbeats
from music_features.config.main import SAMPLING_RATE
from music_features.main import music_features
import plotly.express as px
import numpy as np


@st.cache
def extract_audio_features(audio_path, summary_duration, sampling_rate, time_before_drop, downbeats):
    return music_features(audio_path,
                          summary_duration,
                          sampling_rate,
                          time_before_drop,
                          downbeats,
                          printing=False,
                          plotting=False)


@st.cache
def extract_downbeats(audio_path):
    return get_downbeats(audio_path)


def display():
    st.title("Audio Features")
    new_summary_duration = st.session_state["new_summary_duration"]
    time_before_drop = st.session_state["time_before_drop"]
    audio_path = st.session_state["audio_path"]
    if not new_summary_duration or not audio_path:
        st.write("Audio features require summary duration")
        st.stop()
    if st.button("Extract audio features"):
        stime = time.time()
        downbeats_frequency = st.session_state["downbeats_frequency"]
        downbeat_times = st.session_state["downbeat_times"]

        features = extract_audio_features(audio_path,
                                          new_summary_duration,
                                          SAMPLING_RATE,
                                          time_before_drop,
                                          downbeat_times
                                          )
        etime = time.time()
        if not st.session_state["audio_ext_time"]:
            st.session_state["audio_ext_time"] = int(etime - stime)

        btime, aggregation = features[-2], features[-1]
        beat_start, t_peak, beat_end = features[1], features[2], features[3]
        fig = px.line(x=btime, y=aggregation, labels={"x": "Time in seconds", "y": "Amplitude"})
        fig.update_layout(height=800, width=1200)
        st.write(features[1:4])
        line_colors = ["green", "red", "green"]
        for xv, lc in zip(features[1:4], line_colors):
            fig.add_vline(x=xv, line_dash="dash", line_color=lc)
        st.plotly_chart(fig)


