import streamlit as st
import collections
from itertools import cycle
import os
import datetime


def display():
    st.title("Context")
    col, _, _ = st.columns(3)
    col.header("Summary duration")
    summary_duration = col.slider('Summary duration', 1, 50, 30)
    if summary_duration:
        st.session_state["summary_duration"] = summary_duration
    with col:
        st.header("Video File Selection")
        uploaded_videos = st.file_uploader("Choose video files", accept_multiple_files=True)
    if uploaded_videos:
        ordered_videos = {}
        with st.spinner("Saving video files..."):
            for uploaded_video in uploaded_videos:
                video_path = os.path.join("_app_data", "videos", f"r_{uploaded_video.name}")
                ordered_videos[video_path] = uploaded_video.name.split("_")[0]
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
            ordered_videos = dict(sorted(ordered_videos.items(), key=lambda item: item[1]))
            st.write(ordered_videos)
        st.session_state["ordered_videos"] = ordered_videos

    if st.session_state["ordered_videos"]:
        session_videos = st.session_state["ordered_videos"]
        with st.expander("Video Files", expanded=True):
            n_columns = min(3, len(session_videos))
            columns = st.columns(n_columns)
            for video_file, col in zip(session_videos.keys(), cycle(columns)):
                with col:
                    st.subheader(video_file)
                    with open(video_file, "rb") as vf:
                        st.video(vf.read(), start_time=0)

    st.header("Audio file selection")
    col, _, _ = st.columns(3)
    with col:
        uploaded_audio = st.file_uploader("Choose audio file")
        if uploaded_audio:
            with st.spinner("Saving audio file..."):
                audio_path = os.path.join("_app_data", "audio", f"r_{uploaded_audio.name}")
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                st.session_state["audio_path"] = audio_path
        audio_path = st.session_state["audio_path"]
        if audio_path:
            with open(audio_path, "rb") as af:
                st.audio(af.read())
