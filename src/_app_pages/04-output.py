import streamlit as st
from video_features.generate_summary import (create_summary,
                                             summary_frames_selection)
from video_features.main import (create_clip, ordering_videos,
                                 read_and_save_frames, summary_param)

import time
import uuid

def display():
    st.title("Output")
    frames_list = st.session_state["frames_list"]
    summary_frames_index = st.session_state["summary_frames_index"]
    summary_resolution = st.session_state["summary_resolution"]
    summary_fps = st.session_state["summary_fps"]
    output_path = st.session_state["output_path"]

    summary_path = output_path.joinpath('summary.mp4')
    audio_sequence_path = output_path.joinpath('audio_sequence.wav')

    clip_path = str(output_path.joinpath('summary_clip_'+str(uuid.uuid4())))
    col1, _ = st.columns(2, )
    with col1:
        st.subheader("Creating video summary")
        if st.button("GO !"):
            with st.spinner("Putting it all together..."):
                stime = time.time()
                create_summary(frames_list, summary_frames_index,
                               summary_path, summary_resolution, summary_fps)

                create_clip(summary_video_path=summary_path,
                            audio_path=audio_sequence_path, clip_filename=clip_path)
                etime = time.time()
                if not st.session_state["edit_time"]:
                    st.session_state["edit_time"] = int(etime - stime)
                with open(clip_path+'.mp4', 'rb') as of:
                    st.balloons()
                    st.video(of)

    st.subheader("Timers")
    _, _, _, col1, col2, col3, _, _ = st.columns(8)
    col1.metric("Video features extraction (s)", st.session_state["video_ext_timer"])
    col2.metric("Audio Features extraction (s)", st.session_state["audio_ext_time"])
    col3.metric("Editing (s)", st.session_state["edit_time"])
