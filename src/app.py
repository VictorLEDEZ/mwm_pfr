import pathlib

import streamlit as st
import streamlit.components.v1 as components
import glob
import importlib
from pathlib import Path
import numpy as np
import os

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True


# if check_password():
PARAM_INIT = {
    "output_path": pathlib.Path(__file__).parent.joinpath("_app_data/output"),
    "ordered_videos": {},
    "frames_list": np.empty(0),
    "videos_param": {},
    "col_histogram": [],
    "audio_path": "",
    "summary_duration": 0,
    "new_summary_duration": 0,
    "time_before_drop": 0,
    "downbeats_frequency": [],
    "downbeat_times": [],
    "shots": np.empty(0),
    "ordered_shots": {},
    "shot_ext_timer": 0,
    "video_ext_timer": 0,
    "audio_ext_time": 0,
    "edit_time": 0,
}

for key, value in PARAM_INIT.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.set_page_config(layout="wide")
page_files = os.listdir(str(Path(__file__).parent.joinpath("_app_pages")))
modules = ["_app_pages." + file.replace(".py", "") for file in page_files if file.endswith("py")]
captions = {module.capitalize().split('-')[-1]: module for module in modules}
st.sidebar.title('Debug 🔧')
st.sidebar.markdown('Session tracker:')
st.sidebar.markdown(f'{list(st.session_state.keys())}')

st.sidebar.title('Navigation 🧭')
choice = st.sidebar.radio(
    "Applet to display",
    list(captions.keys())
)

st.success(f'Currently browsing {choice} 🔍')


st.markdown('')
st.markdown('')
st.markdown('')
current_page = importlib.import_module(captions[choice])
current_page.display()






