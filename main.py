from SIFT import Sift
from Flow import Flow
from object_det_score import object_det_score
from shot_detection import define_shots
from shots_order import shots_order

import os
from pathlib import Path
import numpy as np  # Change obj to np array (to remove)

# Video Path
data_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), "Data")
video_path = os.path.join(data_path, "cut.mp4")

# Run
sift = Sift(video_path, display = False, save_video = False)
flow = Flow(video_path, display = False, save_video = False)
obj, obj_score = object_det_score(video_path, gpu=1)

flow = np.append(flow, 0)  # DIFFERENT SHAPE !! (to remove)
sift = np.append(sift, 0)  # DIFFERENT SHAPE !! (to remove)

def agregation(flow=flow, sift=sift, obj=obj_score, flow_coef = 1, sift_coef=1, obj_coef = 1):
    agreg = flow_coef*flow + sift_coef*sift + obj_coef*obj_score
    return agreg

agreg = agregation(flow=flow, sift=sift, obj=obj_score, flow_coef=1, sift_coef=1, obj_coef=1)
shots = define_shots(video_path, nb_shots=10)
shots_order, df_order = shots_order(shots, agregation, display=True)

### To remove ###
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

frame_number = [i for i in range(0, len(flow))]

fig = make_subplots(rows=4, cols=1, subplot_titles=('Objects', 'Optical Flow', 'SIFT', 'Agregation'))

fig.append_trace(go.Scatter(
    x=frame_number,
    y=obj_score,
), row=1, col=1)

fig.append_trace(go.Scatter(
    x=frame_number,
    y=flow,
), row=2, col=1)

fig.append_trace(go.Scatter(
    x=frame_number,
    y=sift,
), row=3, col=1)

fig.append_trace(go.Scatter(
    x=frame_number,
    y=agregation(flow=flow, sift=sift, obj=obj_score, flow_coef=1, sift_coef=1, obj_coef=1),
), row=4, col=1)

fig.update_layout(height=800, width=1000, showlegend=False) # title_text="Stacked Subplots"
fig.show()
#################