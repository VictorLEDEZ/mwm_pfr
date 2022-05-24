from SIFT import Sift
from Flow import Flow
import os
from pathlib import Path

# Video Path
data_path = os.path.join(Path(os.getcwd()).parent.absolute(), "Data")
video_path = os.path.join(data_path, "cut.mp4")

# Run
sift = Sift(video_path, display = False, save_video = False)
flow = Flow(video_path, display = False, save_video = False)

def agregation(flow=flow, sift=sift, flow_coef=1, sift_coef=1):
    agreg = flow_coef * flow + sift_coef * sift
    return agreg

agreg = agregation(flow, sift, flow_coef=1, sift_coef=1)

### To remove ###
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

frame_number = [i for i in range(0, len(sift))]

fig = make_subplots(rows=3, cols=1, subplot_titles=('SIFT', 'Optical Flow', 'Agregation'))

fig.append_trace(go.Scatter(
    x=frame_number,
    y=sift,
), row=1, col=1)

fig.append_trace(go.Scatter(
    x=frame_number,
    y=flow,
), row=2, col=1)

fig.append_trace(go.Scatter(
    x=frame_number,
    y=agreg,
), row=3, col=1)


fig.update_layout(height=800, width=1000, showlegend=False) # title_text="Stacked Subplots"
fig.show()
#################