import cv2
import numpy as np
import sys
import pandas as pd
import plotly.express as px


# System Arguments
# Argument 1: Location of the video
# Argument 2: Nb slots (int)
# Argument 3: Show visualization (True/False)

def color_hist(file_path):
    cap = cv2.VideoCapture(file_path)
    fps=int(cap.get(cv2.CAP_PROP_FPS))

    previous_hist=np.zeros((8,8,8))
    list_diff=[]
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==True:
            hist = cv2.calcHist([frame], [0, 1, 2],None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            b=np.sum(np.abs(hist-previous_hist))
            list_diff.append(b)

            previous_hist=hist
        else:
            cap.release()
    return list_diff,fps


def visualization(hist_diff,shots):
    df=pd.DataFrame()
    x=np.arange(1,len(hist_diff))
    df["x"]=x
    df["hist"]=hist_diff[1:]
    df["color"]=["shot detection" if i in shots else "frames" for i in x]

    fig = px.bar(df,x="x", y="hist", color="color")
    fig.update_traces(dict(marker_line_width=0))
    fig.show()

def define_shots(file_path,nb_shots,range_min=1,show_viz=False):

    hist_diff,fps=color_hist(file_path)

    n = len(hist_diff)
    ranked = np.argsort(hist_diff)
    largest_indices = ranked[::-1][1:n]

    shots_index=[0,max(largest_indices)]
    for i,val in enumerate(largest_indices):

        valid=False
        for j in shots_index:
            if np.abs(j-val)>range_min*fps:
                valid=True
            else:
                valid=False
                break
        if valid==True:
            shots_index.append(val)

    shots=np.sort(shots_index[:nb_shots+1])

    if show_viz==True:
        visualization(hist_diff,shots)

    return shots





if __name__ == '__main__':
    file_path=sys.argv[1]
    nb_shots=int(sys.argv[2])
    range_min=int(sys.argv[3])
    show_viz=bool(sys.argv[4])

    shots=define_shots(file_path,nb_shots,range_min,show_viz)
    print(shots)

