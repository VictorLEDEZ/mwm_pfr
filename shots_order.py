from dis import dis
import numpy as np
import pandas as pd
import plotly.express as px

def shots_order(shots, agregation, display=False):
    shots_order = {str(i) : (np.mean(agregation[shots[i]:shots[i+1]]), np.argmax(agregation[shots[i]:shots[i+1]])+shots[i], (shots[i], shots[i+1])) for i in range(0, len(shots)-1)}
    shots_order = dict(sorted(shots_order.items(), key=lambda item: item[1], reverse=True)) # Sorted by mean value for each shots

    df_order = pd.DataFrame.from_dict(shots_order, orient='index')

    if display == True:
        fig = px.bar(df_order[0], labels={'index' : 'Segment number', 'value' : 'Importance'}, title = 'Video segments order')
        fig.update_layout(showlegend = False)
        fig.show()

    return shots_order, df_order