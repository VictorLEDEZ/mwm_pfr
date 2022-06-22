import numpy as np
import pandas as pd
import plotly.express as px

def shots_order(shots, agregation, len_cum, display=False):
    
    shots_order = {str(i) : (np.mean(agregation[shots[i]:shots[i+1]]), np.argmax(agregation[shots[i]:shots[i+1]])+shots[i], (shots[i], shots[i+1])) for i in range(0, len(shots)-1)}
    # print("Shots order", shots_order)
    shots_order = dict(sorted(shots_order.items(), key=lambda item: item[1], reverse=True)) # Sorted by mean value for each shots

    d = {}
    d2 = {}

    for  items in shots_order.items():
        for i in range(len(len_cum)):
            if items[1][2][1] > np.append(0, len_cum)[i] and items[1][2][1] <= np.append(0, len_cum)[i+1]:
                d[items[0]] = i, items[1:][0]
    # print("Item", items[1:][0])
    # print("D", d)

    for i in range(len(len_cum)):
        for key, value in d.items():
            if value[0] == i:
                d2[key] = value
                break
    for key, value in d2.items():
        del d[key]

    d2.update(d)

    # print("d", d)

    # for key, value in d2.items():
    #     print(key, ' : ', value)

    shots_order = d2

    df_order = pd.DataFrame.from_dict(shots_order, orient='index')
    # print(df_order[1][0][0])
    # print(df_order.iloc[:,1])
    # df_order["score"]=df_order.iloc[:,1][0][0]
    df_order['score'] = df_order.iloc[:,1].apply(lambda x: x[0])
    # print(df_order.index)
    # shots_order.key = df_order.index
    # shots_order.value = df_order[].values

    key_list=df_order.index
    values_list=df_order.iloc[:,1]
    dictionary = dict(zip(key_list, values_list))
    shots_order = dictionary

    if display == True:
        fig = px.bar(df_order['score'], labels={'index' : 'Segment number', 'value' : 'Importance'}, title = 'Video segments order')
        fig.update_layout(showlegend = False)
        fig.show()

    return shots_order, df_order