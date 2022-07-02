#installer la version tensorflow-gpu==1.15.0rc2 pour utiliser la version darkflow de YOLO avec un GPU
import tensorflow as tf
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet


def thema(res):
    '''
    fonction qui remplace le label de l'objet détecté par un thème
    res : résultat (dict) du modèle YOLO
    return : résultat modifié (dict) avec le thème au lieu de l'objet détecté
    '''

    #liste des thèmes en fonction des objets provenant de la base de données COCO
    list_person = ['person']
    list_vehicle = ['bus', 'bicycle', 'motorcycle', 'car', 'airplane', 'train', 'boat', 'truck']
    list_outdoor = ['traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench']
    list_animal = ['bird', 'cat', 'horse', 'dog', 'sheep', 'cow', 'bear', 'elephant', 'zebra', 'giraffe']
    list_accessory = ['hat', 'backpack', 'handbag', 'suitcase', 'umbrella', 'shoe', 'eye glasses', 'tie']
    list_sports = ['frisbee', 'skis', 'sports ball', 'kite', 'baseball bat', 'skateboard', 'baseball glove', 'surfboard', 'tennis racket', 'snowboard']
    list_kitchen = ['bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl']
    list_food = ['banana', 'apple', 'orange', 'sandwich', 'broccoli', 'carrot', 'pizza', 'hot dog', 'donut', 'cake']
    list_furniture = ['chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door']
    list_electronic = ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone']
    list_appliance = ['microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender']
    list_indoor = ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

    #remplacement de l'objet par le thème
    for r in res:
        if r['label'] in list_person:
            r['label'] = 'person'
        elif r['label'] in list_vehicle:
            r['label'] = 'vehicle'
        elif r['label'] in list_outdoor:
            r['label'] = 'outdoor'
        elif r['label'] in list_animal:
            r['label'] = 'animal'
        elif r['label'] in list_accessory:
            r['label'] = 'accessory'
        elif r['label'] in list_sports:
            r['label'] = 'sports'
        elif r['label'] in list_kitchen:
            r['label'] = 'kitchen'
        elif r['label'] in list_food:
            r['label'] = 'food'
        elif r['label'] in list_furniture:
            r['label'] = 'furniture'
        elif r['label'] in list_electronic:
            r['label'] = 'electronic'
        elif r['label'] in list_appliance:
            r['label'] = 'appliance'
        elif r['label'] in list_indoor:
            r['label'] = 'indoor'

    return res


def frame_score(res, row, col):
    '''
    fonction qui renvoie le score d'une frame d'une vidéo à partir des objets détectés dans la frame
    res : résultat modifié (dict) du modèle YOLO
    row x col : dimension de la frame (int x int)
    return : score associé à la frame (float)
    '''
    tmp_score = []

    #définition d'un score par rapport au thème de l'objet
    for r in res:
        if r['label'] == 'person':
            t_score = 0.95
        elif r['label'] == 'animal':
            t_score = 0.8
        elif r['label'] == 'sports':
            t_score = 0.7
        elif r['label'] == 'food':
            t_score = 0.6
        elif r['label'] == 'accessory':
            r['label'] = 0.5
        elif r['label'] == 'vehicle':
            t_score = 0.3
        else:
            t_score = 0
        
        #score de détection de l'objet
        c_score = r['confidence']
        
        #proportion de l'objet dans la frame
        area_score = (abs(r['bottomright']['y'] - r['topleft']['y']) * abs(r['bottomright']['x'] - r['topleft']['x'])) / (row * col)
        
        #score pour un objet de la frame
        tmp_score.append(t_score * c_score * area_score)
    
    #score de la frame correspondant à la somme des scores des objets de la frame
    score = sum(tmp_score)

    return score


# def object_det_score(video_path, model_path="cfg/yolo.cfg", weights_path="bin/yolo.weights", thresold_pred=0.6, gpu=0):
#     '''
#     fonction qui renvoie le score pour toutes les frames d'une vidéo
#     video_path : chemin de la vidéo à analyser (string)
#     model_path : chemin vers le modèle de détection d'objets YOLO (string)
#     weights_path : chemin vers les poids du modèle (string)
#     treshold_pred : seuil de confiance au delà duquel l'objet est détecté (float compris entre 0 et 1)
#     gpu : valeur égale à 1 si le modèle utilise un GPU sinon 0 (float)
#     return : une liste correspondant aux résultats de la détection d'objets dans chaque frame de la vidéo 
#     et une liste correspondant aux scores dans chaque frame de la vidéo
#     '''
#     options = {"model": model_path, 
#            "load": weights_path, 
#            "threshold": thresold_pred,
#            'gpu': gpu}

#     frame_scores = []
#     results_object_det = []

#     tfnet = TFNet(options)
#     cap = cv2.VideoCapture(video_path)

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret:
#             row = frame.shape[0]
#             col = frame.shape[1]
#             results = tfnet.return_predict(frame)
#             new_results = thema(results)
#             frame_scores.append(frame_score(new_results, row, col))
#             results_object_det.append(new_results)
#             if cv2.waitKey(1)  & 0xFF == ord('q'):
#                 break
#         else:
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

#     normed_scores = [float(i)/max(frame_scores) for i in frame_scores]

#     return results_object_det, normed_scores


def object_det_score(frame_list, model_path="cfg/yolo.cfg", weights_path="bin/yolo.weights", config_path="./cfg/", thresold_pred=0.6, gpu=0):
    '''
    fonction qui renvoie le score pour toutes les frames d'une vidéo
    frame_list: liste de frames (list)
    model_path : chemin vers le modèle de détection d'objets YOLO (string)
    weights_path : chemin vers les poids du modèle (string)
    treshold_pred : seuil de confiance au delà duquel l'objet est détecté (float compris entre 0 et 1)
    gpu : valeur égale à 1 si le modèle utilise un GPU sinon 0 (float)
    return : une liste correspondant aux résultats de la détection d'objets dans chaque frame de la vidéo 
    et une liste correspondant aux scores dans chaque frame de la vidéo
    '''
    options = {
        "model": model_path,
        "load": weights_path,
        "config": config_path,
        "threshold": thresold_pred,
        'gpu': gpu}

    frame_scores = []
    results_object_det = []

    tfnet = TFNet(options)
    # frame_list = list(itertools.chain(*frame_list)) 

    for frame in frame_list:
        row = frame.shape[0]
        col = frame.shape[1]
        results = tfnet.return_predict(frame)
        new_results = thema(results)
        frame_scores.append(frame_score(new_results, row, col))
        results_object_det.append(new_results)

    # normed_scores = [float(i)/max(frame_scores) for i in frame_scores]

    return results_object_det, frame_scores


def main():
    if len(sys.argv) != 2:
        print('usage: ./object_det_score.py file')
        sys.exit(1)

    filename = sys.argv[1]

    scores = object_det_score(
        filename,
        model_path=str(DARKFLOW_PATH / "cfg/yolo.cfg"),
        weights_path=str(DARKFLOW_PATH / "bin/yolo.weights"),
        config_path=str(DARKFLOW_PATH / "cfg"),
    )

    return scores

if __name__ == '__main__':
    scores = main()
    
    fig = plt.figure(figsize=(12,6))

    plt.plot(scores[1])

    plt.xlabel('Frames')
    plt.ylabel("Score détection d'objets")
    plt.grid(True)

    plt.show(block=True)