import sys
#sys.path.append("../VSUMM")
import os
import scipy.io
import numpy as np
import cv2
import time
from sklearn.cluster import KMeans
import pafy
from sklearn.mixture import GaussianMixture

# System Arguments
# Argument 1: Location of the video
# Argument 2: Sampling rate (every kth frame is chosen)
# Argument 3: Percentage length of video summary
# Argument 4: Video Output path

# frame chosen every k frames
sampling_rate=int(sys.argv[2])

# percentage of video for summary
percent=int(sys.argv[3])

# skim length per chosen frames in second
# which will be adjusted according to the fps of the video
skim_length=1.8

def get_color_hist(frames_raw, num_bins):
    print ("Generating linear Histrograms using OpenCV")
    channels=['b','g','r']

    hist=[]
    for frame in frames_raw:
        feature_value=[cv2.calcHist([frame],[i],None,[int(num_bins)],[0,256]) for i,col in enumerate(channels)]
        #print("feature_value",feature_value)
        hist.append(np.asarray(feature_value).flatten())

    hist=np.asarray(hist)
    #print "Done generating!"
    print ("Shape of histogram: " + str(hist.shape))

    return hist

def frames_selection(frames_list):



    #for streaming video
    #video = pafy.new(url)
    #best = video.getbest(preftype="mp4")

    #vidcap = cv2.VideoCapture(best.url)

    vidcap = cv2.VideoCapture(os.path.abspath(os.path.expanduser(sys.argv[1])))
    success,image = vidcap.read()
    count = 0
    selection=[]
    while success:
        if count in frames_list:
            selection.append(np.asarray(image))
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1
    selection = np.array(selection)
    return selection

def create_video(frames_array,width,height,fps):
    out = cv2.VideoWriter(sys.argv[4]+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (width, height))
    i=0
    for frame in frames_array:
        out.write(frame)
        time.sleep(0.4)
        i+=1

def max_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).max()


def main():
    threshold=70
    global sampling_rate, percent, skim_length
    print ("Opening Video!")

    """ for streaming video
    url="https://youtu.be/Jw-FcDdxSU8"


    video_input = pafy.new(url)
    best = video_input.getbest(preftype="mp4")

    video = cv2.VideoCapture(best.url)
    """
    video=cv2.VideoCapture(os.path.abspath(os.path.expanduser(sys.argv[1])))
    print ("Video opened\nChoosing frames")

    fps=int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    print("fps",fps)
    frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count',frame_count)
    skim_frames_length=fps*skim_length
    start_time=time.time()
    frames = []
    i=0
    while(video.isOpened()):
        if i%sampling_rate==0:
            video.set(1,i)
            ret, frame = video.read()
            if frame is None :
                break
            #im = np.expand_dims(im, axis=0) #convert to (1, width, height, depth)
            # print frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplace = max_of_laplacian(gray) #the sharpness of an image
            if laplace > threshold:
                frames.append(np.asarray(frame))
        i+=1
    video.release()

    frames = np.array(frames)#convert to (num_frames, width, height, depth)
    finish = time.time()-start_time
    print("finish",finish)
    #frames_sampling=frames[0::sampling_rate,:,:,:]
    print("Frames chosen")
    print("Length of video %d" % frames.shape[0])

    # REPLACE WITH APPROPRIATE FEATURES
    features=get_color_hist(frames,16)
    print(features)
    print("Shape of features: " + str(features.shape))

    # clustering: defaults to using the features
    print ("Clustering")

    # Choosing number of centers for clustering
    num_centroids=int(percent*frame_count/skim_frames_length/100)+1
    print("Number of clusters: "+str(num_centroids))

    kmeans=KMeans(n_clusters=num_centroids).fit(features)
    #kmeans=GaussianMixture(n_components=num_centroids).fit(features)
    print("Done Clustering!")

    centres=[]
    features_transform=kmeans.transform(features)
    for cluster in range(features_transform.shape[1]):
        centres.append(np.argmin(features_transform.T[cluster]))

    centres=sorted(centres)
    frames_indices=[]
    for centre in centres:
        print(centre)
        for idx in range(max(int(centre*sampling_rate-skim_frames_length/2),0),min(int(centre*sampling_rate+skim_frames_length/2)+1,frame_count)):
            frames_indices.append(idx)
    frames_indices=sorted(set(frames_indices))

    selection=frames_selection(frames_indices)
    print(selection.shape)
    #print(frames_indices)
    create_video(selection,width,height,fps)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()