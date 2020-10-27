from utils.dataloader import SkeletonFeeder
from utils.st_gcn import ST_GCN_18
from utils.test import return_predictions
import numpy as np 
import os
import argparse
import pandas as pd
import sklearn.metrics
import os 

def convert_30_25_fps(vector, scores = None, labels = True):
    if scores is None:
        vector = [vector[0]]+[(5-i%6)/5*vector[i]+(i%6)/5*vector[i+1] for i in range(1,len(vector)-1)]
    else:
        vector = [vector[0]]+[(5-i%6)/5*vector[i]*int(scores[i]!=0)
                              +int(scores[i]!=0)*int(scores[i+1]==0)*(i%6)/5*vector[i]
                                +int(scores[i]==0)*int(scores[i+1]!=0)*(5-i%6)/5*vector[i+1]
                              +(i%6)/5*vector[i+1]*int(scores[i+1]!=0) for i in range(1,len(vector)-1)]
    vector = [vector[i] for i in range(len(vector)) if i % 6 != 5]
    if labels==True:
        vector = [int(round(vector[i])) for i in range(len(vector))]
    return np.array(vector)

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def get_start_finish_sub(string_times, true_fps):
    s = string_times.split(' --> ')
    return [round(get_sec(s[0])*true_fps), round(get_sec(s[1])*true_fps)-1]

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, help='Path to folder containing .pkl files with skeleton sequences for a single video.', 
                    default='mediapiskel_data/skeleton_sequences')
parser.add_argument('--which_keypoints', type=str, help='Specify "full", "body", "hands", "head", "headbody", "bodyhands". ',
                    default='full')
parser.add_argument('--video_information', type=str, default='mediapiskel_data/video_information.csv', help='.csv file containing video names, number of frames and framerates')
parser.add_argument('--subtitle_folder', type=str, default='mediapiskel_data/subtitles', help='Folder containing original subtitles in .vtt format')

args = parser.parse_args()
args_list = vars(args)

### arguments
input_folder = args_list['input_folder']
which_keypoints = args_list['which_keypoints']
video_information = args_list['video_information']
subtitle_folder = args_list['subtitle_folder']

# fixed arguments
nth_element = 2
converted_fps = 25

predictions = return_predictions(input_folder, which_keypoints)

videos = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

video_information = pd.read_csv(video_information, decimal=',')
video_information = video_information.iloc[318:]

video_names = [str(v).zfill(5) for v in video_information['video']]
fps = list(video_information['fps'])
frames = list(video_information['frames'])

fps_dict = {}
frames_dict = {}
for i in range(len(video_names)): 
    fps_dict[video_names[i]] = fps[i]
    frames_dict[video_names[i]] = frames[i]

### true labels 
true_labs_video = dict()

for video_id in videos:

    with open(os.path.join(subtitle_folder, video_id+'.fr.vtt')) as f:
        sub = f.readlines()

    true_fps = float(fps_dict[video_id])

    idx = [i for i in range(len(sub)) if '-->' in sub[i]]
    sub_timing = [sub[i] for i in idx]
    sub_timing = [s.replace('\n', '') for s in sub_timing]

    sub_start_finish = [get_start_finish_sub(s, true_fps=true_fps) for s in sub_timing]

    labels = [list(range(l[0],l[1])) for l in sub_start_finish]
    labels = [item for sublist in labels for item in sublist]
    labels = [1 if i in labels else 0 for i in range(frames_dict[video_id])]

    true_labs_video[video_id] = labels

pad_val = 5

new_true_labs = {}

for v in range(len(videos)):
    vi = videos[v]
    if (fps[v]<33 and fps[v]>27):
        temp = convert_30_25_fps(true_labs_video[vi])
    else:
        temp = true_labs_video[vi]
    if (pad_val>0):
        y_numpy = np.array(temp)
        y_numpy = np.pad(y_numpy, pad_val, mode='edge')
        y_numpy = np.convolve(y_numpy, np.ones(pad_val) / pad_val, mode='same')
        y_numpy = y_numpy[(pad_val):-(pad_val)]
        y_numpy = np.array([int(n > 0.999) for n in y_numpy])
        new_true_labs[vi] = list(y_numpy)
    else:
        new_true_labs[vi] = temp

### make new dictionary with predictions per video
predictions_video = {}

for vi in videos:
    predictions_video[vi] = np.zeros(len(new_true_labs[vi]))
    for k in predictions[vi].keys():
        start_frame = int(k.split('/')[-1].split('_')[0])
        if (fps_dict[vi]<33 and fps_dict[vi]>27):
            start_frame = int(round(start_frame*5/6 - 1/6))
        predictions_video[vi][start_frame:(start_frame+len(predictions[vi][k]))]=predictions[vi][k]

flat_truth = []
flat_preds = []

for v in videos: 
    flat_truth.append(new_true_labs[v])
    flat_preds.append(predictions_video[v])

flat_truth = np.array([item for sublist in flat_truth for item in sublist])
flat_preds = np.array([item for sublist in flat_preds for item in sublist])
flat_preds = np.round(flat_preds)

print('percentage frames incorrect (DTW0)', '%.4f' % np.mean(np.abs(flat_truth-flat_preds)))
print('precision', '%.4f' % sklearn.metrics.precision_score(1-flat_truth, 1-flat_preds))
print('recall','%.4f' % sklearn.metrics.recall_score(1-flat_truth, 1-flat_preds))
print('f1', '%.4f' % sklearn.metrics.f1_score(1-flat_truth, 1-flat_preds))

