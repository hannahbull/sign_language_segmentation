### START STOP 
import os
import pandas as pd
import pickle
import numpy as np

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

video_information = pd.read_csv('/media/hannah/hdd/lsf/mediapi/mediapi-skel/video_information.csv', decimal=',')
video_information = video_information.iloc[318:]
subtitle_folder = 'output/'

output_location = 'subtitle_unit_predictions.pkl'

def get_sec(time_str):
    h, m, s = time_str.split(':')
    s = s.replace(',', '.')
    return int(h) * 3600 + int(m) * 60 + float(s)

def get_start_finish_sub(string_times, true_fps):
    s = string_times.split(' --> ')
    return [round(get_sec(s[0])*true_fps), round(get_sec(s[1])*true_fps)-1]

videos = [str(v).zfill(5) for v in video_information['video']]
frames = list(video_information['frames'])
fps = list(video_information['fps'])

length_sub_words = []
length_sub_time = []
subs_vector = []

labels_dict = dict()

for v in range(len(videos)):
    video_id = videos[v]
    print(video_id)

    with open(subtitle_folder+video_id+'.srt') as f:
        sub = f.readlines()

    true_fps = float(fps[v])

    idx = [i for i in range(len(sub)) if '-->' in sub[i]]
    sub_timing = [sub[i] for i in idx]
    sub_timing = [s.replace('\n', '') for s in sub_timing]

    sub_start_finish = [get_start_finish_sub(s, true_fps=true_fps) for s in sub_timing]

    labels = [list(range(l[0],l[1])) for l in sub_start_finish]
    labels = [item for sublist in labels for item in sublist]
    labels = [1 if i in labels else 0 for i in range(frames[v])]

    labels_dict[video_id] = labels

pickle.dump(labels_dict, open(output_location, 'wb'))

### true vals

true_labs_video = pickle.load(open('/media/hannah/hdd/lsf/mediapi/mediapi-skel/subtitle_unit_labels.pkl', 'rb'))

pad_val = 5

new_true_labs = {}

for v in range(len(videos)):
    vi = videos[v]
    #print(vi)
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

flat_truth = []
flat_preds = []

for v in videos: 
    flat_truth.append(new_true_labs[v])
    flat_preds.append(labels_dict[v])

flat_truth = np.array([item for sublist in flat_truth for item in sublist])
flat_preds = np.array([item for sublist in flat_preds for item in sublist])

print(len(flat_truth), len(flat_preds))
print(np.mean(np.abs(flat_truth-flat_preds)))

import sklearn.metrics

print('overall precision', '%.4f' % sklearn.metrics.precision_score(1-flat_truth, 1-flat_preds))
print('overall recall','%.4f' % sklearn.metrics.recall_score(1-flat_truth, 1-flat_preds))
print('overall f1', '%.4f' % sklearn.metrics.f1_score(1-flat_truth, 1-flat_preds))

