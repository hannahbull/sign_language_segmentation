from utils.dataloader import SkeletonFeeder
from utils.st_gcn import ST_GCN_18
from utils.test import return_predictions
import torch
import pickle 
import numpy as np 
import os
import datetime
import srt 
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, help='Path to folder containing .pkl files with skeleton sequences for a single video.', 
                    default='data')
parser.add_argument('--output_folder', type=str, help='Path to location where to save the subtitle .srt file.',
                    default='output')
parser.add_argument('--which_keypoints', type=str, help='Specify "full", "body", "hands", "head", "headbody", "bodyhands". ',
                    default='full')
parser.add_argument('--fps', type=float, default=25., help='Framerate of original video, overwritten by fps_dictionary if not None')
parser.add_argument('--fps_dictionary', type=str, default=None, help='None or path to dictionary with keys as video names and frame rates as values')

args = parser.parse_args()
args_list = vars(args)

### arguments ###
input_folder = args_list['input_folder']
output_folder = args_list['output_folder']
which_keypoints = args_list['which_keypoints']
fps = args_list['fps']
fps_dict = args_list['fps_dictionary']

# fixed arguments
nth_element = 2
converted_fps = 25

predictions = return_predictions(input_folder, which_keypoints)

videos = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

if fps_dict is None: 
    fps_dict = {}
    for v in videos: 
        fps_dict[v] = fps
else: 
    fps_dict = pickle.load(open(fps_dict, 'rb'))

converted_fps_dict = {}
for v in videos: 
    if (fps_dict[v]>27 and fps_dict[v]<33):
        converted_fps_dict[v] = 5/6*fps_dict[v]
    else: 
        converted_fps_dict[v] = fps_dict[v]

### timetags for start and end of subtitle-units
for v in videos: 
    start_SU = []
    end_SU = []

    for k in predictions[v].keys(): 
        starting_time = datetime.timedelta(seconds=int(k.split('/')[-1].split('_')[-3])/fps_dict[v])
        rounded_preds = list(np.round(predictions[v][k]))
        rounded_preds = np.array([0] + rounded_preds + [0], int)
        differentiate = np.diff(rounded_preds)
        on_switches = np.where(differentiate==1)
        on_switches = [starting_time+datetime.timedelta(seconds=(i-1)/converted_fps_dict[v]) for i in on_switches[0]]
        off_switches = np.where(differentiate==-1)
        off_switches = [starting_time+datetime.timedelta(seconds=(i-1)/converted_fps_dict[v]) for i in off_switches[0]]

        start_SU.append(on_switches)
        end_SU.append(off_switches)

    start_SU = [item for sublist in start_SU for item in sublist]
    end_SU = [item for sublist in end_SU for item in sublist]

    start_SU = sorted(start_SU)
    end_SU = sorted(end_SU)

    ### add padding to SUs
    start_SU[0] = min(datetime.timedelta(seconds=0), start_SU[0]-datetime.timedelta(seconds=2.5/25))
    end_SU[-1] = end_SU[0]+datetime.timedelta(seconds=2.5/25)
    for i in range(1, len(start_SU)): 
        start_SU[i] = max(end_SU[i-1], start_SU[i]-datetime.timedelta(seconds=2.5/25)) # max end of previous SU and start of next SU minus padding
        end_SU[i-1] = min(start_SU[i], end_SU[i-1]+datetime.timedelta(seconds=2.5/25)) # min end of previous SU plus padding and start of next SU

    ### write SRT file for subtitles 
    subs = []
    n=0
    for i in range(len(start_SU)): 
        n+=1
        subs.append(srt.Subtitle(index=n, 
                                start=start_SU[i], 
                                end=end_SU[i], 
                                content='SU '+str(n), proprietary=''))

    f = open(os.path.join(output_folder, v+'.srt'), 'w')
    f.writelines(srt.compose(subs))
    f.close()