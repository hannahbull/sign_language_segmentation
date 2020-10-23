from utils.dataloader import SkeletonFeeder
from utils.st_gcn import ST_GCN_18
import torch
import pickle 
import numpy as np 
import os
import datetime
import srt 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, help='Path to folder containing .pkl files with skeleton sequences for a single video.', 
                    default='data/00330')
parser.add_argument('--output_path', type=str, help='Path to location where to save the subtitle .srt file.',
                    default='output/subtitles.srt')
parser.add_argument('--which_keypoints', type=str, help='Specify "full", "body", "hands", "head", "headbody", "bodyhands". ',
                    default='full')
parser.add_argument('--fps', type=float, default=25., help='Framerate of original video')

args = parser.parse_args()
args_list = vars(args)

### arguments ###
input_folder = args_list['input_folder']
output_path = args_list['output_path']
which_keypoints = args_list['which_keypoints']
fps = args_list['fps']

# fixed arguments
nth_element = 2
converted_fps = 25

### make new dictionary with predictions
predictions = {}
for f in [i for i in os.listdir(input_folder) if i.endswith('.pkl')]: 
    predictions[os.path.join(input_folder, f)] = np.zeros(int(f.split('_')[1]))

dataset = SkeletonFeeder(data_path = input_folder, 
                        nth_element=nth_element, 
                        fps = converted_fps, 
                        body_type=which_keypoints)

print('length test dataset ', len(dataset))

meta = dataset.get_metadata()

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

graph_cfg = {'layout': which_keypoints, 'strategy': 'spatial'}
model_cfg = {'in_channels': 3, 'num_class': 1, 'edge_importance_weighting': True, 'graph_cfg': graph_cfg}

### load model
model = torch.nn.Sequential(ST_GCN_18(**model_cfg))

checkpoint = torch.load('models/'+which_keypoints+'30.pth')

state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    name = '0.'+k 
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()
model.zero_grad()

n = 0
for xvals in data_loader:
    with torch.no_grad():
        xvals = xvals.cuda()
        val_pred = model.forward(xvals)
        val_pred = torch.sigmoid(val_pred)
        val_pred = val_pred.view(-1)
    loc = 0
    for j in range(len(meta[n][0])): 
        start_loc = meta[n][1][j]
        end_loc = meta[n][2][j]
        results_extract = val_pred.detach().cpu().numpy()
        if nth_element>1:
            results_extract = np.repeat(results_extract,nth_element)
        predictions[meta[n][0][j]][start_loc:end_loc]=results_extract[loc:(loc+end_loc-start_loc)]
        loc+=end_loc-start_loc
    n+=1


### timetags for start and end of subtitle-units
start_SU = []
end_SU = []

for k in predictions.keys(): 
    starting_time = datetime.timedelta(seconds=int(k.split('/')[-1].split('_')[-3])/fps)
    rounded_preds = list(np.round(predictions[k]))
    rounded_preds = np.array([0] + rounded_preds + [0], int)
    differentiate = np.diff(rounded_preds)
    on_switches = np.where(differentiate==1)
    on_switches = [starting_time+datetime.timedelta(seconds=(i-1)/converted_fps) for i in on_switches[0]]
    off_switches = np.where(differentiate==-1)
    off_switches = [starting_time+datetime.timedelta(seconds=(i-1)/converted_fps) for i in off_switches[0]]

    start_SU.append(on_switches)
    end_SU.append(off_switches)

start_SU = [item for sublist in start_SU for item in sublist]
end_SU = [item for sublist in end_SU for item in sublist]

start_SU = sorted(start_SU)
end_SU = sorted(end_SU)

### write SRT file for subtitles 
subs = []
n=0
for i in range(len(start_SU)): 
    n+=1
    subs.append(srt.Subtitle(index=n, 
                            start=start_SU[i], 
                            end=end_SU[i], 
                            content='SU', proprietary=''))

f = open(output_path, 'w')
f.writelines(srt.compose(subs))
f.close()