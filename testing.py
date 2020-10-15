from utils.dataloader import SkeletonFeeder
from utils.st_gcn import ST_GCN_18
import torch
import pickle 
import numpy as np 
import os
import datetime
import srt 

nth_element = 2
data_path = 'data/00330'
subtitle_path = 'output/subs00330.srt'


### make new dictionary with predictions
predictions = {}
for f in [i for i in os.listdir(data_path) if i.endswith('.pkl')]: 
    predictions[os.path.join(data_path, f)] = np.zeros(int(f.split('_')[1]))

### 

dataset = SkeletonFeeder(data_path = data_path, 
                        nth_element=nth_element, 
                        fps = 25)

print('len dataset ', len(dataset))
meta = dataset.get_metadata()
print(meta)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

graph_cfg = {'layout': 'full', 'strategy': 'spatial'}
model_cfg = {'in_channels': 3, 'num_class': 1, 'edge_importance_weighting': True, 'graph_cfg': graph_cfg}

model = torch.nn.Sequential(ST_GCN_18(**model_cfg))

checkpoint = torch.load('models/full30.pth')

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
        print('meta', meta[n][0][j], meta[n][1][j], meta[n][2][j])
        start_loc = meta[n][1][j]
        end_loc = meta[n][2][j]
        results_extract = val_pred.detach().cpu().numpy()
        if nth_element>1:
            results_extract = np.repeat(results_extract,nth_element)
        predictions[meta[n][0][j]][start_loc:end_loc]=results_extract[loc:(loc+end_loc-start_loc)]
        loc+=end_loc-start_loc
    n+=1



original_fps = 25
original_length_frames = 12370

length_video = datetime.timedelta(seconds=12370/25)
print(length_video)

### THREE modules: 1 -> video to keypoints. 2 -> keypoints to labels. 3 -> labels to original video 

start_SU = []
end_SU = []

### labels to video 
for k in predictions.keys(): 
    starting_time = datetime.timedelta(seconds=int(k.split('/')[-1].split('_')[-3])/original_fps)
    rounded_preds = list(np.round(predictions[k]))
    rounded_preds = np.array([0] + rounded_preds + [0], int)
    differentiate = np.diff(rounded_preds)
    on_switches = np.where(differentiate==1)
    on_switches = [starting_time+datetime.timedelta(seconds=(i-1)/original_fps) for i in on_switches[0]]
    off_switches = np.where(differentiate==-1)
    off_switches = [starting_time+datetime.timedelta(seconds=(i-1)/original_fps) for i in off_switches[0]]

    start_SU.append(on_switches)
    end_SU.append(off_switches)

start_SU = [item for sublist in start_SU for item in sublist]
end_SU = [item for sublist in end_SU for item in sublist]

start_SU = sorted(start_SU)
end_SU = sorted(end_SU)

print(len(start_SU))
print(len(end_SU))

### write Sub file 

### on switches SU times
### off switches SU times 
### match on and off switches

subs = []
n=0
for i in range(len(start_SU)): 
    n+=1
    subs.append(srt.Subtitle(index=n, 
                            start=start_SU[i], 
                            end=end_SU[i], 
                            content='SU', proprietary=''))

f = open(subtitle_path, 'w')
f.writelines(srt.compose(subs))
f.close()