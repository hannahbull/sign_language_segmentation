from utils.dataloader import SkeletonFeeder
from utils.st_gcn import ST_GCN_18
import torch
import pickle 
import numpy as np 

nth_element = 2
data_path = 'data/00330'
save_metadata = 'output/metadata.pkl'

dataset = SkeletonFeeder(data_path = data_path, 
                        save_metadata=save_metadata, 
                        nth_element=nth_element, 
                        fps = 25)

print('len dataset ', len(dataset))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

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

for xvals in data_loader:
    xvals = xvals.cuda()
    val_pred = model.forward(xvals)
    val_pred = torch.sigmoid(val_pred)
    val_pred = val_pred.view(-1)
    print('val pred ', val_pred)

### frame numbers & labels 
metadata = pickle.load(open('output/metadata.pkl', 'rb'))
print(metadata)

original_fps = 25
original_length_frames = 12370

### make new dictionary with predictions

predictions_video = np.zeros(original_length_frames)

for i in range(len(meta)):
    loc = 0
    for j in range(len(meta[i][0])):
        start_frame = int(meta[i][0][j].split('/')[-1].split('_')[0])
        if (original_fps<33 and original_fps>27):
            start_frame = int(round(start_frame*5/6 - 1/6))
            if fill_every_2nd==False:
                start_frame = max(int(round(start_frame * 5 / 6 - 1 / 6)),0)
        start_loc = meta[i][1][j]
        end_loc = meta[i][2][j]
        results_extract = results[0][i]
        labels_extract = results[1][i]
        if fill_every_2nd==True:
            results_extract = np.repeat(results_extract,2)
            labels_extract = np.repeat(labels_extract, 2)
        predictions_video[video_name][(start_frame+start_loc):(start_frame+end_loc)]=results_extract[loc:(loc+end_loc-start_loc)]

        ### some ob1 errors here
        #print('video id ', video_name)
        #print('results ', results_extract[loc:(loc+end_loc-start_loc)][0:10])
        #print('labels ', list(labels_extract[loc:(loc+end_loc-start_loc)][0:10]))
        #print('labelT ', list(new_true_labs[video_name][(start_frame+start_loc):(start_frame+start_loc+12)]))

        loc+=end_loc-start_loc