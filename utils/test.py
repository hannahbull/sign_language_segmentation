from utils.dataloader import SkeletonFeeder
from utils.st_gcn import ST_GCN_18
import torch
import numpy as np 
import os

def return_predictions(input_folder, which_keypoints):
    ''' 
    Args 
    - input_folder: folder containing one subfolder per video. Each video folder contains .pkl files of skeleton keypoint sequences which are the output of clean_op_data_sl
    - which_keypoints: choose to use full, body, body+hands, body+head, head or hands for the predictions. Specify "full", "body", "hands", "head", "headbody", "bodyhands". 
    Returns
    - dictionary object. Keys are video names. Predictions[video_name] is a dictionary where keys are the .pkl file paths. Predictions[video_name][*.pkl] contains a vector 
    of the result probabilities. The length of this vector is the length of the skeleton keypoint sequence of the .pkl file. 
    '''
    # fixed arguments
    nth_element = 2 ### our model is trainined taking every second element 
    converted_fps = 25 ### when we run clean_op_data_sl, we convert all the keypoints to 25 fps

    ### make new dictionary with predictions
    predictions = {}
    videos = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    print('Video names ', videos)
    for v in videos: 
        video_segments = [os.path.join(input_folder, v, i) for i in os.listdir(os.path.join(input_folder, v)) if i.endswith('_data.pkl')]
        predictions[v] = {}
        for s in video_segments: 
            predictions[v][s] = np.zeros(int(s.split('_')[-2]))

    dataset = SkeletonFeeder(data_path = input_folder, 
                            nth_element=nth_element, 
                            fps = converted_fps, 
                            body_type=which_keypoints)

    print('Length dataset ', len(dataset))

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
                results_extract = np.repeat(results_extract, nth_element)
            predictions[meta[n][0][j].split('/')[-2]][meta[n][0][j]][start_loc:end_loc]=results_extract[loc:(loc+end_loc-start_loc)]
            ## predictions[video_name][clean_op_data name][start:end]
            loc+=end_loc-start_loc
        n+=1

    return predictions 