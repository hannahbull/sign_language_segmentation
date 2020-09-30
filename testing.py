from utils.dataloader import SkeletonFeeder
from utils.st_gcn import ST_GCN_18
import torch

dataset = SkeletonFeeder(data_path = 'data/')

print('len dataset ', len(dataset))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

graph_cfg = {'layout': 'full', 'strategy': 'spatial'}
model_cfg = {'in_channels': 3, 'num_class': 1, 'edge_importance_weighting': True, 'graph_cfg': graph_cfg}

model = torch.nn.Sequential(ST_GCN_18(**model_cfg))

checkpoint = torch.load('models/full/full30.pth')

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

