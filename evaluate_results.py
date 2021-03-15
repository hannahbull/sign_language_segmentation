import matplotlib.pyplot as plt 
import numpy as np 
import os
import cv2 
import sklearn.metrics


video_id = ['countryfile/6040895554095723514', 'natural+world/6130095152885987590', 'natural+world/6221289334618958863']

vid = video_id[2]

location_sub = '/media/hannah/hdd/lsf/bbcsl/subtitles'
#location_sub = '/media/hannah/hdd/lsf/bbcsl/realigned_subs'

location_probs = '/media/hannah/hdd/lsf/bbcsl/predicted_SU_probs'
location_vid = '/media/hannah/hdd/lsf/bbcsl/videos'

cap = cv2.VideoCapture(os.path.join(location_vid, vid, 'signhd-dense-fast-audio.mp4'))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

SU_probs = np.load(os.path.join(location_probs, vid, 'SU_probs.npy')) 
if (frame_count>len(SU_probs)):
    SU_probs = np.concatenate([SU_probs, np.zeros(frame_count - len(SU_probs))])
else: 
    SU_probs = SU_probs[0:frame_count]
    
    
np.save(os.path.join(location_probs, vid, 'SU_probs.npy'), SU_probs)

### TRUE subs 
def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def get_start_finish_sub(string_times, true_fps):
    s = string_times.split(' --> ')
    return [round(get_sec(s[0])*true_fps), round(get_sec(s[1])*true_fps)-1]


with open(os.path.join(location_sub, vid, 'signhd.vtt')) as f:
    sub = f.readlines()

idx = [i for i in range(len(sub)) if '-->' in sub[i]]
sub_timing = [sub[i] for i in idx]
sub_timing = [s.replace('\n', '') for s in sub_timing]

sub_start_finish = [get_start_finish_sub(s, true_fps=25) for s in sub_timing]

labels = [list(range(l[0],l[1])) for l in sub_start_finish]
labels = [item for sublist in labels for item in sublist]
labels = [1 if i in labels else 0 for i in range(frame_count)]

pad_val = 5
temp = labels
if (pad_val>0):
    y_numpy = np.array(temp)
    y_numpy = np.pad(y_numpy, pad_val, mode='edge')
    y_numpy = np.convolve(y_numpy, np.ones(pad_val) / pad_val, mode='same')
    y_numpy = y_numpy[(pad_val):-(pad_val)]
    y_numpy = np.array([int(n > 0.999) for n in y_numpy])
    new_true_labs = y_numpy
else:
    new_true_labs = np.array(temp)

print(new_true_labs, new_true_labs.shape)

flat_preds = np.array([int(round(SU_probs[i])) for i in range(len(SU_probs))])
flat_truth = new_true_labs

print('percentage frames incorrect (DTW0)', '%.4f' % np.mean(np.abs(flat_truth-flat_preds)))
print('precision', '%.4f' % sklearn.metrics.precision_score(1-flat_truth, 1-flat_preds))
print('recall','%.4f' % sklearn.metrics.recall_score(1-flat_truth, 1-flat_preds))
print('f1', '%.4f' % sklearn.metrics.f1_score(1-flat_truth, 1-flat_preds))


nox=2051
start = 2050
origin = 0

plt.ioff()

labs = list(flat_truth)
res = list(SU_probs)
res_dis = [int(round(r)) for r in res]


low = start
high = low+nox

x = 1-np.array(labs[low:high])
xdis = 1-np.array(res_dis[low:high])


barprops = dict(aspect = 'auto', cmap = 'binary', interpolation='nearest')

fig = plt.figure(figsize=(30,10))

ax1 = fig.add_axes([0.1,0.5,0.7,0.2])
ax1.plot(np.arange(low, high), labs[low:high], '--', label='True')
ax1.plot(np.arange(low, high), res[low:high], label='Pred.')
# ax1.axvline(x=low+25, color = 'red')
ax1.set_xlim((low,high))
ax1.set_ylim((0,1))
ax1.axes.xaxis.set_ticklabels([])
ax1.tick_params(axis='both', which='both', bottom=False)
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)

#ax1.set_axis_off()

ax2 = fig.add_axes([0.1,0.4,0.7,0.05])
ax2.tick_params(axis='both', which='both', bottom=False,left=False)
ax2.axes.xaxis.set_ticklabels([])
ax2.axes.yaxis.set_ticklabels([])
ax2.axes.set_ylabel('Pred.', rotation=0, labelpad=10, verticalalignment='center')
ax2.imshow(xdis.reshape((1,-1)),**barprops)
#ax2.axvline(x=24.8, color = 'red')

ax3 = fig.add_axes([0.1,0.3,0.7,0.05])
ax3.tick_params(axis='both', which='both', bottom=False, left=False)
ax3.axes.xaxis.set_ticklabels(np.arange(low-10-origin,high+21-origin,10))
ax3.axes.yaxis.set_ticklabels([])
ax3.axes.set_ylabel('True', rotation=0, labelpad=10, verticalalignment='center')
ax3.imshow(x.reshape((1,-1)),**barprops)
#ax3.axvline(x=24.8, color = 'red')


# ax4 = fig.add_axes([0.1,0.75,0.7,0.2])
# ax4.tick_params(axis='both', which='both', bottom=False, left=False)
# #ax3.axes.xaxis.set_ticklabels([])
# ax4.axes.yaxis.set_ticklabels([])
# ax4.axes.set_ylabel('True', rotation=0, labelpad=10, verticalalignment='center')
# ax4.imshow(x.reshape((1,-1)),**barprops)

plt.tick_params(axis='both', which='both', bottom=False, left=False)

plt.savefig('/media/hannah/hdd/lsf/bbcsl/'+str(start).zfill(5)+'.png')
plt.cla()
plt.close()
