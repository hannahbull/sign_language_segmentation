import os
import glob

data_path = '/media/hannah/hdd/lsf/mediapi/mediapi-skel/join_test/'

file_list = sorted(glob.glob(data_path+'*/*_data.pkl', recursive=True))

print(file_list)

for f in file_list: 
    os.rename(f, data_path+f.split('/')[-2]+'_'+f.split('/')[-1])
    print(f, data_path+f.split('/')[-2]+'_'+f.split('/')[-1])