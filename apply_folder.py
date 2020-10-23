import os 
import pandas as pd

### category 
video_information = pd.read_csv('/localHD/data_corpus/mediapi-skel/video_information.csv', decimal=',')
videos = [str(v).zfill(5) for v in video_information['video']]
fps = list(video_information['fps'])

for i in range(318,368):
    DIR = videos[i] 
    print(DIR)
    FPS = str(fps[i])
    cmd = 'python apply_model.py --input_folder /localHD/data_corpus/mediapi-skel/clean_op_data_test/'+DIR+' --output_path output/'+DIR+'.srt --fps '+FPS+' --which_keypoints "body"'
    print(cmd)
    os.system(cmd)


### START STOP 
