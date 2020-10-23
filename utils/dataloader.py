import numpy as np
import pickle
import torch
import glob
import re
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

class SkeletonFeeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to the folder containing the skeleton sequences
        nth_element: take every nth frame 
        sequence_length: temporal length of sequences
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """
    def __init__(self,
                 data_path, 
                 nth_element = 2, 
                 sequence_length = 125, 
                 random_shuffle = False,
                 random_flip = False, 
                 normalise = True, 
                 visualise = False, 
                 body_type = 'full', 
                 fps = 25):

        self.nth_element = nth_element
        self.data_path = data_path
        self.sequence_length = sequence_length*self.nth_element
        self.random_shuffle = random_shuffle
        self.random_flip = random_flip
        self.normalise = normalise
        self.visualise = visualise
        self.body_type = body_type
        self.fps = fps

        ### get all files ending in *_data.pkl
        file_list = np.array(sorted([os.path.join(self.data_path,i) for i in os.listdir(self.data_path) if i.endswith('.pkl')])) 

        if self.random_shuffle==True:
            np.random.shuffle(file_list)

        ### get list of video names, start frame and length for each file in file_list
        file_df = [re.sub('/*_data.pkl', '', a) for a in file_list]
        file_df = [re.sub(self.data_path, '', a) for a in file_df]
        file_df = [re.sub('/', '_', a) for a in file_df]
        file_df = [a.split('_') for a in file_df]
        file_df = [[file_list[i], int(file_df[i][1]), int(file_df[i][2])] for i in range(len(file_df))] # file name, start frame, length

        ### make sequences of 200 frames 
        list_sequences = []
        temp_length = 0
        next_video = file_df[0][0]
        next_start = 0
        next_sequence_length = self.sequence_length
        i = 0
        end_while = 0
        while (end_while==0):  
            videos = []
            starts = []
            stops = []
            while (temp_length < self.sequence_length):
                videos.append(next_video)
                starts.append(next_start)
                stop_val = min(next_start+next_sequence_length, file_df[i][2])
                stops.append(stop_val)
                temp_length += stop_val-next_start
                if (stop_val == file_df[i][2] and temp_length<self.sequence_length):
                    if i+1 < len(file_df):
                        i+=1
                    else:
                        i=0
                        end_while=1
                    next_sequence_length = self.sequence_length-temp_length
                    next_video = file_df[i][0]
                    next_start = 0
                elif (stop_val == file_df[i][2] and temp_length==self.sequence_length):
                    if i+1 < len(file_df):
                        i+=1
                    else:
                        i=0
                        end_while=1
                    next_sequence_length = self.sequence_length
                    next_video = file_df[i][0]
                    next_start = 0
                else:
                    next_start = stop_val
                    next_sequence_length = self.sequence_length

            list_sequences.append([videos, starts, stops])
            temp_length = 0
    
        if self.random_shuffle==True:
            random.shuffle(list_sequences)

        self.list_sequences = list_sequences  
        self.get_metadata()


    def __len__(self):
        return len(self.list_sequences)

    def get_metadata(self): 
        return self.list_sequences

    def __getitem__(self, index):
        
        # get data
        list_sequences = self.list_sequences

        x_numpy = []
        for a in range(len(list_sequences[index][0])):
            data = pickle.load(open(list_sequences[index][0][a], 'rb'))
            temp_data = data[2][list_sequences[index][1][a]:list_sequences[index][2][a]]

            ### get most likely signer
            if (np.max(temp_data[:,2,:,1])>0):
                temp_data = get_main_signer(temp_data, self.fps)
            else: 
                temp_data = temp_data[:,:,:,0][:,:,:,np.newaxis]

            ### which skeleton keypoints to include? 
            if self.body_type == 'headbody': 
                temp_data = temp_data[:,:,0:85,:]
            elif self.body_type == 'head': 
                temp_data = temp_data[:,:,0:70,:]
            elif self.body_type == 'body': 
                temp_data = temp_data[:,:,70:85,:]
            elif self.body_type == 'hands': 
                temp_data = temp_data[:,:,85:,:]
            elif self.body_type == 'bodyhands': 
                temp_data = temp_data[:,:,70:,:]
            elif self.body_type == 'full': 
                pass
            else: 
                print('please select full/headbody/head/body/hands/bodyhands')

            #### normalise
            if self.normalise==True: 
                for m in range(temp_data.shape[-1]):
                    mean_x = np.mean(temp_data[:,0,:,m])
                    sd_x = np.sqrt(np.var(temp_data[:,0,:,m]))
                    mean_y = np.mean(temp_data[:,1,:,m])
                    sd_y = np.sqrt(np.var(temp_data[:,1,:,m]))
                    if sd_x!=0:
                        temp_data[:,0,:,m] = (temp_data[:,0,:,m]-mean_x)/sd_x
                    if sd_y!=0:
                        temp_data[:,1,:,m] = (temp_data[:,1,:,m]-mean_y)/sd_y

            x_numpy.append(temp_data)

        x_numpy = np.array([item for sublist in x_numpy for item in sublist])
                
        ### select every nth element 
        if (self.nth_element > 1):
            x_numpy = x_numpy[0::self.nth_element, :, :, :]

        ### signer is left or right handed?
        if self.random_flip == True:
            if np.random.choice([0, 1]) == 1:
                if self.normalise==True: 
                    x_numpy[:, 0, :, :] = -1 * x_numpy[:, 0 , :, :]
                else:
                    x_numpy[:, 0, :, :] = -1 * x_numpy[:, 0 , :, :]+1

        if self.normalise==True: 
            x_numpy[:,1,:,:] = -x_numpy[:,1,:,:]
        else:
            x_numpy[:,1,:,:] = -x_numpy[:,1,:,:]+1

        ### display data 
        if self.visualise == True: 
            for i in range(len(x_numpy)):
                plt.ion()
                plt.scatter(x_numpy[i,0,:,0],x_numpy[i,1,:,0])
                if self.normalise==True: 
                    plt.xlim((-3, 3))
                    plt.ylim((-3, 3))
                else: 
                    plt.xlim((0, 1))
                    plt.ylim((0, 1))
                plt.show()
                plt.pause(0.1) # pause time
                plt.cla()
            # plt.close()

        x_numpy = np.moveaxis(x_numpy, [0, 1, 2, 3], [1, 0, 2, 3])
        data_numpy = torch.FloatTensor(x_numpy) 

        return data_numpy


def compute_size_movement(comb_data_numpy):
    ### size Time * 3 * 127 * 1
    comb_data_numpy = np.moveaxis(comb_data_numpy, [0, 1, 2, 3], [1, 0, 2, 3])
    ### size 3 * T * 127 * 1
    #### measure size and variation in hand movement
    ### make dataframe of hands
    lh_y = comb_data_numpy[1, :, 85:106, 0]
    lh_score = comb_data_numpy[2, :, 85:106, 0]
    rh_y = comb_data_numpy[1, :, 106:127, 0]
    rh_score = comb_data_numpy[2, :, 106:127, 0]

    max_lh_y = []
    min_lh_y = []
    max_rh_y = []
    min_rh_y = []
    wrist_lh_y = []
    wrist_rh_y = []
    for t in range(lh_y.shape[0]):
        if len(lh_y[t][lh_score[t] > 0.1]) > 0:
            max_lh_y.append(np.max(lh_y[t][lh_score[t] > 0.1]))
            min_lh_y.append(np.min(lh_y[t][lh_score[t] > 0.1]))
        if len(rh_y[t][rh_score[t] > 0.1]) > 0:
            max_rh_y.append(np.max(rh_y[t][rh_score[t] > 0.1]))
            min_rh_y.append(np.min(rh_y[t][rh_score[t] > 0.1]))
        if lh_score[t][0] > 0.05:
            wrist_lh_y.append(lh_y[t][0])
        if rh_score[t][0] > 0.05:
            wrist_rh_y.append(rh_y[t][0])

    #### height of hands
    lh_height = 0
    rh_height = 0
    lh_movement = 0
    rh_movement = 0
    if len(max_lh_y) > 0:
        lh_height = np.median(np.array(max_lh_y) - np.array(min_lh_y))
    if len(max_rh_y) > 0:
        rh_height = np.median(np.array(max_rh_y) - np.array(min_rh_y))
    if len(wrist_lh_y) > 0:
        lh_movement = np.var(np.array(wrist_lh_y))
    if len(wrist_rh_y) > 0:
        rh_movement = np.var(np.array(wrist_rh_y))

    height = max(lh_height, rh_height)
    movement = max(lh_movement, rh_movement)
    return height, movement


def get_main_signer(keypoints, interval_length = 25):
    measures = []
    for i in range(max(1,keypoints.shape[0]-interval_length)):
        temp_measures = []
        for j in range(keypoints.shape[3]):
            height, movement = compute_size_movement(keypoints[i:min(i+interval_length,keypoints.shape[0]),:,:,j, np.newaxis])
            temp_measures.append([height, movement, height*movement])
        measures.append(temp_measures)
    
    winner = []
    for i in range(len(measures)):
        meas_np = np.array(measures[i])
        winner.append(np.bincount(np.argmax(meas_np, axis=0)).argmax())
    
    for i in range(interval_length):
        winner.append(winner[-1])
    
    final_winners = []
    for i in range(keypoints.shape[0]):
        final_winners.append(np.bincount(winner[i:min(len(winner),interval_length+i)]).argmax())
    
    #import matplotlib.pyplot as plt
    
    #plt.plot(np.arange(len(final_winners)), final_winners)
    #plt.plot(np.arange(len(winner)), winner)
    
    new_keypoints = np.zeros(keypoints.shape)[:,:,:,0]
    for i in range(len(keypoints)):
        new_keypoints[i] = keypoints[i,:,:,final_winners[i]]
    
    return new_keypoints[:,:,:,np.newaxis]

    