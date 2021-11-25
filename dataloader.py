
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, RandomSampler


class get_loader(nn.Module):
    
    def __init__(self, root_dir, channel_list = None, n_sess = 2, mode = 1):
        '''
        root_dir: the main folder with subject-wise subfolders
        n_sess:   number of sessions to consider. 1/2/3
        mode:     0-pronounced 
                  1-inner 
                  2-visualized
                  
        '''
        super(get_loader, self).__init__()
        self.root_dir = root_dir
        self.n_sess = n_sess
        self.mode = mode
        self.channel_list = channel_list
        
    def load_single_subject(self, sub_idx):
        '''
        sub_idx: 1, 2, ..... integers
        '''
        data = dict()
        y = dict()
        N_B_arr = np.arange(1, self.n_sess+1, 1)
        N_S = sub_idx
        
        for N_B in N_B_arr:
            # name correction if N_Subj is less than 10
            if N_S<10:
                Num_s='sub-0'+str(N_S)
            else:
                Num_s='sub-'+str(N_S)
            
            file_name = self.root_dir  + Num_s + '/ses-0'+ str(N_B) + '/' +Num_s+'_ses-0'+str(N_B)+'_events.dat'
            y[N_B] = np.load(file_name,allow_pickle=True)
        
        
            #  load data and events
            file_name = self.root_dir  + Num_s + '/ses-0'+ str(N_B) + '/' +Num_s+'_ses-0'+str(N_B)+'_eeg-epo.fif'
            X= mne.read_epochs(file_name,verbose='WARNING')
            if self.channel_list:
                X = X.pick_channels(ch_names = self.channel_list)
            data[N_B]= X._data
            
        #stack the sessions
        X = data.get(1)
        Y = y.get(1)
        for i in range(2, self.n_sess+1, 1):
            X = np.vstack((X, data.get(i)))
            Y = np.vstack((Y, y.get(i)))
        
        #select the recordings from required mode only
        X_mode = X[Y[:,2] == self.mode]
        Y_mode = Y[Y[:, 2] == self.mode]
        
        return X_mode, Y_mode
    
    def load_multiple_subjects(self, subjects):
        '''
        Load all subjects required and stack them into single array (n_rec*n_sub, 128, 1153)
        '''
        X_t, Y_t = [], []
        for idx in subjects:
            Xi, Yi = self.load_single_subject(idx)
            X_t.append(Xi)
            Y_t.append(Yi)
    
        return np.vstack(X_t), np.vstack(Y_t)
    
    def forward(self, subjects, batch_size = 1):
        '''
        subjects: list of subject indices to load data from
        '''
        X, Y = self.load_multiple_subjects(subjects)
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        
        train_data = TensorDataset(X, Y)
        train_sampler = RandomSampler(X)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
                  
        # can add test/validation loader too 
        
        return train_dataloader
