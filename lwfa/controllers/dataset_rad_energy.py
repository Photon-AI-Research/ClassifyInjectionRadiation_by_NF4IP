from torch.utils.data import Dataset
import torch
import numpy as np
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    def __init__(self, angle=4, norm=['none'], data_sets=[1,3], balance=0, evol=True):
        
        if norm == 'none':
            print('No normalization will be applied')
            
        data = {}
        rad_val = {}
        if evol==True:
            ev_labels = {}
        else:
            labels = {}
        theta = {}

        for i in range(1,4):
            data[i] = np.load("./data/input/LWFA_NN_dataExtraction_0{}_evolution.npy".format(i), allow_pickle=True).item()
            theta[i] = np.round(data[i]["theta_rad"]).T[0]
        self.theta = theta
        # Get matching angles
        from functools import reduce
        match = reduce(np.intersect1d, ([theta[i] for i in theta.keys()]))
        
        match_positions = []
        for i in [theta[i] for i in theta.keys()]:
            positions = []
            for j in match:
                positions.append(np.where(i==j)[0][0])
            match_positions.append(positions)
        self.match_positions = match_positions
        def find_nearest(array,value):
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
                return array[idx-1]
            else:
                return array[idx]
        user_angle = angle
        angle = int(find_nearest(match, angle))
        if user_angle != angle:
            print('The angle {} was not available, so angle {} was chosen instead'.format(user_angle, angle)) 
        angle = np.where(match==angle)[0][0]
        
        def get_inter_e_label(index):
        
            # transform bool to 0/1
            label_e_num = np.zeros(data[index]["label_e"].shape)
            for i in range(0, data[index]["label_e"].size):
                if data[index]["label_e"][i] == False:
                    label_e_num[i] = 0
                else:
                    label_e_num[i] = 1

            # interpolate over label_e to get values for rad timesteps
            inter_e_label = np.zeros(data[index]["index_rad"].shape)
            for i in range(0, data[index]["index_rad"].size):
                inter_e_label[i] = np.interp(data[index]["index_rad"][i], data[index]["index_e"], label_e_num)
            inter_e_label = np.round(inter_e_label)
            
            return inter_e_label
        def get_evol_label(index):
            # interpolate over label_e to get values for rad timesteps
            inter_e_label = np.zeros(data[index]["index_rad"].shape)
            for i in range(0, data[index]["index_rad"].size):
                inter_e_label[i] = np.interp(data[index]["index_rad"][i], data[index]["index_e"], data[index]["injection_evolution_label"])
            return inter_e_label
        
        
        # get radiation values and labels
        for i in data.keys():
            if evol==True:
                ev_labels[i] = get_evol_label(i)
            else:
                labels[i] = get_inter_e_label(i)
            if i == 2:
                d2_rad_val = data[i]["spectrum_rad"][:,match_positions[i-1][angle],:]
                interp_rad_val = np.zeros((600,512))
                for j in range(0,d2_rad_val.shape[0]): 
                    for k in range(0, interp_rad_val[j].size):
                        interp_rad_val[j][k] = np.interp(data[1]["omega_rad"][k], data[2]["omega_rad"], d2_rad_val[j])
                rad_val[i] = interp_rad_val
            else:
                rad_val[i] = data[i]["spectrum_rad"][:,match_positions[i-1][angle],:]
            
        
        def norm_data(rad_val, label):
            if 'scaler' in norm:
                from sklearn.preprocessing import StandardScaler
                rad_val = StandardScaler().fit_transform(rad_val)
            # cut the peak created by the laser
            if 'cut' in norm:
                rad_val = np.delete(rad_val, range(120,221), 1)
            # min/max naromalization
            if 'max' in norm:
                rad_val = rad_val / np.max(rad_val)
            # log10 normalization
            if 'log' in norm:
                
                for i in range(0,rad_val.shape[0]):
                    
                    #rad_val[i] = [0 if x < 1e-60 else ((np.log10(x)/60) +1) for x in rad_val[i]]
                   
                    rad_val[i] = [0 if x < 1e-10 else x for x in rad_val[i]]
                rad_val = np.ma.log10(rad_val).filled(0)
                for i in range(0,rad_val.shape[0]):
                    rad_val[i] = [((x/10)+1) if x < 0 else x for x in rad_val[i]]
            # use only every 'balance' sample with label zero
            if balance != 0:
                one_label = np.where(label==1)[0]
                zero_label = np.where(label==0)[0]
                rad_val_zero = []
                for i in range(0,len(zero_label)):
                    if i%balance == 0:
                        rad_val_zero.append(zero_label[i])
                
                rad_val = np.vstack((rad_val[one_label], rad_val[rad_val_zero]))
                label = np.concatenate((label[one_label], label[rad_val_zero]))   
            if 'drop 0' in norm:
                rad_val_no_zeros = np.zeros(rad_val.shape[1])
                label_no_zeros = []
                for i in range(0, rad_val.shape[0]):
                    if np.max(rad_val[i]) != 0:
                        rad_val_no_zeros = np.vstack((rad_val_no_zeros, rad_val[i]))
                        label_no_zeros.append(label[i])
                rad_val = rad_val_no_zeros[1:]
                label = np.array(label_no_zeros)
            return rad_val, label
        
        if evol == True:
            norm = [norm_data(rad_val[i], ev_labels[i]) for i in data_sets]
        else:
            norm = [norm_data(rad_val[i], labels[i]) for i in data_sets]
        self.rad_val = np.vstack([norm[i][0] for i in range(0,len(norm))])
        self.label = np.concatenate([norm[i][1] for i in range(0,len(norm))])
        #rad_val =np.vstack(([rad_val[i] for i in data_sets]))
        #label = np.concatenate(([labels[i] for i in data_sets]))
        #self.rad_val, self.label = norm_data(rad_val, label)
        #self.rad_val = torch.tensor(self.rad_val).to(device)
        #self.label = torch.tensor(self.label).to(device).float()
        self.rad_val = torch.tensor(self.rad_val)
        self.label = torch.tensor(self.label).float()

        return

    def __len__(self):
        return self.rad_val.shape[0]

    def __getitem__(self, index):
        rad = self.rad_val[index]
        label = self.label[index, None]
        return rad, label
