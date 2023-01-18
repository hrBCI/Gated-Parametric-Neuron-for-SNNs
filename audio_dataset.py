import numpy as np
import torch.utils.data
import torch
import tables


"https://compneuro.net/"


def AUDIO_binary_image_readout(times,units,T):
    img = []
    dt=1/T
    for i in range(T):
        
        idxs = np.argwhere(times<=i*dt).flatten()
    
        vals = units[idxs]

        vals = vals[vals > 0]
        #vector = np.zeros(700)
        #vector[700-vals] = 1
        vector = np.bincount(700-vals)
        vector = np.pad(vector,(0,700-vector.shape[0]))
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
    return np.array(img)

def AUDIO_generate_dataset(file_name,T):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    print("Number of samples: ",len(times))
    print('data processing...')
    X = []
    y = []
    for i in range(len(times)):
        
        tmp = AUDIO_binary_image_readout(times[i], units[i],T=T)
        X.append(tmp)
        y.append(labels[i])

        
    return np.array(X),np.array(y)

def AUDIO_datasets(file_name,T,transform=False):

    x,y=AUDIO_generate_dataset(file_name,T)

    if transform:
        x = transform(x)

    x=torch.from_numpy(x/1.0).type(torch.FloatTensor)
    y=torch.from_numpy(y/1.0).type(torch.LongTensor)
  
    

    AUDIO_datasets = torch.utils.data.TensorDataset(x, y)

    return AUDIO_datasets

