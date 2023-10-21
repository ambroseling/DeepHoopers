import numpy 
import pickle
import pandas as pd
import os
import time
import math

import sqlalchemy
from sqlalchemy import create_engine,text
import numpy as np
from torch.utils.data import Dataset,DataLoader
engine = create_engine("sqlite+pysqlite:///deephoopers.db",echo=True,future=True)
def query_data():
    with engine.connect() as conn:
        data = conn.execute(text('SELECT Ball_X,Ball_Y,Ball_Z,Player_H_0_X,Player_H_1_X,Player_H_2_X,Player_H_3_X,Player_H_4_X,Player_V_0_X,Player_V_1_X,Player_V_2_X,Player_V_3_X,Player_V_4_X, Player_H_0_Y,Player_H_1_Y,Player_H_2_Y,Player_H_3_Y,Player_H_4_Y,Player_V_0_Y,Player_V_1_Y,Player_V_2_Y,Player_V_3_Y,Player_V_4_Y FROM TrackingDataTable'))
        start = time.time()
        data = [list(t) for t in data]
        t_data = np.array(data,dtype=float)
        engine.dispose()
        return t_data

class TrackingDataDataset(Dataset):
    def __init__(self,flag='train',size=None,scale=True,freq = 25,data =None):
        self.dataset_len = len(data)
        self.data = data
        self.flag=flag
        self.freq = freq
        self.scale = scale
        if size == None:
            self.seq_len = 10
            self.target_len = 10
            self.pred_len = 5
        else:
            self.seq_len = size[0]
            self.target_len = size[1]
            self.pred_len = size[2]
        self.__read_data__()
    def __read_data__(self):
        if self.flag == 'train':
            index = list(range(0,int(self.dataset_len*0.6),int(25/self.freq)))
        elif self.flag ==  'val':
            index = list(range(int(self.dataset_len*0.6),int(self.dataset_len*0.8),int(25/self.freq)))
        elif self.flag ==  'test':
            index = list(range(int(self.dataset_len*0.8),self.dataset_len,int(25/self.freq)))
        if self.scale:
            mask = np.isnan(data) | (data == None)
            self.data = self.data[~mask.any(axis=1)]
            self.data[:, [0] + list(range(3, 14))] = self.data[:, [0] + list(range(3, 14))] / 100.
            self.data[:,[1] + list(range(14,23))] = self.data[:,[1] + list(range(14,23))] / 50.
            print(self.data)
        self.data_x = self.data
        self.data_y = self.data

    #         [              seq_len           ]     
    #                          [  target_len   ][   pred_len   ]
    #------------------------------------------------------------------
    #         |                |                |               |
    #        seq_start     target_start      seq_end        target_end
     
    def __getitem__(self,index):
        seq_start = index
        seq_end = index + self.seq_len
        target_start = seq_end - self.target_len
        target_end = seq_end + self.pred_len
        seq_x = self.data_x[seq_start:seq_end]
        seq_y = self.data_y[target_start:target_end]
        return seq_x,seq_y
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

def data_provider(args,flag,data):
    dataset = TrackingDataDataset(args.flag,args.size,args.scale,args.freq,data)
    dataloader = DataLoader(dataset,batch_size = args.batch_size,shuffle=None,num_workers=args.num_workers,drop_last = None)
    return dataset,dataloader


#THIS CODE IS JUST FOR TESTING OUT THE DATASET AND DATALOADER, SO FAR IT WORKS:)
# def data_provider(args,flag,data):
#     dataset = TrackingDataDataset(flag,None,True,25,data)
#     dataloader = DataLoader(dataset,batch_size = 32,shuffle=None,num_workers=0,drop_last = False)
#     return dataset,dataloader

# data = query_data()
# dataset,dataloader = data_provider(None, 'train', data)
# print(dataset)
# for batch_x,batch_y in dataloader:
#     print(batch_x.shape,batch_y.shape)
#     break
