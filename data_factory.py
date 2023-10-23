import numpy 
import pickle
import pandas as pd
import os
import time
import math

import sqlalchemy
from sqlalchemy import create_engine,text
import numpy as np
from torch.utils.data import Dataset,DataLoader,Subset
engine = create_engine("sqlite+pysqlite:///deephoopers-mod.db",echo=True,future=True)
def query_data():
    with engine.connect() as conn:
        data = conn.execute(text('SELECT Event_ID,Moment_Time,Ball_X,Ball_Y,Ball_DXDT,Ball_DYDT,Player_H_0_X,Player_H_1_X,Player_H_2_X,Player_H_3_X,Player_H_4_X,Player_V_0_X,Player_V_1_X,Player_V_2_X,Player_V_3_X,Player_V_4_X, Player_H_0_Y,Player_H_1_Y,Player_H_2_Y,Player_H_3_Y,Player_H_4_Y,Player_V_0_Y,Player_V_1_Y,Player_V_2_Y,Player_V_3_Y,Player_V_4_Y ,Player_H_0_DXDT,Player_H_1_DXDT,Player_H_2_DXDT,Player_H_3_DXDT,Player_H_4_DXDT,Player_V_0_DXDT,Player_V_1_DXDT,Player_V_2_DXDT,Player_V_3_DXDT,Player_V_4_DXDT, Player_H_0_DYDT,Player_H_1_DYDT,Player_H_2_DYDT,Player_H_3_DYDT,Player_H_4_DYDT,Player_V_0_DYDT,Player_V_1_DYDT,Player_V_2_DYDT,Player_V_3_DYDT,Player_V_4_DYDT FROM TrackingDataTable'))
        start = time.time()
        data = [list(t) for t in data]
        t_data = np.array(data,dtype=float)
        engine.dispose()
        return t_data

class TrackingDataDataset(Dataset):
    def __init__(self,flag='train',size=None,scale=True,freq = 25,velocity=True,data =None):
        self.dataset_len = len(data)
        self.data = data
        self.flag=flag
        self.freq = freq
        self.scale = scale
        self.velocity = velocity
        self.windowed = None
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
        '''
        data split need to be split by games or by events, rn 
        '''
        if self.flag == 'train':
            index = list(range(0,int(self.dataset_len*0.6),int(25/self.freq)))
        elif self.flag ==  'val':
            index = list(range(int(self.dataset_len*0.6),int(self.dataset_len*0.8),int(25/self.freq)))
        elif self.flag ==  'test':
            index = list(range(int(self.dataset_len*0.8),self.dataset_len,int(25/self.freq)))
        #self.data = self.data[index]
        if self.scale:
            mask = np.isnan(self.data ) | (self.data  == None)
            self.data = self.data[~mask.any(axis=1)]
            self.data[:, [2] + [4] +list(range(6, 16)) + list(range(26, 36))] = self.data[:, [2] + [4] +list(range(6, 16)) + list(range(26, 36))] / 100.
            self.data[:, [3] + [5] + list(range(16,26)) + list(range(36, 46))] = self.data[:,[3] + [5] + list(range(16,26)) + list(range(36, 46))] / 50.

        if self.velocity:
            self.data = self.data[:,[0]+[4]+[5]+list(range(26,46))]
        print(self.data.shape)
        #===================TAKING TOO LONG
        # for i in range(len(self.data)):
        #     if i%10000 ==0:
        #         print("Processed time step ",i)
        #     curr_event_id = self.data[i][0] #get the current moment id
        #     length = len(np.where(self.data[:,0]<=curr_event_id)) #how many snapshots have moment id smaller or equal to current moment id
        #     cutoff_index = length - self.pred_len - self.seq_len
        #     if i <= cutoff_index:
        #         seq_start = i
        #         seq_end = index + self.seq_len
        #         target_start = seq_end - self.target_len
        #         target_end = seq_end + self.pred_len
        #         if i==0:
        #             self.seq_x = self.data[seq_start:seq_end]
        #             self.seq_y = self.data[target_start:target_end]
        #         else:
        #             self.seq_x = np.vstack(self.seq_x ,self.data[seq_start:seq_end])
        #             self.seq_y = np.vstack(self.seq_y,self.data[target_start:target_end])
        #     else:
        #         continue
        #===================TAKING TOO LONG
        self.data_x = self.data
        self.data_y = self.data

    #         [              seq_len           ]     
    #                          [  target_len   ][   pred_len   ]
    #------------------------------------------------------------------
    #         |                |                |               |
    #        seq_start     target_start      seq_end        target_end
     
    def __getitem__(self,index):
        seq_begin = index
        seq_end = index + self.seq_len
        target_start = seq_end - self.target_len
        target_end = seq_end + self.pred_len
        seq_x = self.data[seq_begin:seq_end]
        seq_y = self.data[target_start:target_end]
        return seq_x,seq_y

    def __len__(self):
        return len(self.data)

    def get_window_indices(self):
        indices = []
        index = 0
        train_index = 0
        train_index_found = False
        val_test_index = 0
        val_test_index_found = False
        for i in range(len(self.data)-self.pred_len-self.seq_len):
            if self.data[i,0] != self.data[i+self.pred_len+self.seq_len-1,0]: #if you are transitioning from one event to another
                if not train_index_found and i >= int((len(self.data)-self.pred_len-self.seq_len)*0.6):
                    train_index = i
                    train_index_found = True
                elif not val_test_index_found and i >= int((len(self.data)-self.pred_len-self.seq_len)*0.8):
                    val_test_index_found = True
                    val_test_index = i
                continue
            else:
                indices.append(i)
        return train_index,val_test_index,indices


# def data_provider(args,data):
#     dataset = TrackingDataDataset(args.flag,args.size,args.scale,args.freq,args.velocity,data)
    # indices = dataset.get_window_indices()
    #train_index,val_test_index,train_indices = indices[0:int(len(indices)*0.6)]

    # train_indices = indices[0:indices.index(train_index)]
    # val_indices = indices[indices.index(train_index):indices.index(val_test_index)]
    # test_indices = indices[indices.index(val_test_index):len(indices)]
    # if args.flag == 'train':
    #     dataset = Subset(dataset,train_indices)
    # elif args.flag == 'val':
    #     dataset = Subset(dataset,val_indices)
    # else:
    #     dataset = Subset(dataset,test_indices)
#     dataloader = DataLoader(dataset,batch_size = args.batch_size,shuffle=None,num_workers=args.num_workers,drop_last = False)
#     return dataset,dataloader


#THIS CODE IS JUST FOR TESTING OUT THE DATASET AND DATALOADER, SO FAR IT WORKS:)
def data_provider(args,flag,data):
    dataset = TrackingDataDataset(flag,None,True,25,True,data)
    indices = dataset.get_window_indices()
    train_index,val_test_index,train_indices = indices[0:int(len(indices)*0.6)]
    
    train_indices = indices[0:indices.index(train_index)]
    val_indices = indices[indices.index(train_index):indices.index(val_test_index)]
    test_indices = indices[indices.index(val_test_index):len(indices)]
    if args.flag == 'train':
        dataset = Subset(dataset,train_indices)
    elif args.flag == 'val':
        dataset = Subset(dataset,val_indices)
    else:
        dataset = Subset(dataset,test_indices)
    dataloader = DataLoader(dataset,batch_size = 32,shuffle=None,num_workers=0,drop_last = False)
    return dataset,dataloader

data = query_data()
dataset,dataloader = data_provider(None, 'train', data)
print(dataset)
for batch_x,batch_y in dataloader:
    print(batch_x.shape,batch_y.shape)
    break
