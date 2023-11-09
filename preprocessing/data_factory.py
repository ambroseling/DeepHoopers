import numpy 
import pickle
import pandas as pd
import os
import time
import math
import itertools
import sqlalchemy
from sqlalchemy import create_engine,text
import numpy as np
from torch.utils.data import Dataset,DataLoader,Subset

#need to cd inot preprocessing to run main.py
def query_data():
    engine = create_engine("sqlite+pysqlite:///deephoopers-mod.db",echo=True,future=True)
    with engine.connect() as conn:
        data = conn.execute(text('SELECT Event_ID,Moment_Time,Ball_X,Ball_Y,Ball_DXDT,Ball_DYDT,Player_H_0_X,Player_H_1_X,Player_H_2_X,Player_H_3_X,Player_H_4_X,Player_V_0_X,Player_V_1_X,Player_V_2_X,Player_V_3_X,Player_V_4_X, Player_H_0_Y,Player_H_1_Y,Player_H_2_Y,Player_H_3_Y,Player_H_4_Y,Player_V_0_Y,Player_V_1_Y,Player_V_2_Y,Player_V_3_Y,Player_V_4_Y ,Player_H_0_DXDT,Player_H_1_DXDT,Player_H_2_DXDT,Player_H_3_DXDT,Player_H_4_DXDT,Player_V_0_DXDT,Player_V_1_DXDT,Player_V_2_DXDT,Player_V_3_DXDT,Player_V_4_DXDT, Player_H_0_DYDT,Player_H_1_DYDT,Player_H_2_DYDT,Player_H_3_DYDT,Player_H_4_DYDT,Player_V_0_DYDT,Player_V_1_DYDT,Player_V_2_DYDT,Player_V_3_DYDT,Player_V_4_DYDT FROM TrackingDataTable'))
        start = time.time()
        data = [list(t) for t in data]
        t_data = np.array(data,dtype=float)
        engine.dispose()
        return t_data

class TrackingDataDataset(Dataset):
    def __init__(self,size=None,scale=True,freq = 25,velocity=True,data =None):
        self.dataset_len = len(data)
        self.data = data
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
        #self.data = self.data[index]
        mask = np.isnan(self.data ) | (self.data  == None)
        self.data = self.data[~mask.any(axis=1)]
        self.data = self.data[::int((25/self.freq))]
        if self.scale:

            self.data[:, [2] + [4] +list(range(6, 16)) + list(range(26, 36))] = self.data[:, [2] + [4] +list(range(6, 16)) + list(range(26, 36))] / 100.
            self.data[:, [3] + [5] + list(range(16,26)) + list(range(36, 46))] = self.data[:, [3] + [5] + list(range(16,26)) + list(range(36, 46))] / 50.

        if self.velocity:
            self.data = self.data[:,[0]+[4]+[5]+list(range(26,46))]*1000 #convert back to 
        else:
            self.data = self.data[:,[0]+[2]+[3]+list(range(6, 16))+ list(range(16,26)) ]
        self.data_x = np.delete(self.data,0,axis=1)
        self.data_y = np.delete(self.data,0,axis=1)


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
        seq_x = self.data_x[seq_begin:seq_end]
        seq_y = self.data_y[target_start:target_end]
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
                    indices.append(i)
                    train_index_found = True
                elif not val_test_index_found and i >= int((len(self.data)-self.pred_len-self.seq_len)*0.8):
                    val_test_index_found = True
                    indices.append(i)
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
def data_provider(size = (10,7,3),scale=True,velocity=True,freq = 25 ):
    data = query_data()
    dataset = TrackingDataDataset(size,scale,freq,velocity,data)
    train_index,val_test_index,indices = dataset.get_window_indices()
    #indices = indices[0:int(len(indices)*0.6)]
    
    train_indices = indices[0:indices.index(train_index)]
    val_indices = indices[indices.index(train_index):indices.index(val_test_index)]
    test_indices = indices[indices.index(val_test_index):len(indices)]
    train_dataset = Subset(dataset,train_indices)
    val_dataset = Subset(dataset,val_indices)
    test_dataset = Subset(dataset,test_indices)

    train_dataloader = DataLoader(train_dataset,batch_size = 32,shuffle=None,num_workers=0,drop_last = True)
    #train_dataloader = iter(itertools.islice(train_dataloader, 0, len(train_dataloader) - 3))
    # print(next(iter(train_dataloader))[0].shape)
    val_dataloader = DataLoader(val_dataset,batch_size = 32,shuffle=None,num_workers=0,drop_last = True)
    test_dataloader = DataLoader(test_dataset,batch_size = 32,shuffle=None,num_workers=0,drop_last = True)

    return train_dataloader,val_dataloader,test_dataloader

# data = query_data()
# dataset,dataloader = data_provider(None, 'train', data)
# print(dataset)
# for batch_x,batch_y in dataloader:
#     print(batch_x.shape,batch_y.shape)
#     break
# if __name__ == '__main__':
#     data_provider()