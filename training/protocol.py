import torch
import numpy as np 
import torch.nn as nn
import math
from preprocessing.data_factory import data_provider
from models.discrete_seq_models import GRUEncoderDecoder, LSTM, BiLSTM
from models.baseline import MLP
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class Protocol():
    def __init__(self,model,epoch,learning_rate,batch_size,num_features,seq_len,pred_len,target_len,scale,velocity,freq):
        super(Protocol,self).__init__()
        #model
        self.model = model
        self.model_name = ""
        if isinstance(model, GRUEncoderDecoder):
            self.model_name = 'GRUED'
        elif isinstance(model, LSTM):
            self.model_name = 'LSTM'
        elif isinstance(model, BiLSTM):
            self.model_name = 'BiLSTM'
        elif isinstance(model,MLP):
            self.model_name = 'MLP'

        self.model_size = 0
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        #hyper param
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        #data parameters
        self.num_features = num_features
        self.seq_len = seq_len
        self.target_len = target_len
        self.pred_len = pred_len
        self.scale = scale
        self.velocity = velocity
        self.freq = freq
        #loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        #training metrics
        self.training_loss = []
        self.val_loss = []
        self.avg_train_loss = 0.0
        self.avg_val_loss = 0.0
        self.avg_test_loss = 0.0

        self.best_val_loss = 0.0
        self.best_epoch = 0
        self.best_path = ""

        #other metrics 
        self.training_time = 0.0
        self.avg_inference_time = 0.0
        self.avg_time_per_epoch = 0.0

        #loss fn & optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

    def load_data(self):
        self.train_loader,self.val_loader,self.test_loader = data_provider(size = (self.seq_len,self.target_len,self.pred_len),scale=self.scale,velocity=self.velocity,freq = self.freq)
        print("\n")
        print("###############DATA LOADING SUCCESS###############")
        print("Data paremters: ")
        print("Seq len: ",self.seq_len)
        print("Target len: ",self.target_len)
        print("Pred len: ",self.pred_len)
        print("Normalization: ",self.scale)
        print("Using velocity as features: ",self.velocity)
        print("Sampling Frequency: ",self.freq)
        print("\n")

    def train(self):
        print("###############STARTING TRAINING###############")
        self.model.train()
        self.model = self.model.to(self.device)
        train_start = time.time()
        
        for epoch in range(self.epoch):
            epoch_start = time.time()
            self.avg_train_loss = 0.0
            self.avg_val_loss = 0.0
            for batch_x,batch_y in tqdm(self.train_loader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                inf_start = time.time()
                output = self.model(batch_x)
                inf_end = time.time()
                self.avg_inference_time += inf_end-inf_start
                loss = self.loss_fn(output,batch_y)
                self.avg_train_loss +=loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.avg_train_loss /= len(self.train_loader)
            self.training_loss.append(self.avg_train_loss )
            self.avg_inference_time /= len(self.train_loader)
            with torch.no_grad():
                for batch_x,batch_y in tqdm(self.val_loader):
                    batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                    output = self.model(batch_x)
                    loss = self.loss_fn(output,batch_y)
                    self.avg_val_loss += loss.item()
                self.avg_val_loss /= len(self.val_loader)
                self.val_loss.append(self.avg_val_loss) 
                if self.avg_val_loss < self.best_val_loss:
                    self.best_val_loss = self.avg_val_loss
                    self.best_epoch = epoch
                    self.best_path = f'../checkpoints/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_win{self.seq_len}-{self.target_len}-{self.pred_len}_v{self.velocity}_sc{self.scale}_ep{epoch}.pth'
            epoch_end = time.time()
            epoch_time = epoch_end-epoch_start
            self.save(epoch,self.model,self.avg_val_loss)
            self.avg_time_per_epoch += epoch_time/self.epoch
            print(f'===Epoch: {epoch} | Training Loss: {self.avg_train_loss} | Validation Loss: {self.avg_val_loss} | Avg inference time: {self.avg_inference_time} | Time per epoch: {epoch_time}')
            
        train_end = time.time()
        self.training_time = train_end-train_start
        print("###############TRAINING COMPLETE###############")
        print("Training metrics: ")
        print("Final training loss: ",self.avg_train_loss)
        print("Final validation loss: ",self.avg_val_loss)
        print("Best validation loss: ",self.best_val_loss)
        print("Avg training inference time: ",self.avg_inference_time)
        print("Total training time: ",self.training_time)
        print(f"Model size: {self.find_model_size(self.model)} MB")
        self.plot_training_curve()
        return

    def find_model_size(self,model):
        model = self.model
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def eval(self):
        with torch.no_grad():
            for batch_x,batch_y in self.test_loader:
                output = self.model(batch_x.float().to(self.device))
                loss = self.loss_fn(output,batch_y.float().to(self.device))
                self.avg_test_loss+=loss.item()
            self.avg_test_loss /= len(self.test_loader)
        print("Test Loss: ",self.avg_test_loss)
        return 

    def save(self,epoch,model,loss):
        torch.save({
            'epoch':epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, f'../checkpoints/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_win{self.seq_len}-{self.target_len}-{self.pred_len}_v{self.velocity}_sc{self.scale}_ep{epoch}.pth')
        return
    def plot_training_curve(self):
        n = len(self.training_loss) # number of epochs
        fig = plt.figure()
        plt.title("Train vs Validation Loss")
        plt.plot(range(1,n+1), self.training_loss, label="Train")
        plt.plot(range(1,n+1), self.val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.savefig(f"../training_curves/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_win{self.seq_len}-{self.target_len}-{self.pred_len}_v{self.velocity}_sc{self.scale}.png")