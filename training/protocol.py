import torch
import numpy as np 
import torch.nn as nn
import math
from preprocessing.data_factory import data_provider
from models.discrete_seq_models import GRUEncoderDecoder, LSTM, BiLSTM
from models.spatial_models import GNNLSTM, GNNBiLSTM,GNNGRUED
from models.baseline import MLP
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Protocol():
    def __init__(self,model,epoch,learning_rate,batch_size,num_features,seq_len,pred_len,target_len,scale,velocity,freq,graph,device):
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
        elif isinstance(model, GNNLSTM):
            self.model_name = "GNNLSTM"
        elif isinstance(model, GNNBiLSTM):
            self.model_name = "GNNBiLSTM"
        elif isinstance(model, GNNGRUED):
            self.model_name = "GNNGRUED"

        self.model_size = 0
        self.device = device
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
        self.graph = graph
        #loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.test_data = None

        #training metrics
        self.training_loss = []
        self.val_loss = []
        self.avg_train_loss = 0.0
        self.avg_val_loss = 0.0
        self.avg_test_loss = 0.0

        self.best_val_loss = 100
        self.best_epoch = 0
        self.best_path = ""

        #other metrics 
        self.training_time = 0.0
        self.avg_inference_time = 0.0
        self.avg_time_per_epoch = 0.0

        #loss fn & optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

        self.court_path = "/Users/ambroseling/Desktop/DeepHoopers/DeepHoopers/assets/nba_court_T.png"

    def load_data(self):
        self.train_loader,self.val_loader,self.test_loader,self.test_data = data_provider(size = (self.seq_len,self.target_len,self.pred_len),scale=self.scale,velocity=self.velocity,graph=self.graph,freq = self.freq)
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
        if not self.graph:
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
        else:
            for epoch in range(self.epoch):
                epoch_start = time.time()
                self.avg_train_loss = 0.0
                self.avg_val_loss = 0.0
                for batch in tqdm(self.train_loader):
                    batch_y = torch.reshape(batch.y.float(),(self.batch_size,self.seq_len,self.num_features))
                    inf_start = time.time()
                    output = self.model(batch)
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
                    for batch in tqdm(self.val_loader):
                        batch_y = torch.reshape(batch.y.float(),(self.batch_size,self.seq_len,self.num_features))
                        output = self.model(batch)
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
        rpint("Best epoch: ",self.best_epoch)
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
        if not self.graph:
            with torch.no_grad():
                for batch_x,batch_y in self.test_loader:
                    output = self.model(batch_x.float().to(self.device))
                    loss = self.loss_fn(output,batch_y.float().to(self.device))
                    self.avg_test_loss+=loss.item()
                self.avg_test_loss /= len(self.test_loader)
            print("Test Loss: ",self.avg_test_loss)
        else:
            with torch.no_grad():
                for batch in self.test_loader:
                    batch_y = torch.reshape(batch.y.float(),(self.batch_size,self.seq_len,self.num_features))
                    output = self.model(batch)
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

    def generate_predictions(self,model):
        pred_data = self.test_data[0:self.seq_len]
        pred_data = torch.unsqueeze(torch.tensor(pred_data),dim=0)
        predictions = []
        for _ in range(len(self.test_data) - self.seq_len):
            input_tensor = torch.tensor(pred_data[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)

            last_output = output[0, -self.pred_len:].cpu().numpy()
            predictions.append(last_output)
            pred_data = np.vstack([pred_data, last_output])
        predictions = np.vstack(predictions)
        return predictions

    def plot_test_data(self):
        #model = model.load_state_dict(torch.load(ckpt_path))
        pred_data = self.generate_predictions(self.model)
        print("Legnth of test data: ",len(self.test_data))
        fig = plt.figure()
        ax = plt.gca()
        img = mpimg.imread(self.court_path)  # read image. I got this image from gmf05's github.
        plt.imshow(img, extent=[0,94,0,50], zorder=0)  # show the image.
        plt.xlim([0,94])
        plt.ylim([0,50])
        plt.tight_layout(pad=0, w_pad=0.5, h_pad=0)
        ball_x = self.test_data[:,2] #4,5 ball x,y 6-
        ball_y = self.test_data[:,3]

        pred_ball_x = self.pred_data[:,2] #4,5 ball x,y 6-
        pred_ball_y = self.pred_data[:,3]

        home_color = '#E13A3E'
        away_color = '#008348'

        pred_home_color = '#F09C9E'
        pred_away_color = '#7FC1A3'

        if self.scale:
            rescale_x = 50.0
            rescale_y = 100.0
        else:
            rescale_x = 1.0
            rescale_y = 1.0   
        player_x = self.test_data[:,list(range(6,11))+list(range(11,16))]*rescale_x
        player_y = self.test_data[:,list(range(16,21))+list(range(21,26))]*rescale_y

        pred_player_x = self.pred_data[:,list(range(6,11))+list(range(11,16))]*rescale_x
        pred_player_y = self.pred_data[:,list(range(16,21))+list(range(21,26))]*rescale_y

        for t in range(int(len(self.test_data)*0.1)):
            pred_ball_circ = plt.Circle((pred_ball_x[t], pred_ball_y[t]), 0, color=[1, 0.4, 0])  # create circle object for bal
            ball_circ = plt.Circle((ball_x[t], ball_y[t]), 0, color=[1, 0.4, 0])  # create circle object for bal
            ax.add_artist(pred_ball_circ)
            ax.add_artist(ball_circ)
            l = [0,6] #[0,1,2,3,4,5,6,7,8,9]
            for a in l:
                if a<5: #home x & y
                    color = home_color
                    pred_color = pred_home_color
                else:
                    color = away_color
                    pred_color = pred_away_color

                p_x = player_x[t,a]
                p_y =player_y[t,a]
                pred_p_x = pred_player_x[t,a]
                pred_p_y =pred_player_y[t,a]
                player_circ = plt.Circle((p_x,p_y), 0.3,
                    facecolor=color,edgecolor='k')
                pred_player_circ = plt.Circle((pred_p_x,pred_p_y), 0.3,
                    facecolor=color,edgecolor='k')
                ax.add_artist(player_circ)
                ax.add_artist(pred_player_circ)
        plt.show()







