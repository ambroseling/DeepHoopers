import torch
import numpy as np 
import torch.nn as nn
import math
import torch.autograd as Variable
from training.protocol import Protocol
from models.discrete_seq_models import GRUEncoderDecoder, LSTM, BiLSTM
from models.spatial_models import GNNLSTM,GNNBiLSTM,GNNGRUED
from models.baseline import MLP
#You need to cd into preprocessing and then run main.py
def main():
    # print(torch.backends.mps.is_available())
    # print(torch.backends.mps.is_built())
    epoch = 30
    learning_rate = 0.01
    batch_size = 32
    num_features = 22
    seq_len = 50 
    target_len = 25
    pred_len = 25
    scale = True
    velocity = False
    graph = False
    freq = 5

    input_size = 22
    hidden_size = 64 
    num_layers = 3
    output_size = 22
    fc_units_bi = [128,64,22]
    fc_units = [64,32,22]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # lstm = LSTM(input_size, hidden_size, fc_units, output_size, num_layers,device)
    # bilstm = BiLSTM(input_size, hidden_size, fc_units_bi, output_size, num_layers,device)
    # grued = GRUEncoderDecoder(num_layers, input_size, hidden_size, fc_units, output_size,device)
    


    batch_size = 32
    seq_len = 50
    # lstm_layers = 3
    # gat_layers = 5
    # input_heads = 4
    # output_size = 22
    # hidden_size = [64,128]
    # fc_units = [64,32,22]
    # activation = "elu"
    # device = "cpu"
    # aggr = "mean"
    # gat = "gcn"

    # gcnlstm = GNNLSTM(batch_size=32,seq_len = 50,lstm_layers = 3,gat_layers=5, input_heads=4,output_size=22, hidden_size=[128,256],fc_units = [512,64,32,22], activation = "elu",device="cpu",aggr = "mean",gnn="gcn")
    # gcnlstm_protocol = Protocol(gcnlstm, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, graph, freq,device)
    # gcnlstm_protocol.load_data()
    # gcnlstm_protocol.train()
    # gcnlstm_protocol.eval()


    # gcnbilstm = GNNBiLSTM(batch_size=32,seq_len = 50, bilstm_layers = 3,gat_layers=5, input_heads=4,output_size=22, hidden_size=[64,128],fc_units =  [512,64,32,22], activation = "elu",device="cpu",aggr = "mean",gnn="gcn")
    # gcnbilstm_protocol = Protocol(gcnbilstm, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, graph, freq,device)
    # gcnbilstm_protocol.load_data()
    # gcnbilstm_protocol.train()
    # gcnbilstm_protocol.eval()


    # gcngrued = GNNGRUED(batch_size=32,seq_len = 50, gru_layers = 3,gat_layers=5, input_heads=4,output_size=22, hidden_size = [128,256],fc_units = [512,64,32,22], activation = "elu",device="cpu",aggr = "mean",gnn="gcn")
    # gcngrued_protocol = Protocol(gcngrued, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, graph, freq,device)
    # gcngrued_protocol.load_data()
    # gcngrued_protocol.train()
    # gcngrued_protocol.eval()



    # gatlstm = GNNLSTM(batch_size=32,seq_len = 50,lstm_layers = 3,gat_layers=5, input_heads=4,output_size=22, hidden_size=[128,256],fc_units = [512,64,32,22], activation = "elu",device="cpu",aggr = "mean",gnn="gat")
    # gatlstm_protocol = Protocol(gatlstm, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, graph, freq,device)
    # gatlstm_protocol.load_data()
    # gatlstm_protocol.train()
    # gatlstm_protocol.eval()


    # gatbilstm = GNNBiLSTM(batch_size=32,seq_len = 50, bilstm_layers = 3,gat_layers=5, input_heads=4,output_size=22, hidden_size=[64,128],fc_units =  [512,64,22], activation = "elu",device="cpu",aggr = "mean",gnn="gat")
    # gatbilstm_protocol = Protocol(gatbilstm, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, graph, freq,device)
    # gatbilstm_protocol.load_data()
    # gatbilstm_protocol.train()
    # gatbilstm_protocol.eval()


    # gatgrued = GNNGRUED(batch_size=32,seq_len = 50, gru_layers = 3,gat_layers=5, input_heads=4,output_size=22, hidden_size = [128,256],fc_units = [512,64,32,22], activation = "elu",device="cpu",aggr = "mean",gnn="gat")
    # gatgrued_protocol = Protocol(gatgrued, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, graph, freq,device)
    # gatgrued_protocol.load_data()
    # gatgrued_protocol.train()
    # gatgrued_protocol.eval()





    # lstm_protocol = Protocol(lstm, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, freq)
    # lstm_protocol.load_data()
    # lstm_protocol.train()
    # lstm_protocol.eval()

    # bilstm_protocol = Protocol(bilstm, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, freq)
    # bilstm_protocol.load_data()
    # bilstm_protocol.train()
    # bilstm_protocol.eval()
    
    # grued_protocol = Protocol(grued, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, freq)
    # grued_protocol.load_data()
    # grued_protocol.plot_test_data()

    # grued_protocol.train()
    # grued_protocol.eval()
    #mlp = MLP(num_features,device)


    # lstm = LSTM(input_size, 512, [512,64,22], output_size, num_layers,device)
    # checkpoint = torch.load("/Users/ambroseling/Desktop/DeepHoopers/DeepHoopers/checkpoints/LSTM_bs32_lr0.003_win50-25-25_vFalse_scTrue_ep22.pth",map_location="cpu")
    # print(checkpoint.keys())
    # lstm.load_state_dict(checkpoint['model_state_dict'],strict=False)

    bilstm = BiLSTM(22, 512, [1024,256,64,22], 22, 3,device)
    checkpoint = torch.load("/Users/ambroseling/Desktop/DeepHoopers/DeepHoopers/checkpoints/BiLSTM_bs32_lr0.001_win50-25-25_vFalse_scTrue_ep21.pth",map_location="cpu")
    print(checkpoint.keys())
    bilstm.load_state_dict(checkpoint['model_state_dict'],strict=False)
    lstm_protocol = Protocol(bilstm, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, freq,graph,device)
    lstm_protocol.load_data()
    lstm_protocol.plot_test_data()
    lstm_protocol.plot_test_data()
    lstm_protocol.plot_test_data()

    

    return


if __name__ == "__main__":
    main()