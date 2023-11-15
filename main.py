import torch
import numpy as np 
import torch.nn as nn
import math
import torch.autograd as Variable
from training.protocol import Protocol
from models.discrete_seq_models import GRUEncoderDecoder, LSTM, BiLSTM
from models.baseline import MLP

#You need to cd into preprocessing and then run main.py
def main():
    # print(torch.backends.mps.is_available())
    # print(torch.backends.mps.is_built())
    epoch = 20
    learning_rate = 0.01
    batch_size = 32
    num_features = 22
    seq_len = 50 
    target_len = 25
    pred_len = 25
    scale = True
    velocity = False
    freq = 5

    input_size = 22
    hidden_size = 64 
    num_layers = 3
    output_size = 22
    fc_units_bi = [128,64,22]
    fc_units = [64,32,22]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lstm = LSTM(input_size, hidden_size, fc_units, output_size, num_layers,device)
    bilstm = BiLSTM(input_size, hidden_size, fc_units_bi, output_size, num_layers,device)
    grued = GRUEncoderDecoder(num_layers, input_size, hidden_size, fc_units, output_size,device)
    mlp = MLP(num_features,device)
    

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

    mlp_protocol = Protocol(mlp, epoch, learning_rate, batch_size, num_features, seq_len, pred_len, target_len, scale, velocity, freq,device)
    mlp_protocol.load_data()
    # mlp_protocol.train()
    mlp_protocol.eval()

    return


if __name__ == "__main__":
    main()