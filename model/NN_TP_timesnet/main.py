import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
from sat_dataset import *
from model import *
import logging
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
from train_test import *
from TimesNet import *

os.makedirs("pics", exist_ok=True)
predict_object = 'throughput'  # latency/throughput
dataset_path = '../../data_4_26/dataset_tp_sat.pkl'

# Sequence Parameters
window_size = 15
output_size = 15
columns_to_exclude = ['timestamp', 'latency', 'throughput']  # attributes

# Model Parameters
n_fea_expanded = 4
hidden_size = 512
num_layers = 2
dropout = 0.0
positional_scale = 213

# Training Parameters
lr = 0.001
tr_batchsize = 256


if '__main__' == __name__:
    train_loader, val_loader, feature_size = get_data_loader(dataset_path, window_size, tr_batchsize, columns_to_exclude)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SatTimesNet(window_size * 15, hidden_size, num_layers, output_size, feature_size, device
                                 , positional_scale, teacher_forcing_ratio=0.0, n_fea_expanded=n_fea_expanded,
                                 dropout=dropout)
    if torch.cuda.device_count() > 1:
        print(f"**********{torch.cuda.device_count()} GPUs*********")
        model = nn.DataParallel(model)
    model.to(device)
    count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    train_test(model, train_loader, val_loader, device, optimizer)


