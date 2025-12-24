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
import optuna

os.makedirs("pics", exist_ok=True)
predict_object = 'throughput'  # latency/throughput
dataset_path = '../../data_4_26/dataset_tp_sat.pkl'

# Sequence Parameters
window_size = 15
output_size = 15
columns_to_exclude = ['timestamp', 'latency', 'throughput']  # attributes

# Model Parameters
n_fea_expanded = 4
hidden_size = 128
num_layers = 2
dropout = 0.2
positional_scale = 200

# Training Parameters
lr = 0.01
tr_batchsize = 256

train_loader, val_loader, feature_size = get_data_loader(dataset_path, window_size, tr_batchsize, columns_to_exclude)


def objective(trial):
    # Hyperparameter search space
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
    num_layers = trial.suggest_int('num_layers',1, 4)
    dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.3, 0.5])
    lr = trial.suggest_float('lr', 1e-4, 0.1, log=True)
    positional_scale = trial.suggest_int('positional_scale', 0, 400)

    # train_loader, val_loader, feature_size = get_data_loader(dataset_path, window_size, tr_batchsize,
    #                                                          columns_to_exclude)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SatelliteGRU_seq2seq(window_size * 15, hidden_size, num_layers, output_size, feature_size, device
                                 , positional_scale, teacher_forcing_ratio=0.0, n_fea_expanded=n_fea_expanded, dropout=dropout)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    # count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    val_loss = train_test(model, train_loader, val_loader, device, optimizer, n_epochs=20)
    return val_loss


if '__main__' == __name__:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = SatelliteGRU_seq2seq(window_size * 15, hidden_size, num_layers, output_size, feature_size, device
    #                              ,positional_scale, teacher_forcing_ratio=0.0, n_fea_expanded=n_fea_expanded, dropout=dropout)
    # if torch.cuda.device_count() > 1:
    #     print(f"**********{torch.cuda.device_count()} GPUs*********")
    #     model = nn.DataParallel(model)
    # model.to(device)
    # count_parameters(model)
    #
    # optimizer = optim.AdamW(model.parameters(), lr=lr)
    # train_test(model, train_loader, val_loader, device, optimizer)
    #

