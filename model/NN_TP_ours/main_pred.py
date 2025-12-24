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
import os
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
from train_test import *
from datetime import datetime
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a seq2seq model for predicting Starlink network bandwidth.")
    parser.add_argument('--tag', type=str, default='',
                        help='Comments to the Exp')
    parser.add_argument('--dataset_path', type=str, default='../../data_4_26/dataset_tp_sat.pkl',
                        help='Path to the dataset')
    parser.add_argument('--n_epochs', type=int, default=25, help='No. of training epochs')
    parser.add_argument('--random', action='store_true', help='Whether to use random dataset splitting')

    parser.add_argument('--input_len', type=int, default=30, help='Input sequence length')
    parser.add_argument('--output_len', type=int, default=15, help='Output sequence length')
    parser.add_argument('--step_len', type=int, default=46, help='Step length for sequence generation')
    parser.add_argument('--columns_to_exclude', nargs='+', default=['timestamp', 'latency', 'throughput'],
                        help='Columns to exclude from the dataset')

    # Not applicable to Baselines
    parser.add_argument('--n_fea_expanded', type=int, default=12, help='Number of expanded features')
    parser.add_argument('--hidden_size', type=int, default=60, help='Hidden size of the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--positional_scale', type=int, default=213, help='Positional encoding scale')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--tr_batchsize', type=int, default=512, help='Training batch size')

    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs("runs", exist_ok=True)
    start_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    folder_path = f'./runs/{args.tag}_{start_time}'
    print(folder_path)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path + '/pics', exist_ok=True)
    os.makedirs(folder_path + '/models', exist_ok=True)

    train_loader, val_loader, feature_size, satellite_dataset = get_data_loader(args.dataset_path, args.input_len, args.output_len, args.step_len,
                                                             args.tr_batchsize, args.columns_to_exclude, args.random)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SatelliteGRU_attention(args.input_len, args.hidden_size, args.num_layers, args.output_len, feature_size, device,
                        n_fea_expanded=args.n_fea_expanded)

    if torch.cuda.device_count() > 1:
        print(f"**********{torch.cuda.device_count()} GPUs*********")
        model = nn.DataParallel(model)
    model.to(device)
    # model.load_state_dict(torch.load('./runs/G_all-ours_outlen15_2024-08-06-02-21/models/task.pt'))
    # model.freeze_encoder()

    # count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # pretrain(model, train_loader, val_loader, device, optimizer, folder_path)
    train_test(model, train_loader, val_loader, device, optimizer, folder_path, n_epochs=args.n_epochs)

if __name__ == '__main__':
    main()
