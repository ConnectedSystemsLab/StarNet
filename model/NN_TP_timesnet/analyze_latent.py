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
from datetime import  datetime
import shutil
from sklearn.manifold import TSNE

os.makedirs("runs", exist_ok=True)
predict_object = 'throughput'  # latency/throughput
dataset_path = '../../data_4_26/dataset_tp_sat.pkl'

# Sequence Parameters
input_len = 30
output_len = 15
step_len = input_len + output_len
columns_to_exclude = ['timestamp', 'latency', 'throughput',

                      # # Satellite Info
                      # 'alt',
                      # 'az',
                      # 'distance',
                      # 'sat_name',
                      # 'n_candidates',
                      #
                      # # Weather Info
                      # 'clouds',
                      # 'pressure',
                      # 'humidity',
                      #
                      # # Time Info
                      # 't_s',
                      # 't_minute',  # List of minute timestamps
                      # 't_hour',  # List of hour timestamps
                      # 't_d_of_w'
                      ]  # attributes  # attributes

# Model Parameters
n_fea_expanded = 4
hidden_size = 128
num_layers = 1
dropout = 0.0
positional_scale = 213

# Training Parameters
lr = 0.001
tr_batchsize = 512


if '__main__' == __name__:
    model_path = './runs/run_41_87/models/task.pt'

    train_loader, val_loader, feature_size, satellite_dataset = get_data_loader(dataset_path, input_len, output_len, step_len,
                                                             tr_batchsize, columns_to_exclude)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SatelliteGRU_attention(input_len, hidden_size, num_layers, output_len, feature_size, device
                                 , positional_scale, teacher_forcing_ratio=0.0, n_fea_expanded=n_fea_expanded,
                                 dropout=dropout)
    if torch.cuda.device_count() > 1:
        print(f"**********{torch.cuda.device_count()} GPUs*********")
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    count_parameters(model)

    model.eval()

    latent_vectors = []
    distance_classes = []

    for i, (X_seq_tp, X_seq_attr, y_seq) in enumerate(train_loader):
        X_seq_tp = X_seq_tp.to(device)
        X_seq_attr = X_seq_attr.to(device)
        outputs, latent = model.encoder_forward(X_seq_tp, X_seq_attr)
        latent = latent.permute(1, 0, 2).contiguous().view(latent.size(1), -1)
        latent_vectors.append(latent.cpu().detach().numpy())

        tp = outputs.cpu().detach().numpy().mean(1)*500
        tp_bin = np.linspace(0, 400, 5)
        tp_class = np.digitize(tp, tp_bin)-1

        # X_seq_attr = X_seq_attr.mean(1)
        # distance = X_seq_attr[:, 2].cpu().numpy() * stds['distance'] + means['distance']
        # # distance_bin = np.linspace(290, 1000, 5)
        # distance_bin = [645]
        # distance_class = np.digitize(distance, distance_bin) - 1
        # distance_classes.extend(distance_class)
        distance_classes.extend(tp_class)

    latent_vectors = np.vstack(latent_vectors)
    distance_classes = np.array(distance_classes)

    # Limit each class to 100 samples
    unique_classes = np.unique(distance_classes)
    sampled_latent_vectors = []
    sampled_distance_classes = []

    for cls in unique_classes:
        indices = np.where(distance_classes == cls)[0]
        if len(indices) > 500:
            indices = np.random.choice(indices, 500, replace=False)
        sampled_latent_vectors.append(latent_vectors[indices])
        sampled_distance_classes.append(distance_classes[indices])

    sampled_latent_vectors = np.vstack(sampled_latent_vectors)
    sampled_distance_classes = np.concatenate(sampled_distance_classes)

    # Apply t-SNE to reduce the dimensionality of latent vectors
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(sampled_latent_vectors)

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=sampled_distance_classes, cmap='viridis', alpha=0.7, s=15)
    plt.colorbar(scatter, label='Distance Class')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE of Latent Space Colored by Distance Classes')
    plt.show()
    data_save = {'latent_tsne': latent_tsne, 'sampled_distance_classes': sampled_distance_classes}
    np.save('latent.npy', data_save)


