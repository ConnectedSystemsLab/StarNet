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
from sklearn.metrics import mean_absolute_error
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
    parser.add_argument('--output_len', type=int, default=5, help='Output sequence length')
    parser.add_argument('--step_len', type=int, default=29, help='Step length for sequence generation')
    parser.add_argument('--columns_to_exclude', nargs='+', default=['timestamp', 'latency', 'throughput'],
                        help='Columns to exclude from the dataset')

    # Not applicable to Baselines
    parser.add_argument('--n_fea_expanded', type=int, default=12, help='Number of expanded features')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--positional_scale', type=int, default=213, help='Positional encoding scale')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--tr_batchsize', type=int, default=512, help='Training batch size')

    return parser.parse_args()

os.makedirs("eval/data", exist_ok=True)
os.makedirs("eval/pics", exist_ok=True)
predict_object = 'throughput'  # latency/throughput


def evaluate_best_model(model, dataset, window_size, batch_size=256):
    model.eval()
    predictions = []
    true_values = []

    # Create DataLoader for the entire dataset
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for X_seq_tp_batch, X_seq_attr_batch, true_y_batch in eval_loader:
            X_seq_tp_batch = X_seq_tp_batch.to(device)
            X_seq_attr_batch = X_seq_attr_batch.to(device)
            true_y_batch = true_y_batch.to(device)

            outputs_batch = model(X_seq_tp_batch, X_seq_attr_batch)
            predictions.append(outputs_batch.cpu().numpy())
            true_values.append(true_y_batch.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(true_values)

if '__main__' == __name__:
    args = parse_args()
    model_name = 'chi-timesnet_outlen5_2024-08-02-02-00'
    model_path = f'./runs/{model_name}/models/task.pt'
    val_loader, feature_size = get_eval_data_loader(args.dataset_path, args.input_len,
                                                    args.output_len, args.output_len,
                                                    args.tr_batchsize,
                                                    args.columns_to_exclude)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SatTimesNet(args.input_len, args.hidden_size, args.num_layers, args.output_len, feature_size,
                                   device,
                                   n_fea_expanded=args.n_fea_expanded)
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    predictions = []
    true_values = []
    time = []
    from tqdm import tqdm

    # Create DataLoader for the entire dataset
    val_loader_tqdm = tqdm(val_loader)
    with torch.no_grad():
        for i, (X_seq_tp, X_seq_attr, y_seq) in enumerate(val_loader_tqdm):
            X_seq_tp = X_seq_tp.to(device)
            X_seq_attr = X_seq_attr.to(device)
            y_seq = y_seq.to(device)
            outputs = model(X_seq_tp, X_seq_attr)

            predictions.append(outputs.cpu().numpy() * 500)
            true_values.append(y_seq.cpu().numpy() * 500)

            tmp = X_seq_attr[:, -1, -4].cpu().numpy() * 14
            time_tmp = np.zeros([tmp.shape[0], 5]) + np.expand_dims(tmp, -1) + np.arange(5)
            time.append(time_tmp)
            # if i > 10:
            #     break
    predictions = np.concatenate(predictions).flatten()
    true_values = np.concatenate(true_values).flatten()
    time = np.concatenate(time).flatten()
    data = {'pred': predictions, 'gt': true_values, 'time': time}
    np.save(f'./eval/data/eval_data.npy', data)
    #
    # # Define the chunk size
    # chunk_size = 150
    # num_chunks = len(predictions) // chunk_size + (1 if len(predictions) % chunk_size != 0 else 0)
    #
    # # Plot each chunk
    # for i in range(num_chunks):
    #     start_idx = i * chunk_size
    #     end_idx = min((i + 1) * chunk_size, len(predictions))
    #
    #     chunk_true_values = true_values[start_idx:end_idx]
    #     chunk_predictions = predictions[start_idx:end_idx]
    #
    #     mae = mean_absolute_error(chunk_true_values, chunk_predictions)
    #
    #     plt.figure(figsize=(15, 5))
    #     plt.plot(chunk_true_values, label=f'True {predict_object}', color='blue')
    #     plt.plot(chunk_predictions, label=f'Predicted {predict_object}', color='red')
    #     plt.xlabel('Time(s)')
    #     plt.ylabel(f'{predict_object}')
    #     plt.legend()
    #     plt.title(f'Predictions on Entire Dataset (Chunk {i + 1})\nMAE: {mae:.4f}')
    #     plt.savefig(f'./eval/pics/predictions_chunk_{i + 1}.png')
    #     # plt.show()
    #     plt.close()