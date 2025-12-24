from collections import defaultdict

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
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time

def train_test(model, train_loader, val_loader, device, optimizer, folder_path, n_epochs=100000):
    criterion = nn.MSELoss()
    logging.basicConfig(
        filename=f'{folder_path}/training.log',  # Specify the desired log file name
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    mae_criterion = nn.L1Loss()
    print('Start training...')
    best_val_mae = float("inf")

    test_mae_loss = []
    train_losses = []
    val_losses = []
    best_mse = float('inf')

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for i, (X_seq_tp, X_seq_attr, y_seq) in enumerate(train_loader):
            X_seq_tp = X_seq_tp.to(device)
            X_seq_attr = X_seq_attr.to(device)
            y_seq = y_seq.to(device)
            outputs, _ = model(X_seq_tp, X_seq_attr)
            loss = criterion(outputs, y_seq)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        dic_attentions = defaultdict(list)
        dic_2d = defaultdict(list)
        mae_all = []
        time_all = []
        with torch.no_grad():
            for i, (X_seq_tp, X_seq_attr, y_seq) in enumerate(val_loader):
                X_seq_tp = X_seq_tp.to(device)
                X_seq_attr = X_seq_attr.to(device)
                y_seq = y_seq.to(device)

                start_time = time.time()
                outputs, attentions = model(X_seq_tp, X_seq_attr)
                end_time = time.time()
                time_all.append(end_time - start_time)

                attentions_value = attentions.detach().cpu().numpy().mean(1)
                attentions_tmp = attentions.detach().cpu().numpy()
                times = X_seq_attr[:, :, -4].detach().cpu().numpy()*14
                for batch in range(times.shape[0]):
                    for j, time_step in enumerate(times[batch]):
                        dic_attentions[int(time_step)].append(attentions_value[batch][j])
                        for jj in range(attentions.shape[1]):
                            dic_2d[(int(time_step), int(jj+times[batch][-1]+1)%15)].append(attentions_tmp[batch][jj][j])
                loss = criterion(outputs, y_seq)

                outputs = outputs*500
                y_seq = y_seq*500
                mae = mae_criterion(outputs, y_seq)
                mae_all.extend(nn.L1Loss(reduction='none')(outputs, y_seq).mean(1).cpu().numpy())
                mse = np.sqrt(torch.mean((outputs - y_seq) ** 2).item())  # Calculate MSE

                val_loss += loss.item()
                val_mae += mae.item()
                val_mse += mse  # Accumulate MSE
        showAttention(X_seq_tp, X_seq_attr, outputs, attentions, folder_path, dic_attentions, dic_2d, epoch)
        test_mae_loss.append(val_mae / len(val_loader))
        val_losses.append(val_loss / len(val_loader))
        best_val_mae = save_best_model(model, epoch, best_val_mae, val_mae / len(val_loader), folder_path, mae_all,
                                       model_name='task')
        plot_and_save_losses(epoch, train_losses, val_losses, test_mae_loss, folder_path)  # Pass val_mses

        scheduler.step()
        np.save(folder_path + '/time.npy', time_all)
        best_mse = min(best_mse, val_mse / len(val_loader))
        sentence = f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(train_loader):.4f}, " \
                   f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation MAE: {val_mae / len(val_loader):.4f}, " \
                   f"Validation MSE: {val_mse / len(val_loader):.4f} || Best MSE: {best_mse:.3f}, Best MAE: {best_val_mae:.3f}"
        logger.info(sentence)
        print(sentence)

        if epoch % 10 == 0:
            for i, (X_seq_tp, X_seq_attr, y_seq) in enumerate(val_loader):
                X_seq_tp = X_seq_tp.to(device)
                X_seq_attr = X_seq_attr.to(device)
                y_seq = y_seq.to(device)
                outputs = model(X_seq_tp, X_seq_attr)
                visualize_predictions(epoch, X_seq_tp[0], y_seq[0], outputs[0][0], folder_path=folder_path)
                break  # Visualize the first batch for simplicity
        epoch += 1
    return best_val_mae
