import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def pad_trace(trace, max_length=40):
    n_pad = max_length - len(trace)
    while n_pad > 0:
        trace.append(np.zeros_like(trace[0]))
        n_pad -= 1
    return np.array(trace)


def ensure_15_values(x):
    if len(x) < 15:
        avg_value = np.mean(x)
        x.extend([avg_value] * (15 - len(x)))
    return x

def assign_chunk(timestamp):
    second = timestamp.second
    minute = timestamp.minute
    hour = timestamp.hour
    month = timestamp.month
    day = timestamp.day

    if 12 <= second < 27:
        return f"{month:02d}-{day:02d}-{hour:02d}:{minute:02d}:12-26"
    elif 27 <= second < 42:
        return f"{month:02d}-{day:02d}-{hour:02d}:{minute:02d}:27-41"
    elif 42 <= second < 57:
        return f"{month:02d}-{day:02d}-{hour:02d}:{minute:02d}:42-56"
    else:
        return f"{month:02d}-{day:02d}-{hour:02d}:{minute:02d}:57-12"


def visualize_predictions(epoch, X_seq_tp, y_seq, outputs, folder_path):
    # Flatten x_seq_tp for plotting
    x_seq_tp_flat = X_seq_tp.cpu().numpy().flatten()

    plt.figure(figsize=(12, 6))

    # Plot the predicted throughput
    plt.plot(np.concatenate((x_seq_tp_flat, outputs.detach().cpu().numpy().flatten())),
             label='Predicted Throughput', color='red')

    # Plot the true throughput
    plt.plot(np.concatenate((x_seq_tp_flat, y_seq.cpu().numpy().flatten())),
             label='True Throughput', color='blue')

    plt.xlabel('Time Steps')
    plt.ylabel('Throughput')
    plt.legend()
    plt.title(f'Epoch {epoch} Predictions')
    plt.savefig(f'{folder_path}/pics/epoch_{epoch}.png')
    plt.close()

def plot_and_save_losses(epoch, train_losses, val_losses, val_maes, folder_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss up to Epoch {epoch}')
    plt.savefig(f'{folder_path}/loss_train_test.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_maes)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'MAE Loss up to Epoch {epoch}, Best MAE: {min(val_maes):.2f}')
    plt.savefig(f'{folder_path}/loss_MAE.png')
    plt.close()

    np.save(f'{folder_path}/train_losses.npy', train_losses)
    np.save(f'{folder_path}/train_losses.npy', val_losses)

import os
def showAttention(X_seq_tp, X_seq_attr, outputs, attentions, folder_path, dic_attentions, dic_2d, epoch):
    os.makedirs(folder_path+'/pics_attention/', exist_ok=True)
    for i in range(5):
        time_output = X_seq_attr[i, :, -4].cpu().detach().numpy()*14
        time_input = np.arange(0, len(outputs[0])) + 1 + time_output[-1]
        # Create the attention heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(attentions[i].cpu().detach().numpy(), cmap='bone', aspect='auto')
        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(time_output)))
        ax.set_yticks(np.arange(len(time_input)))
        ax.set_xticklabels(time_output.astype(int))
        ax.set_yticklabels(time_input.astype(int))
        ax.set_xlabel('Time Output')
        ax.set_ylabel('Time Input')
        plt.savefig(folder_path+f'/pics_attention/{i}.png')
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(X_seq_tp[i, :].cpu().detach().numpy())
        plt.savefig(folder_path+f'/pics_attention/{i}_tp.png')
        plt.close()

    matrix = [[0 for i in range(15)] for j in range(15)]
    for i in range(15):
        for j in range(15):
            matrix[i][j] = np.mean(dic_2d[(j, i)])
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='bone', aspect='auto')
    # plt.show()
    # plt.colorbar()
    plt.savefig(folder_path+f'/pics_attention/attentions_2d_{epoch}.png')
    np.save(folder_path+f'/pics_attention/attentions_2d_{epoch}.npy', matrix)
    plt.close()


    array = [np.mean(dic_attentions[k]) for k in range(15)]
    plt.figure(figsize=(10, 8))
    plt.plot(array)
    np.save(folder_path+f'/attentions.npy', dic_attentions)
    plt.savefig(folder_path+f'/pics_attention_mean.png')
    plt.close()
    return


def save_best_model(model, epoch, best_val_mae, val_mae, folder_path, mae_all, model_name='pretrain'):
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), folder_path+f'/models/{model_name}.pt')
        np.save(folder_path+f'/models/mae_all.npy', mae_all)
        print(f"Model saved at epoch {epoch} with MAE {val_mae:.4f}")
    return best_val_mae

class SatelliteSequenceDataset(Dataset):
    def __init__(self, traces, window_size=3, columns=None, object='throughput'):
        self.traces = traces.reset_index(drop=True)
        self.window_size = window_size
        self.object = object
        self.columns = columns
        self.object = object
    def __len__(self):
        return len(self.traces) - self.window_size

    def __getitem__(self, idx):
        X_seq_tp = np.hstack(self.traces.iloc[idx:idx + self.window_size][self.object].to_numpy())
        X_seq_attr = [self.traces.iloc[idx + self.window_size][col] for col in self.columns]
        y_seq = np.array(self.traces.iloc[idx + self.window_size][self.object]).astype(np.float32)

        X_seq_tp = torch.ones_like(torch.tensor(X_seq_tp, dtype=torch.float32))
        return (torch.tensor(X_seq_tp, dtype=torch.float32), torch.tensor(X_seq_attr, dtype=torch.float32),
                torch.tensor(y_seq, dtype=torch.float32))


class SatelliteSequenceDataset_Continous(Dataset):
    def __init__(self, traces, input_len=3, output_len=15, step_len=10, columns=None, object='throughput'):
        self.traces = traces.reset_index(drop=True)
        self.input_len = input_len
        self.step_len = step_len
        self.output_len = output_len
        self.object = object
        self.columns = columns
        self.object = object

        self.scalers = {col: MinMaxScaler() for col in self.columns}
        self.scalers[self.object] = MinMaxScaler()
        for col in self.columns:
            self.traces[col] = self.scalers[col].fit_transform(self.traces[[col]])
        self.traces[self.object] = self.scalers[self.object].fit_transform(self.traces[[self.object]])
        # self.analysis()

    def __len__(self):
        return (len(self.traces) - self.input_len - self.output_len)//self.step_len

    def __getitem__(self, idx):
        input_start = idx * self.step_len
        input_end = input_start + self.input_len
        output_start = input_end
        output_end = output_start + self.output_len

        X_seq_tp = self.traces.iloc[input_start:input_end][self.object].to_numpy()
        X_seq_attr = np.array([self.traces.iloc[input_start:input_end][col].to_numpy() for col in self.columns]).swapaxes(0, 1)
        # X_seq_attr = np.array([self.traces.iloc[output_start:output_end][col].to_numpy() for col in self.columns]).swapaxes(0, 1)
        # interests = [3, 4]
        # for i in interests:
        #     X_seq_attr[:, i] = np.ones_like(X_seq_attr[:, i]) * self.traces.iloc[output_start][self.columns[i]]

        # X_seq_tp = np.zeros_like(X_seq_tp)
        # X_seq_attr = np.zeros_like(X_seq_attr)
        y_seq = self.traces.iloc[output_start:output_end][self.object].to_numpy()

        return (torch.tensor(X_seq_tp, dtype=torch.float32), torch.tensor(X_seq_attr, dtype=torch.float32),
                torch.tensor(y_seq, dtype=torch.float32))

    def analysis(self):
        X = self.traces[self.columns]
        y = self.traces[self.object]

        # Sample 10% of the data
        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.9, random_state=42)

        # Split the sampled data into training and testing sets (e.g., 80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

        # Train a RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Get feature importances
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'feature': self.columns, 'importance': importances})

        # Print the feature importances
        print(feature_importance.sort_values(by='importance', ascending=False))

        # Check if the importances sum up to 1
        total_importance = sum(importances)
        print(f"Total feature importance: {total_importance}")


def get_data_loader(dataset_path, input_len, output_len, step_len, tr_batchsize, columns_to_exclude, shuffle=False):
    combined_df = pd.read_pickle(dataset_path).dropna()
    # subset_fraction = 0.2
    # combined_df = combined_df[:len(combined_df)//20]
    # Preparing for Dataset
    print('Preparing data...')
    combined_df['t_s'] = (combined_df['timestamp'].dt.second - 12) % 15
    combined_df['t_d_of_w'] = combined_df['timestamp'].dt.dayofweek
    combined_df['t_minute'] = combined_df['timestamp'].dt.minute
    combined_df['t_hour'] = combined_df['hour']
    label_encoder = LabelEncoder()
    combined_df['sat_name'] = label_encoder.fit_transform(combined_df['sat_name'])

    selected_columns = [
        'timestamp', 'throughput', 'latency',
        'alt', 'az', 'distance', 'sat_name','n_candidates',
        'clouds', 'pressure', 'humidity',
        't_s', 't_minute', 't_hour', 't_d_of_w'
    ]

    dataset = combined_df[selected_columns]

    selected_columns = [col for col in dataset.columns if col not in columns_to_exclude]
    feature_size = len(selected_columns)

    if 'all' in dataset_path:
        train_df = dataset[combined_df['location'].isin(['vic', 'osn'])]
        val_df = dataset[~combined_df['location'].isin(['vic', 'osn'])]
        train_dataset = SatelliteSequenceDataset_Continous(train_df, input_len=input_len, output_len=output_len, step_len=step_len, columns=selected_columns)
        val_dataset = SatelliteSequenceDataset_Continous(val_df, input_len=input_len, output_len=output_len, step_len=step_len, columns=selected_columns)
    else:
        satellite_dataset = SatelliteSequenceDataset_Continous(dataset, input_len=input_len, output_len=output_len, step_len=step_len, columns=selected_columns)
        train_size = int(0.8 * len(satellite_dataset))
        val_size = len(satellite_dataset) - train_size

        if shuffle:
            train_dataset, val_dataset = torch.utils.data.random_split(satellite_dataset, [train_size, val_size])
        else:
            train_dataset = torch.utils.data.Subset(satellite_dataset, range(train_size))
            val_dataset = torch.utils.data.Subset(satellite_dataset, range(train_size, len(satellite_dataset)))

    # if subset_fraction < 1.0:
    #     subset_size = int(subset_fraction * train_size)
    #     train_indices = np.random.choice(range(train_size), size=subset_size, replace=False)
    #     train_dataset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_dataset, batch_size=tr_batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    print('Number of Batches:', len(train_loader))
    return train_loader, val_loader, feature_size, train_dataset


def get_eval_data_loader(dataset_path, input_len, output_len, step_len, tr_batchsize, columns_to_exclude):
    combined_df = pd.read_pickle(dataset_path).dropna()
    print('Preparing data...')
    time_diff = combined_df['timestamp'].diff().dt.total_seconds()
    combined_df['chunk'] = combined_df['timestamp'].apply(assign_chunk)

    combined_df['t_s'] = (combined_df['timestamp'].dt.second - 12) % 15
    combined_df['t_d_of_w'] = combined_df['timestamp'].dt.dayofweek
    combined_df['t_minute'] = combined_df['timestamp'].dt.minute
    combined_df['t_hour'] = combined_df['hour']
    label_encoder = LabelEncoder()
    combined_df['sat_name'] = label_encoder.fit_transform(combined_df['sat_name'])

    selected_columns = [
        'timestamp', 'throughput', 'latency',
        'alt', 'az', 'distance', 'sat_name', 'n_candidates',
        'clouds', 'pressure', 'humidity',
        't_s', 't_minute', 't_hour', 't_d_of_w'
    ]

    dataset = combined_df[selected_columns]

    selected_columns = [col for col in dataset.columns if col not in columns_to_exclude]
    feature_size = len(selected_columns)
    satellite_dataset = SatelliteSequenceDataset_Continous(dataset, input_len=input_len, output_len=output_len,
                                                           step_len=4, columns=selected_columns)
    train_size = int(0.8 * len(satellite_dataset))
    al_dataset = torch.utils.data.Subset(satellite_dataset, range(train_size, len(satellite_dataset)))
    val_loader = DataLoader(al_dataset, batch_size=256, shuffle=False)
    print('dataloader length: ', len(val_loader))
    return val_loader, feature_size

if __name__ == '__main__':
    a = 1