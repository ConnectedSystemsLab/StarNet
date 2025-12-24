import os
import subprocess
import threading
import time
import re
from datetime import datetime
from multiprocessing import Process, set_start_method
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

log_file = './data/network_measurements.log'


def update_plot(frame, folder_path, ax_image, ax3, ax2, ax1, ax4):
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        data = pd.read_csv(log_file, delimiter=' ', header=None, names=['datetime', 'ms', 'type', 'value', 'unit'])
        data['timestamp'] = pd.to_datetime(data['datetime'] + '-' + data['ms'])
        data.set_index('timestamp', inplace=True)
        time_window = pd.Timestamp.now() - pd.Timedelta(seconds=15*60)

        # Filter data for the last 15 seconds
        tp_data = data[(data['type'] == 'TP:') & (data.index > time_window)]
        lt_data = data[(data['type'] == 'Lt:') & (data.index > time_window)]

        tp_data = tp_data.resample('1S').mean()
        lt_data = lt_data.resample('1S').mean()

        # Clear previous data in the axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        # Update the first subplot with the throughput data
        if not tp_data.empty:
            ax1.plot(tp_data.index, tp_data['value'], label='Throughput (Mbits/sec)')
            ax1.set_title('Real-Time Throughput')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Throughput (Mbits/sec)')
            ax1.legend()
            ax1.grid(True)

        # Update the second subplot with the latency data
        if not lt_data.empty:
            ax2.plot(lt_data.index, lt_data['value'], label='Latency (ms)', color='orange')
            ax2.set_title('Real-Time Latency')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Latency (ms)')
            ax2.legend()
            ax2.grid(True)

        # Example updates for the remaining subplots
        # ax3.plot(...)
        # ax4.plot(...)

        plt.tight_layout()


if __name__ == "__main__":
    log_file = './data/network_measurements.log'

    # if os.path.exists(log_file):
    #     os.remove(log_file)
    folder_path = './data'

    # Set start method for multiprocessing
    set_start_method('spawn')

    # Set up the figure and subplots
    fig = plt.figure(figsize=(10, 24))
    ax_image = plt.subplot2grid((4, 2), (0, 0), colspan=1)
    ax1 = plt.subplot2grid((4, 2), (0, 1), colspan=1)
    ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
    ax4 = plt.subplot2grid((4, 2), (3, 0), colspan=2)

    ani = FuncAnimation(fig, update_plot, fargs=(folder_path, ax_image, ax1, ax2, ax3, ax4), interval=1,
                        cache_frame_data=False)
    plt.show()
