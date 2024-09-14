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

# server_ip_iperf = '172.202.73.177'  # Chicago
server_ip_iperf = '172.179.105.242'  # Victoria
# server_ip_iperf = '20.52.253.103'    # osnabrueck

server_ip_ping = '8.8.8.8'

def log_to_file(message, log_file):
    with open(log_file, 'a') as file:
        file.write(f"{message}\n")
        file.flush()
    # print(message)

def run_iperf3(folder_path):
    log_file = os.path.join(folder_path, 'network_measurements.log')
    # iperf3_command = ['which', 'iperf3']
    iperf3_command = ['iperf3', '-c', server_ip_iperf, '-R', '-P', '20', '-t', '0', '-i', '1', '-f', 'm', '--forceflush']  # -t set to 10 for demonstration
    process = subprocess.Popen(iperf3_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in process.stdout:
        # print(line)
        timestamp_with_ms = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
        if "[SUM]" in line:
            # print(line)
            throughput = re.search(r'\d+(\.\d+)? Mbits/sec', line).group(0)
            # timestamp_with_ms = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            log_to_file(f"{timestamp_with_ms} TP: {throughput}", log_file)

def restart_required(zero_throughput_time, zero_threshold_seconds):
    """Check if the zero throughput condition has exceeded the threshold."""
    if zero_throughput_time is None:
        return False
    return (datetime.now() - zero_throughput_time).total_seconds() >= zero_threshold_seconds

def run_iperf3_new(folder_path, zero_threshold_seconds=600):
    """Run iperf3 network performance testing and restart if throughput stays at 0."""
    log_file = os.path.join(folder_path, 'network_measurements.log')
    iperf3_command = [
        '/usr/local/bin/iperf3', '-c', server_ip_iperf, '-R', '-P', '20', '-t', '0', '-i', '1', '-f', 'm',
        '--forceflush'
    ]  # Infinite duration

    zero_throughput_time = None

    while True:
        # Start iperf3 process
        process = subprocess.Popen(iperf3_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True)
        for line in process.stdout:
            timestamp_with_ms = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            throughput_match = re.search(r'\d+(\.\d+)? Mbits/sec', line)

            if throughput_match and "[SUM]" in line:
                throughput_value = float(throughput_match.group(0).split()[0])
                if throughput_value == 0:
                    if zero_throughput_time is None:
                        zero_throughput_time = datetime.now()
                else:
                    zero_throughput_time = None  # Reset if throughput is above zero

                log_to_file(f"{timestamp_with_ms} TP: {throughput_match.group(0)}", log_file)

            # Check if we need to restart due to prolonged zero throughput
            if restart_required(zero_throughput_time, zero_threshold_seconds):
                print('Restarting iperf3 due to zero throughput')
                process.terminate()
                process.wait()  # Ensure the current process is fully terminated before restarting
                os.system('pkill -f iperf3')
                break

def ping_continuously(folder_path):
    log_file = os.path.join(folder_path, 'network_measurements.log')

    ping_command = ['ping', '-i', '1', server_ip_ping]
    echo_command = ['echo', 'power4ever']  # Highly insecure

    # Combine echo and ping commands
    command = echo_command + ['|', 'sudo', '-S'] + ping_command
    command = ping_command

    try:
        # Convert command to string and use shell=True for piping
        command_str = ' '.join(command)
        process = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, bufsize=1)

        for line in process.stdout:
            # print(line)
            if "time=" in line:
                # print(line)
                latency = re.search(r'time=([^ ]+)', line).group(1)
                timestamp_with_ms = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
                log_to_file(f"{timestamp_with_ms} Lt: {latency} ms", log_file)
    except KeyboardInterrupt:
        process.kill()


def plot_real_time(log_file):
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    def update(frame):
        data = pd.read_csv(log_file, delimiter=' ', header=None, names=['datetime', 'ms', 'type', 'value', 'unit'])
        data['timestamp'] = pd.to_datetime(data['datetime'] + '.' + data['ms'])

        tp_data = data[data['type'] == 'TP:']
        lt_data = data[data['type'] == 'Lt:']
        tp_data = tp_data.resample('1S').mean()
        lt_data = lt_data.resample('1S').mean()

        ax[0].clear()
        ax[1].clear()

        if not tp_data.empty:
            ax[0].plot(tp_data['timestamp'], tp_data['value'].astype(float), label='Throughput (Mbits/sec)')
            ax[0].set_title('Real-Time Throughput')
            ax[0].set_xlabel('Time')
            ax[0].set_ylabel('Throughput (Mbits/sec)')
            ax[0].legend()
            ax[0].grid(True)

        if not lt_data.empty:
            ax[1].plot(lt_data['timestamp'], lt_data['value'].str.replace('ms', '').astype(float), label='Latency (ms)',
                       color='orange')
            ax[1].set_title('Real-Time Latency')
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Latency (ms)')
            ax[1].legend()
            ax[1].grid(True)

        plt.tight_layout()

    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    log_file = './data/network_measurements.log'

    if os.path.exists(log_file):
        os.remove(log_file)
    folder_path = f'./data'

    # Set start method for multiprocessing
    set_start_method('spawn')

    # Create and start separate processes for iperf3 and ping
    iperf_process = Process(target=run_iperf3, args=(folder_path,))
    ping_process = Process(target=ping_continuously, args=(folder_path,))
    # plot_process = Process(target=plot_real_time, args=(log_file,))

    iperf_process.start()
    ping_process.start()
    # plot_process.start()

    try:
        # Wait for processes to complete (they won't unless manually stopped)
        iperf_process.join()
        ping_process.join()
        # plot_process.join()

    except KeyboardInterrupt:
        print("Measurement stopped by user.")
        iperf_process.terminate()
        ping_process.terminate()
        # plot_process.terminate()

