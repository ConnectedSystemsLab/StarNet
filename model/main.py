import subprocess
import os
from datetime import datetime
import logging


start_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
folder_path = f'./runs/{start_time}'
os.makedirs(folder_path, exist_ok=True)
log_filename = folder_path+"/run_log.log"
logging.basicConfig(filename=log_filename,
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Define the list of baselines and their directories
baselines = [
    # "itransformer",  # Replace with actual names of your baselines
    # "LightTS",
    # "LSTM",
    # "PatchTST",
    # "timesnet",
    # "xgboost",
    "ours",
]

# Define the common parameters for running the main_pred.py
common_params = {
    "n_epochs": 50,
}

all_output_len = [15]
all_input_len = [75]
all_datasets = {
    # 'chi': {'path': '../../data_4_26/dataset_tp_sat.pkl', 'step_len': 46},
    'vic': {'path': '../../data_global/data_victoria/dataset_tp_sat.pkl', 'step_len': 6},
    'osn': {'path': '../../data_global/data_osn/dataset_tp_sat.pkl', 'step_len': 29},
    # 'all': {'path': '../../data_global/data_all/dataset_tp_sat.pkl', 'step_len': 37},
}

# Construct the command string
def construct_command(baseline_name, input_len, output_len, dataset_name, dataset_info):
    cmd = ["python", "main_pred.py"]
    for key, value in common_params.items():
        cmd.append(f"--{key}={value}")
    cmd.append(f"--input_len={input_len}")
    cmd.append(f"--output_len={output_len}")
    cmd.append(f"--step_len={dataset_info['step_len']}")
    cmd.append(f"--dataset_path={dataset_info['path']}")
    cmd.append(f"--tag=G_{dataset_name}-{baseline_name}_inlen{input_len}_outlen{output_len}")

    # if dataset_name == 'all':
    #     cmd.append("--random")

    print('=============================')
    print(f'{dataset_name}-{baseline_name}_outlen{output_len}')
    print('=============================')
    logging.info('=============================')
    logging.info(f'{dataset_name}-{baseline_name}_outlen{output_len}')
    logging.info('=============================')
    return cmd

# Run the script for each baseline, output_len, and dataset
for dataset_name, dataset_info in all_datasets.items():
    for output_len in all_output_len:
        for input_len in all_input_len:
            for baseline in baselines:
                baseline_dir = f'./NN_TP_{baseline}'
                try:
                    if os.path.exists(baseline_dir):
                        os.chdir(baseline_dir)
                        cmd = construct_command(baseline, input_len, output_len, dataset_name, dataset_info)
                        print(f"Running command in {baseline_dir}: {' '.join(cmd)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        print(result.stdout)
                        print(result.stderr)
                        logging.info(result.stdout)
                        logging.error(result.stderr)
                        os.chdir("../")  # Change this to your current working directory
                    else:
                        print(f"Baseline directory {baseline_dir} does not exist.")
                except Exception as e:
                    print(e)