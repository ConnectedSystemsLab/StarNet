import requests
from skyfield.api import Topos, load
import matplotlib.pyplot as plt
import math
import pytz
import time
import argparse
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import starlink_grpc
from tp_lt import *
from utils import *
import shutil
global my_location, times, measure_trace
from tzlocal import get_localzone

# server_ip_iperf = '172.202.73.177'  # Replace with your server's IP address
# server_ip_ping = '8.8.8.8'
ground_station_coord = Topos(41.85486264725333, -87.85876625623841, elevation_m=190)  # Broadview, IL

# Function to be executed in parallel for each satellite
def process_satellites(satellite_chunk, my_location, times, measure_trace):
    result = []
    # global my_location, times, measure_trace  # Make sure these are defined and accessible
    sequence_1_cartesian = np.array([polar_to_cartesian(r, theta) for r, theta in measure_trace])
    sequence_1_cartesian.sort(axis=0)
    # plt.figure()
    # plt.scatter(sequence_1_cartesian[:, 0],sequence_1_cartesian[:, 1], color='blue')
    distances = np.linalg.norm(np.diff(sequence_1_cartesian, axis=0), axis=1)
    gap_index = np.argmax(distances)
    if max(distances) > 0.1:
        segment_1 = sequence_1_cartesian[:gap_index + 1]
        segment_2 = sequence_1_cartesian[gap_index + 1:]
        if len(segment_1) > len(segment_2):
            sequence_1_cartesian = segment_1
        else:
            sequence_1_cartesian = segment_2
    # print(max(distances))
    # plt.scatter(sequence_1_cartesian[:, 0],sequence_1_cartesian[:, 1], color='red')
    # plt.show()

    for sat in satellite_chunk:
        difference = sat - my_location
        difference_gs = sat - ground_station_coord

        sat_trace = []
        bypass = False
        for t in times:
            topocentric = difference.at(t)
            alt, az, distance = topocentric.altaz()

            topocentric2 = difference_gs.at(t)
            _, _, d2 = topocentric2.altaz()

            if alt.degrees > 30:
                sat_trace.append((alt.degrees, np.radians(az.degrees), distance.km, d2.km))
            else:
                bypass = True
                break
        if bypass:
            continue
        if len(sat_trace) > 0:
            # sat_trace = sorted(sat_trace, key=lambda point: point[0]*math.cos(point[1]))
            sat_trace = np.array(sat_trace)
            sequence_2_cartesian = np.array([polar_to_cartesian(r, theta) for r, theta, _, _ in sat_trace])
            distance, path = fastdtw(sequence_1_cartesian, sequence_2_cartesian, dist=euclidean)
            result.append([sat.name, sat_trace, distance, measure_trace])

            # print([sat.name, sat_trace, distance, path])
    return result


def divide_work(satellites, num_processes=4):
    chunk_size = len(satellites) // num_processes
    chunks = [satellites[i:i + chunk_size] for i in range(0, len(satellites), chunk_size)]

    # Ensure all satellites are covered (due to integer division)
    if len(chunks) > num_processes:
        # Append the remaining satellites to the last chunk
        chunks[num_processes - 1].extend(chunks[-1])
        chunks = chunks[:num_processes]

    return chunks

def starlink_satellite_from_map(folder_path):
    log_file_sat = os.path.join(folder_path, 'sat_measurements.log')
    figure_path = f'{folder_path}/figures'
    figure_path_2d = f'{folder_path}/figures_2d'
    if os.path.exists(figure_path):
        shutil.rmtree(figure_path)
    os.mkdir(figure_path)
    os.mkdir(figure_path_2d)

    num_processes = 1
    satellite_interval = 15

    chicago_timezone = get_localzone()
    print('Start Collecting')
    default_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    response = requests.get(default_url)
    with open('starlink.txt', 'w') as file:
        file.write(response.text)
    tle_file = 'starlink.txt'
    satellites = load.tle_file(tle_file)
    # satellite_chunks = divide_work(satellites, num_processes)

    context = starlink_grpc.ChannelContext()
    location = starlink_grpc.get_location(context)
    my_location = Topos(location.lla.lat, location.lla.lon, elevation_m=location.lla.alt)

    # For TUM
    # my_location = Topos(48.15088866379816, 11.56873767083095, elevation_m=525)

    print(f"[Location]: {my_location.latitude.degrees}°, {my_location.longitude.degrees}")

    starlink_grpc.reset_obstruction_map(context)

    ts = load.timescale()
    snr_data_array = []
    timeline = []

    # Loop until we get a result
    all_results_save = {'Candidate_Sat': [], 'Connected_Sat': [], 'Timestamp': [], 'Candidate_Trace': []}
    id_pic = 0
    while True:
        try:
            # Get current time
            current_time = time.localtime()

            # Check if it's exactly on the half hour or hour
            if current_time.tm_sec % satellite_interval == 12:
                t = ts.now()
                try:
                    snr_data = get_snr_data(current_time)
                    context = starlink_grpc.ChannelContext()
                    starlink_grpc.reset_obstruction_map(context)
                except:
                    time.sleep(2)
                    continue
                timeline.append(t)
                # snr_data_array.append(snr_data)


                print(t.utc_datetime().replace(tzinfo=pytz.utc).astimezone(chicago_timezone))

                if len(timeline) >= 2:
                    # get the measure_trace between two snr data

                    # np.save('./snr_data.npy', np.array(snr_data))
                    try:
                        az_off = starlink_grpc.get_status().boresight_azimuth_deg
                        ele_off = starlink_grpc.get_status().boresight_elevation_deg
                    except:
                        time.sleep(2)
                        continue
                    measure_trace = diff_obstruction_2d_to_3d(0-np.ones_like(snr_data), snr_data,
                                                     azimuth_offset=az_off,
                                                     elevation_offset=ele_off,
                                                              save_path=folder_path)
                    if not measure_trace:
                        time.sleep(1)
                        continue

                    times = ts.linspace(timeline[-2], timeline[-1], num=satellite_interval)

                    all_results = process_satellites(satellites, my_location, times, measure_trace)

                    min_distance = 100000
                    best_sat_name = None
                    all_trace = []
                    all_sat = []
                    best_sat_trace = []
                    all_measure_trace = []
                    with open(log_file_sat, 'a') as file:
                        for i_c, result in enumerate(all_results):
                            if result is not None:
                                # print(result)
                                sat_name, sat_trace, distance, measure_trace = result
                                file.write(f"[Candidate {i_c}] {sat_name}, {sat_trace[0][0]:.5f}, {sat_trace[0][1]:.5f}, {sat_trace[0][2]:.5f}, {sat_trace[0][3]:.5f}\n")
                                all_trace.append(sat_trace)
                                all_sat.append(sat_name)
                                all_measure_trace.append(measure_trace)
                                if abs(distance) < min_distance:
                                    min_distance = abs(distance)
                                    best_sat_name = sat_name
                                    best_sat_trace = sat_trace
                        # file.write('\n')
                        for i in range(satellite_interval):
                            file.write(f"[Connected] {times[i].utc_datetime().timestamp()}, {best_sat_name}, "
                                       f"{best_sat_trace[i][0]:.5f}, {best_sat_trace[i][1]:.5f}, {best_sat_trace[i][2]:.5f},"
                                       f" {min_distance}, {ele_off:.5f}, {az_off:.5f}, {best_sat_trace[i][3]:.5f}\n")
                            file.flush()
                        # file.write('\n')
                    print("from " + str(timeline[-1].utc_datetime().replace(tzinfo=pytz.utc).astimezone(chicago_timezone)) + " to " + str(timeline[-2].utc_datetime().replace(tzinfo=pytz.utc).astimezone(chicago_timezone)))
                    print("best match satellite is: " + best_sat_name)
                    print(f"[Finish Time]: {ts.now().utc_datetime().replace(tzinfo=pytz.utc).astimezone(chicago_timezone)}")
                    print("--------------------\n")
                    # all_results_save['Candidate_Sat'].append(all_sat)
                    # all_results_save['Connected_Sat'].append(best_sat_name)
                    # all_results_save['Candidate_Trace'].append(all_trace)
                    # all_results_save['Timestamp'].append(times)
                    # np.save(f'{folder_path}/sat_stats.npy', all_results_save)

                    # Set up the plot
                    # fig = plt.figure()
                    if id_pic % 50 == 0:
                        # plt.figure()
                        # plt.imshow(np.array(snr_data), cmap='gray')  # Display the image with a grayscale colormap
                        # plt.grid()
                        # plt.tight_layout()
                        # plt.savefig(
                        #     f'{figure_path_2d}/{id_pic}' + '.png')
                        # plt.close()

                        azimuth_rad = np.deg2rad(az_off)
                        fig, axs = plt.subplots(1, 3, subplot_kw={'polar': True}, figsize=(15, 6))
                        axs = np.array([axs])
                        ax0 = axs[0, 0]

                        # ax = fig.add_subplot(111, polar=True)
                        ax0.set_theta_zero_location("N")
                        ax0.set_theta_direction(-1)
                        ax0.set_rlim(90, 0)  # limit the radius to go from 90 to 0
                        ax0.grid(True)

                        ax1 = axs[0, 1]

                        # ax = fig.add_subplot(111, polar=True)
                        ax1.set_theta_zero_location("N")
                        ax1.set_theta_direction(-1)
                        ax1.set_rlim(90, 0)  # limit the radius to go from 90 to 0
                        ax1.grid(True)

                        ax2 = axs[0, 2]
                        # ax = fig.add_subplot(111, polar=True)
                        ax2.set_theta_zero_location("N")
                        ax2.set_theta_direction(-1)
                        ax2.set_rlim(90, 0)  # limit the radius to go from 90 to 0
                        ax2.grid(True)

                        # Plot the measure_trace
                        measure_trace = np.array(measure_trace)
                        ax0.scatter(measure_trace[:, 1], measure_trace[:, 0], label="measure_trace")

                        # Plot the best satellite trace
                        best_sat_trace = np.array(best_sat_trace)
                        ax1.scatter(best_sat_trace[:, 1], best_sat_trace[:, 0],
                                    label=best_sat_name)

                        for trace, sat in zip(all_trace, all_sat):
                            ax2.scatter(trace[:, 1], trace[:, 0], label=sat)

                        # Plot the path
                        ax0.set_title('SNR Difference')
                        ax1.set_title('match Satellite Trace')
                        ax2.set_title('Starlink Satellite Trace')
                        ax0.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
                        ax1.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
                        ax2.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
                        # plt.show()# Save the figure
                        plt.savefig(
                            f'{figure_path}/{id_pic}_' + str(timeline[-1].utc_datetime().replace(tzinfo=pytz.utc).astimezone(chicago_timezone)) + '.png')
                        plt.close(fig)

                        np.save(f'{figure_path}/{id_pic}.npy',
                                np.array({'snr_data': np.array(snr_data), 'best_sat_name': best_sat_name,
                                          'best_sat_trace': best_sat_trace, 'all_trace': all_trace, 'all_sat': all_sat,
                                          'measure_trace': measure_trace, 'az_off': az_off, 'ele_off': ele_off}))
                    id_pic += 1
                if len(timeline) >= 60:
                    default_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
                    try:
                        response = requests.get(default_url)
                        if response.status_code == 200:
                            with open('starlink.txt', 'w') as file:
                                file.write(response.text)
                        else:
                            print(f"Failed to download data, status code: {response.status_code}")
                    except:
                        pass
                    timeline = timeline[-5:]
            # Sleep for 1 second to prevent constant checking
            time.sleep(1)
        except:
            time.sleep(60)
            print("Try Again")
            continue

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    start_time = datetime.now().strftime('%Y-%m-%d_%H_%M')
    tag = f'data_{start_time}'
    folder_path = f'/artifacts/{tag}'

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path+'/figures_3d')


    # start throughput and latency measurement
    iperf_process = Process(target=run_iperf3, args=(folder_path,))
    ping_process = Process(target=ping_continuously, args=(folder_path,))
    iperf_process.start()
    ping_process.start()
    starlink_satellite_from_map(folder_path)