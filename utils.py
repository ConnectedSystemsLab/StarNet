from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
import numpy as np
import starlink_grpc
import time
import os
import subprocess
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
password = 'power4ever'

def run_command(command):
    full_command = f"echo {password} | sudo -S {command}"
    print(full_command)
    subprocess.run(full_command, shell=True)



def get_snr_data(timenow):
    context = starlink_grpc.ChannelContext()
    attempt = 0
    retry_limit = 5
    while attempt < retry_limit:
        try:
            snr_data = starlink_grpc.obstruction_map(context)
            return snr_data
        except:
            if attempt >= retry_limit - 1:  # Last attempt
                raise Exception(f"Failed to fetch SNR data after {retry_limit} retries")
            time.sleep(0.5)
            attempt += 1

def polar_distance(a, b):
    r1, theta1 = a[0], a[1]
    r2, theta2 = b[0], b[1]
    delta_theta = np.pi - abs(abs(theta1 - theta2) - np.pi)
    return abs(r1 - r2) + delta_theta

def polar_to_cartesian(theta, phi):
    """
    x = r cos
    """
    x = np.cos(np.radians(theta)) * np.sin(phi)
    y = np.cos(np.radians(theta)) * np.cos(phi)
    return np.array([x, y])

def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    # theta = math.atan2(x, y)

    if theta < 0:
        theta += 2 * math.pi
    return 90-r, theta

def cartesian_to_polar_3d(point):
    x, y, z = point
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def find_intersection_points(R, p, d):
    # Coefficients of the quadratic equation At^2 + Bt + C = 0
    A = np.dot(d, d)
    B = 2 * np.dot(p, d)
    C = np.dot(p, p) - R ** 2

    # Calculate the discriminant
    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        return "No real intersection points."
    elif discriminant == 0:
        # One intersection point (tangent)
        t = -B / (2 * A)
        intersection_point = p + t * d
        r, theta, phi = cartesian_to_polar_3d(intersection_point)
        return (r, theta, phi)
    else:
        # Two intersection points
        t1 = (-B + np.sqrt(discriminant)) / (2 * A)
        t2 = (-B - np.sqrt(discriminant)) / (2 * A)
        intersection_point_1 = p + t1 * np.array(d)
        intersection_point_2 = p + t2 * np.array(d)
        if t1 > 0:
            r, theta, phi = cartesian_to_polar_3d(intersection_point_1)
        else:
            r, theta, phi = cartesian_to_polar_3d(intersection_point_2)
        return (r, theta, phi)



def diff_obstruction(array1, array2, azimuth_offset=0, elevation_offset=0):
    difference = []
    for i in range(len(array1)):
        for j in range(len(array1[0])):
            if array1[i][j] != array2[i][j]:
                # transfer 2d array to polar coordinates
                y = 62 - i
                x = j - 62
                r, theta = cartesian_to_polar(y,x)
                # r = 90-r
                # r = r + elevation_offset - 90
                # r = 90-r
                theta += math.radians(azimuth_offset + 180)
                difference.append((r, theta))
    return difference

def get_coord_from_2d(phi_, theta_, x, y):
    """
    Given the 2d coordinate from the obstruction map of grpc, convert this into 3D given the panel boresight angles.
    """
    x_tmp = x
    y_tmp = y * math.cos(theta_)
    r2 = x**2 + y**2
    y_ = y_tmp*math.cos(phi_) - x_tmp*math.sin(phi_)
    x_ = y_tmp*math.sin(phi_) + x_tmp*math.cos(phi_)
    if y == 0:
        z_ = 0
    else:
        z_ = math.sqrt(r2 - x_*x_ - y_*y_) if y >0 else -1*math.sqrt(r2 - x_*x_ - y_*y_)
    return [-1*x_, y_, z_]

def plot_3d_debug(R, p, d, results_list, save_path):
    '''
    p: list
    '''
    # Hemisphere
    phi, theta = np.mgrid[0:2*np.pi:100j, 0:np.pi/2:50j]
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # Plane - simplified for Plotly example
    x_plane = np.linspace(-R, R, 100)
    y_plane = np.linspace(-R, R, 100)
    x_plane, y_plane = np.meshgrid(x_plane, y_plane)
    z_plane = (-d[0] * x_plane - d[1] * y_plane) / (d[2] if d[2] != 0 else 1e-6)

    p_x = [point[0] for point in p]
    p_y = [point[1] for point in p]
    p_z = [point[2] for point in p]
    vector_end = np.array([d[0], d[1], d[2]]) * R * 0.1  # Scale factor for visualization

    re_x = [R* np.sin(point[1]) * np.cos(point[2]) for point in results_list]
    re_y = [R* np.sin(point[1]) * np.sin(point[2]) for point in results_list]
    re_z = [R* np.cos(point[1]) for point in results_list]


    fig = go.Figure(data=[
        go.Surface(z=z, x=x, y=y, colorscale='Blues', opacity=0.5),
        go.Surface(z=z_plane, x=x_plane, y=y_plane, colorscale='Oranges', opacity=0.5),
        go.Scatter3d(x=p_x, y=p_y, z=p_z, mode='markers', marker=dict(size=5, color='green')),
        go.Scatter3d(x=re_x, y=re_y, z=re_z, mode='markers', marker=dict(size=5, color='red')),

        go.Scatter3d(x=[0, R], y=[0, 0], z=[0, 0], marker=dict(size=1), line=dict(color="black", width=2)),
        go.Scatter3d(x=[0, 0], y=[0, R], z=[0, 0], marker=dict(size=1), line=dict(color="black", width=2)),
        go.Scatter3d(x=[0, vector_end[0]], y=[0, vector_end[1]], z=[0, vector_end[2]],
                     mode='lines+markers', line=dict(color="red", width=5),
                     marker=dict(size=7, color='red'), name='Vector d'),
    ])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        xaxis=dict(range=[-80, 80]),
        yaxis=dict(range=[-80, 80]),
        zaxis=dict(range=[-30, 80]),
        # aspectmode='manual',
        # aspectratio=dict(x=1, y=1, z=1),
    ))

    # fig.show()
    # start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # start_time = datetime.now().strftime('%Y-%m-%d %H')
    # filename = f"{save_path}/figures_3d/{start_time}.html"
    # if not os.path.exists(filename):
    #     fig.write_html(filename)

def diff_obstruction_2d_to_3d(array1, array2, azimuth_offset=0, elevation_offset=0, save_path='./'):
    # d = {'array1': array1, 'array2': array2, 'azimuth': azimuth_offset, 'elevation': elevation_offset}
    # np.save('debug.npy', d)
    # debug = True
    # if debug == True:
    #     d = np.load('debug.npy', allow_pickle=True).item()
    #     array1, array2, azimuth_offset, elevation_offset = d['array1'], d['array2'], d['azimuth'], d['elevation']

    # plt.figure()
    # plt.imshow(np.array(array2), cmap='gray')  # Display the image with a grayscale colormap
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    difference = []
    theta_ = math.pi/2 - elevation_offset*math.pi/180
    phi_ = (180*2 - azimuth_offset)*math.pi/180
    d = [math.sin(theta_) * math.sin(phi_), -math.sin(theta_) * math.cos(phi_), math.cos(theta_)]
    R = 52

    p_list = []
    results_list = []
    for i in range(len(array1)):
        for j in range(len(array1[0])):
            if array1[i][j] != array2[i][j]:
                y = 62 - i
                x = 62 - j
                x *= -1
                p = get_coord_from_2d(phi_, theta_, x, y)
                p_list.append(p)
                r = math.sqrt(x ** 2 + y ** 2)
                if r > R:
                    print(f'error, {r} > {R}')
                results = find_intersection_points(R, p, d)
                results_list.append(results)
                difference.append((90-results[1]*180/math.pi, math.pi*2 - (results[2]+math.pi/2)))
    plot_3d_debug(R, p_list, d, results_list, save_path)
    return difference


def calculate_distance_to_satellite(observer_location, sat, ts):
    """
    Calculates the distance from the observer to each satellite in the TLE list
    at the current time and logs the closest satellite and its distance.
    """
    t = ts.now()
    closest_satellite = None
    min_distance = float('inf')  # Initialize with a very high value

    difference = sat - observer_location
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    dis = distance.km
    return dis
    # if distance.km < min_distance:
    #     min_distance = distance.km
    #     closest_satellite = sat

    # if closest_satellite:
    #     print(f"Closest Satellite: {closest_satellite.name}, Distance: {min_distance:.2f} km")
    #     log_to_file(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Closest Satellite: {closest_satellite.name}, Distance: {min_distance:.2f} km")


if __name__ == "__main__":
    d = np.load('debug.npy', allow_pickle=True).item()
    array1, array2, azimuth_offset, elevation_offset = d['array1'], d['array2'], d['azimuth'], d['elevation']
    diff_obstruction_2d_to_3d(array1, array2, azimuth_offset, elevation_offset)