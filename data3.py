import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from collections import defaultdict

r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6,1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3,1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self):
        Inn = self.Z[:3] - np.dot(self.H, self.Sf[:3])  # Innovation
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / 3.14
    az = math.atan(y / x)
    if x > 0.0:
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az
    az = az * 180 / 3.14
    if az < 0.0:
        az = (360 + az)
    if az > 360:
        az = (az - 360)
    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])
            ma = float(row[8])
            me = float(row[9])
            mt = float(row[10])
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((x, y, z, mt))
    return measurements

def compute_mahalanobis_distance(predicted_state, measurement, covariance):
    innovation = measurement - predicted_state
    return np.sqrt(np.dot(np.dot(innovation.T, np.linalg.inv(covariance)), innovation))

def compute_gaussian_probability(distance, dim=3):
    return (1 / (np.sqrt((2 * np.pi)**dim))) * np.exp(-0.5 * distance**2)

def cluster_measurements(measurements, distance_threshold=1000):
    clusters = []
    current_cluster = [measurements[0]]

    for measurement in measurements[1:]:
        distance = np.linalg.norm(np.array(measurement[:3]) - np.array(current_cluster[-1][:3]))
        if distance <= distance_threshold:  # Within ±1 km
            current_cluster.append(measurement)
        else:
            clusters.append(current_cluster)
            current_cluster = [measurement]
    clusters.append(current_cluster)
    return clusters

def generate_hypotheses(clusters):
    hypotheses = []
    for cluster in clusters:
        hypotheses.append(cluster)
    return hypotheses

def compute_joint_probabilities(hypotheses, filter):
    joint_probabilities = []
    for hypothesis in hypotheses:
        hypothesis_probabilities = []
        for measurement in hypothesis:
            measurement_vector = np.array(measurement[:3]).reshape(-1, 1)
            distance = compute_mahalanobis_distance(filter.Sf[:3], measurement_vector, filter.Pf[:3, :3])
            probability = compute_gaussian_probability(distance)
            hypothesis_probabilities.append(probability)
        joint_probabilities.append(np.prod(hypothesis_probabilities))
    return joint_probabilities

def associate_measurements(filter, hypotheses):
    joint_probabilities = compute_joint_probabilities(hypotheses, filter)
    most_likely_hypothesis_index = np.argmax(joint_probabilities)
    return hypotheses[most_likely_hypothesis_index]

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_84.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Cluster measurements
clusters = cluster_measurements(measurements)

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Iterate through clusters
for cluster in clusters:
    hypotheses = generate_hypotheses([cluster])
    for i, (x, y, z, mt) in enumerate(cluster):
        r, az, el = cart2sph(x, y, z)
        if i == 0:
            kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
        elif i == 1:
            prev_r, prev_az, prev_el = cluster[i - 1][:3]
            dt = mt - cluster[i - 1][3]
            vx = (r - prev_r) / dt
            vy = (az - prev_az) / dt
            vz = (el - prev_el) / dt
            kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
        else:
            kalman_filter.predict_step(mt)
            most_likely_measurement = associate_measurements(kalman_filter, hypotheses)
            kalman_filter.InitializeMeasurementForFiltering(*most_likely_measurement[:3], 0, 0, 0, mt)
            kalman_filter.update_step()
            time_list.append(mt)
            r_list.append(r)
            az_list.append(az)
            el_list.append(el)

# Plotting results
def plot_results(time_list, data_list, label, ylabel, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(time_list, data_list, label=label, color='green', marker='*')
    plt.xlabel('Time', color='black')
    plt.ylabel(ylabel, color='black')
    plt.title(title, color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

plot_results(time_list, r_list, 'filtered range (code)', 'Range (r)', 'Range vs. Time')
plot_results(time_list, az_list, 'filtered azimuth (code)', 'Azimuth (az)', 'Azimuth vs. Time')
plot_results(time_list, el_list, 'filtered elevation (code)', 'Elevation (el)', 'Elevation vs. Time')
