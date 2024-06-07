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
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def update_step(self):
        # Update step with JPDA
        Inn = self.Z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    el = math.atan(z / np.sqrt(x ** 2 + y ** 2)) * 180 / 3.14
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

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2))
        el.append(math.atan(z[i] / np.sqrt(x[i] ** 2 + y[i] ** 2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))

        if x[i] > 0.0:
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]

        az[i] = az[i] * 180 / 3.14

        if az[i] < 0.0:
            az[i] = (360 + az[i])

        if az[i] > 360:
            az[i] = (az[i] - 360)

    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            print("Cartesian coordinates (x, y, z):", x, y, z)
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            print("Spherical coordinates (r, az, el):", r, az, el)
            measurements.append((r, az, el, mt))
    return measurements

# Function to initialize tracks based on the ±1 km condition
def initialize_tracks(measurements):
    tracks = defaultdict(list)
    track_id = 0

    for measurement in measurements:
        r, az, el, mt = measurement
        assigned = False
        for tid, track in tracks.items():
            if abs(track[0][0] - r) <= 1:  # Check the range condition
                tracks[tid].append(measurement)
                assigned = True
                break
        if not assigned:
            track_id += 1
            tracks[track_id].append(measurement)

    return tracks

# Function to find the most likely measurement for each track using a simple distance-based probability
def associate_measurements(kalman_filter, track):
    predicted_position = (kalman_filter.Sf[0][0], kalman_filter.Sf[1][0], kalman_filter.Sf[2][0])
    closest_measurement = min(track, key=lambda m: np.linalg.norm(np.array(predicted_position) - np.array(m[:3])))
    return closest_measurement

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_84.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Initialize tracks
tracks = initialize_tracks(measurements)

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Iterate through tracks
for track_id, track in tracks.items():
    for i, measurement in enumerate(track):
        r, az, el, mt = measurement
        if i == 0:
            # Initialize filter state with the first measurement
            kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
        elif i == 1:
            # Initialize filter state with the second measurement and compute velocity
            prev_r, prev_az, prev_el = track[i - 1][:3]
            dt = mt - track[i - 1][3]
            vx = (r - prev_r) / dt
            vy = (az - prev_az) / dt
            vz = (el - prev_el) / dt
            kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
        else:
            kalman_filter.predict_step(mt)

            # Perform JPDA for associating measurements
            most_likely_measurement = associate_measurements(kalman_filter, track)
            kalman_filter.InitializeMeasurementForFiltering(*most_likely_measurement[:3], 0, 0, 0, mt)

            # Once you've identified the most likely measurement, perform the update step
            kalman_filter.update_step()

            # Append data for plotting
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
