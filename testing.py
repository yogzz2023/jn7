import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

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
        """Initialize filter state."""
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def initialize_measurement_for_filtering(self, x, y, z, mt):
        """Initialize measurement for filtering."""
        self.Z = np.array([[x], [y], [z]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        """Predict step of the Kalman filter."""
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self, report):
        """Update step of the Kalman filter using the associated report."""
        Inn = report - np.dot(self.H, self.Sf)  # Calculate innovation using associated report
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    el = math.atan(z / np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
    az = math.atan(y / x)

    if x > 0.0:
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x:float,y:float,z:float,filtered_values_csv):
    r=[]
    az=[]
    el=[]
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i]/np.sqrt(x[i]**2 + y[i]**2))*180/3.14)
        az.append(math.atan(y[i]/x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14/2 - az[i]
        else:
            az[i] = 3*3.14/2 - az[i]       
        
        az[i]=az[i]*180/3.14 

        if(az[i]<0.0):
            az[i]=(360 + az[i])
    
        if(az[i]>360):
            az[i]=(az[i] - 360)   
  
    return r,az,el

def read_measurements_from_csv(file_path):
    """Read measurements from CSV file."""
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt, mr))  # Include range for track initiation
    return measurements

def group_measurements_into_tracks(measurements):
    """Group measurements into tracks."""
    tracks = []
    for measurement in measurements:
        x, y, z, mt, r = measurement
        assigned = False
        for track in tracks:
            if abs(track[0][4] - r) <= 1:  # Check if within ±1 km range
                track.append(measurement)
                assigned = True
                break
        if not assigned:
            tracks.append([measurement])
    return tracks

def is_valid_hypothesis(hypothesis):
    """Check if a hypothesis is valid."""
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    """Calculate Mahalanobis distance."""
    delta = y[:3] - x[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def perform_clustering_hypothesis_association(tracks, reports, cov_inv):
    """Perform clustering, hypothesis generation, and association."""
    clusters = []
    for report in reports:
        distances = [np.linalg.norm(track[0][:3] - report[:3]) for track in tracks]
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < chi2_threshold:
            clusters.append([min_distance_idx])
    print("Clusters:", clusters)

    hypotheses = []
    for cluster in clusters:
        num_tracks = len(cluster)
        base = len(reports) + 1
        for count in range(base ** num_tracks):
            hypothesis = []
            for track_idx in cluster:
                report_idx = (count // (base ** track_idx)) % base
                hypothesis.append((track_idx, report_idx - 1))
            if is_valid_hypothesis(hypothesis):
                hypotheses.append(hypothesis)

    probabilities = calculate_probabilities(hypotheses, tracks, reports, cov_inv)
    return hypotheses, probabilities

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    """Calculate probabilities for each hypothesis."""
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_idx][0], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance ** 2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

def find_max_associations(hypotheses, probabilities, reports):
    """Find the most likely association for each report."""
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations, max_probs

def main():
    """Main function."""
    file_path = 'ttk_84.csv'
    measurements = read_measurements_from_csv(file_path)
    tracks = group_measurements_into_tracks(measurements)

    # Initialize filters for each track
    filters = [CVFilter() for _ in range(len(tracks))]
    for i, track in enumerate(tracks):
        x, y, z, mt, _ = track[0]
        filters[i].initialize_filter_state(x, y, z, 0, 0, 0, mt)

    filtered_values_csv = pd.read_csv(file_path)

    results = {i: {'x': [], 'y': [], 'z': [], 'time': []} for i in range(len(tracks))}

    for time_step in sorted(list(set([mt for _, _, _, mt, _ in measurements]))):
        reports = [measurement for measurement in measurements if measurement[3] == time_step]

        cov_inv = np.linalg.inv(filters[0].Pf[:3, :3])  # Covariance matrix inverse
        hypotheses, probabilities = perform_clustering_hypothesis_association(tracks, reports, cov_inv)

        max_associations, _ = find_max_associations(hypotheses, probabilities, reports)
        for report_idx, track_idx in enumerate(max_associations):
            if track_idx != -1:
                report = reports[report_idx][:3]
                filters[track_idx].predict_step(time_step)
                filters[track_idx].update_step(report)
                results[track_idx]['x'].append(filters[track_idx].Sf[0, 0])
                results[track_idx]['y'].append(filters[track_idx].Sf[1, 0])
                results[track_idx]['z'].append(filters[track_idx].Sf[2, 0])
                results[track_idx]['time'].append(time_step)

    for track_idx, data in results.items():
        r, az, el = cart2sph2(data['x'], data['y'], data['z'], filtered_values_csv)
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(data['time'], r, label='Filtered Range')
        plt.xlabel('Time')
        plt.ylabel('Range (km)')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(data['time'], az, label='Filtered Azimuth')
        plt.xlabel('Time')
        plt.ylabel('Azimuth (degrees)')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(data['time'], el, label='Filtered Elevation')
        plt.xlabel('Time')
        plt.ylabel('Elevation (degrees)')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
