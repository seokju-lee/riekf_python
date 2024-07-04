import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from liegroup import LieGroup
from noiseparams import NoiseParams
from robotstate import RobotState
from inekf import InEKF, Observation

DT_MIN = 1e-6
DT_MAX = 1

class Kinematics:
    def __init__(self, id, pose, covariance):
        self.id = id
        self.pose = pose
        self.covariance = covariance

def stod98(s):
    return float(s)

def stoi98(s):
    return int(float(s))

def main():
    #  ---- Initialize invariant extended Kalman filter ----- #
    device = 'cpu'  # Use 'cuda' for GPU acceleration
    initial_state = RobotState(device=device)

    # Initialize state mean
    R0 = torch.tensor([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1]], dtype=torch.float64, device=device)
    v0 = torch.zeros(3, dtype=torch.float64, device=device)
    p0 = torch.zeros(3, dtype=torch.float64, device=device)
    bg0 = torch.zeros((3, 1), dtype=torch.float64, device=device)  # Reshape to (3, 1)
    ba0 = torch.zeros((3, 1), dtype=torch.float64, device=device)  # Reshape to (3, 1)
    initial_state.setRotation(R0)
    initial_state.setVelocity(v0)
    initial_state.setPosition(p0)
    initial_state.setGyroscopeBias(bg0)
    initial_state.setAccelerometerBias(ba0)

    # Initialize state covariance
    noise_params = NoiseParams(device=device)
    noise_params.setGyroscopeNoise(0.01)
    noise_params.setAccelerometerNoise(0.1)
    noise_params.setGyroscopeBiasNoise(0.00001)
    noise_params.setAccelerometerBiasNoise(0.0001)
    noise_params.setContactNoise(0.01)

    # Initialize filter
    filter = InEKF(state=initial_state, params=noise_params, device=device)
    print("Noise parameters are initialized to:")
    print(filter.getNoiseParams())
    print("Robot's state is initialized to:")
    print(filter.getState())

    # Open data file
    filepath = os.path.abspath('data/imu_kinematic_measurements.txt')
    with open(filepath, 'r') as infile:
        imu_measurement = torch.zeros(6, dtype=torch.float64, device=device)
        imu_measurement_prev = torch.zeros(6, dtype=torch.float64, device=device)
        t = 0
        t_prev = 0

        # ---- Loop through data file and read in measurements line by line ---- #
        for line in infile:
            measurement = line.split()
            if measurement[0] == "IMU":
                print("Received IMU Data, propagating state")
                assert (len(measurement) - 2) == 6
                t = stod98(measurement[1])
                imu_measurement = torch.tensor([stod98(measurement[2]),
                                                stod98(measurement[3]),
                                                stod98(measurement[4]),
                                                stod98(measurement[5]),
                                                stod98(measurement[6]),
                                                stod98(measurement[7])], dtype=torch.float64, device=device)

                # Propagate using IMU data
                dt = t - t_prev
                if DT_MIN < dt < DT_MAX:
                    filter.Propagate(imu_measurement_prev, dt)

            elif measurement[0] == "CONTACT":
                print("Received CONTACT Data, setting filter's contact state")
                assert (len(measurement) - 2) % 2 == 0
                contacts = []
                t = stod98(measurement[1])
                for i in range(2, len(measurement), 2):
                    id = stoi98(measurement[i])
                    indicator = bool(stod98(measurement[i+1]))
                    contacts.append((id, indicator))
                filter.setContacts(contacts)

            elif measurement[0] == "KINEMATIC":
                print("Received KINEMATIC observation, correcting state")
                assert (len(measurement) - 2) % 44 == 0
                t = stod98(measurement[1])
                measured_kinematics = []
                for i in range(2, len(measurement), 44):
                    id = stoi98(measurement[i])
                    q = [stod98(measurement[i+1]),
                         stod98(measurement[i+2]),
                         stod98(measurement[i+3]),
                         stod98(measurement[i+4])]
                    r = R.from_quat(q)
                    rotation_matrix = torch.tensor(r.as_matrix(), dtype=torch.float64, device=device)
                    p = torch.tensor([stod98(measurement[i+5]),
                                      stod98(measurement[i+6]),
                                      stod98(measurement[i+7])], dtype=torch.float64, device=device)
                    pose = torch.eye(4, dtype=torch.float64, device=device)
                    pose[:3, :3] = rotation_matrix
                    pose[:3, 3] = p
                    covariance = torch.zeros((6, 6), dtype=torch.float64, device=device)
                    for j in range(6):
                        for k in range(6):
                            covariance[j, k] = stod98(measurement[i+8 + j*6 + k])
                    frame = Kinematics(id, pose, covariance)
                    measured_kinematics.append(frame)
                filter.CorrectKinematics(measured_kinematics)

            t_prev = t
            imu_measurement_prev = imu_measurement

    print(filter.getState())

if __name__ == "__main__":
    main()
