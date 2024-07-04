# Right-Invariant Extended Kalman Filter

This repository contains an implementation of a right-invariant extended Kalman filter [1] in Python.

## How to Run

To execute the code, simply run the following command:

```bash
python3 test.py
```
## Data File

The `data/imu_kinematic_measurements.txt` file used in this project was sourced from the following repository:

[RossHartley/invariant-ekf](https://github.com/RossHartley/invariant-ekf/tree/master)

This data file contains IMU/Contact/Kimenatics value. Each dataset begins with a time value. For the IMU (Inertial Measurement Unit), the data includes angular velocity and linear acceleration. The Contact data contains an ID as the first value and a boolean indicating contact status as the second value. Since this pertains to Cassie, which has two legs, there are four data values in total for Contact.

For Kinematics, the data structure is as follows: the first value is the ID, followed by four values representing a quaternion, three values for position, and 36 values for the 6x6 covariance data. Since this is for a bipedal robot, there are two legs, resulting in a total of 88 values for the Kinematics data.

Please ensure that you have this file in the `data` directory before running the code.

## Citation
[1] Hartley, R., Ghaffari, M., Eustice, R. M., & Grizzle, J. W. (2020). Contact-aided invariant extended Kalman filtering for robot state estimation. The International Journal of Robotics Research, 39(4), 402-430.

