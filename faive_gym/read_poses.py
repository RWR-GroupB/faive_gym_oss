# Reads the joint poses from a .npy file and forwards them to to the 

import numpy as np

joint_poses = np.load('/home/ubuntu/faive_gym_oss/faive_gym/recordings/2023-12-13_10-28-14_dof_poses.npy')
print(joint_poses)
