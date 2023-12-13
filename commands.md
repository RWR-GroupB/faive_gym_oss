## Record policy
python3 train.py task=hand_BP0_sphere headless=True test=True num_envs=10 task.logging.record_dofs=True task.logging.record_observations=True task.logging.record_length=500 checkpoint=runs/Hand_B/nn/saves/new_tendon_ratio_rolling_direction_-1_x_rotvel_0.1_drop_-5.0_hand_angle_25_seed_42_relative_True_action_0.0.pth

## train batch
python3 train_batch.py