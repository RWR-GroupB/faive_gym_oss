import subprocess

# the command that will be kept the same for all runs
max_iterations = 1000
no_envs = 8000
wandb_group = "hand_b_sphere_optimisation_runs"
base_command = f"python train.py max_iterations={max_iterations} num_envs={no_envs} wandb_group={wandb_group} wandb_activate=True capture_video=True force_render=False"

# not yet implemented
hand_angles = {"20":[ 0.6963642, -0.1227878, 0.6963642, -0.1227878 ],"25":[ 0.6903455, -0.1530459, 0.6903455, -0.1530459 ]}
rolling_direction = [-1,1] #-1 is easier, 1 is harder

# The parameters that we want to test (try 2 different random seeds for each parameter):
action_panelties = [0.] # make sure these are floats
x_rotvel_rewards = [0.12,0.1] #,0.15,0.05]# [0.01,0.02,0.05,0.1,0.001,0.005] # make sure these are floats
drop_penalities = [-5.] # [-5,-1] # make sure these are floats

# relative control 
relative_control = [True, False]

# iterate through seeds
seeds = [42, 31, 5, 8]


for relative_ctrl in relative_control:
    for seed in seeds:
        for rolling_direction in rolling_direction:
            for action_panelty in action_panelties:
                for drop_penalty in drop_penalities:
                    for x_rotvel_reward in x_rotvel_rewards:
                        wandb_name = f"new_tendon_ratio_rolling_direction_{rolling_direction}_x_rotvel_{x_rotvel_reward}_drop_{drop_penalty}_hand_angle_25_seed_{seed}_relative_{relative_ctrl}_action_{action_panelty}"

                        command = f"{base_command} task.env.x_rotation_dir={rolling_direction} task.rewards.scales.action_penalty={action_panelty} task.rewards.scales.rottask_obj_xrotvel={x_rotvel_reward} task.env.use_relative_control={relative_ctrl} task.rewards.scales.drop_penalty={drop_penalty} seed={seed} wandb_name={wandb_name}"
                        print("------COMMAND-----")
                        print(command)
                        print("------------------")
                        
                        subprocess.run(command, shell=True)
                        # save the weights so that it can be tested later
                        subprocess.run(f"cp runs/Hand_B/nn/Hand_B.pth runs/Hand_B/nn/saves/{wandb_name}.pth", shell=True)
                        
                        # exit()