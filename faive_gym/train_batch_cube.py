import subprocess

# the command that will be kept the same for all runs
max_iterations = 500
no_envs = 8000
wandb_group = "hand_b_cube_initial_runs"
base_command = f"python train.py task=hand_BP0_cube max_iterations={max_iterations} num_envs={no_envs} wandb_group={wandb_group} wandb_activate=True capture_video=True force_render=False test=False"

# The parameters that we want to test (try 2 different random seeds for each parameter):
action_panelties = [0.] # make sure these are floats
success_rewards = [3., 5.]
drop_penalities = [-5.] # [-5,-1] # make sure these are floats
reorienttask_obj_dist_rewards = [-0.05]
reorienttask_obj_rot_rewards = [0.03, 0.02, 0.01]
  
# relative control 
relative_control = [True, False]

# iterate through seeds
seeds = [42, 31]

for relative_ctrl in relative_control:
    for seed in seeds:
        for action_panelty in action_panelties:
            for drop_penalty in drop_penalities:
                for reorienttask_obj_dist in reorienttask_obj_dist_rewards:
                    for reorienttask_obj_rot in reorienttask_obj_rot_rewards:
                        for success in success_rewards:
                            wandb_name = f"new_tendon_ratio_success_{success}_reorienttask_obj_dist_{reorienttask_obj_dist}_reorienttask_obj_rot_{reorienttask_obj_rot}_drop_{drop_penalty}_seed_{seed}_relative_{relative_ctrl}_action_{action_panelty}"

                            command = f"{base_command} task.rewards.scales.success={success} task.rewards.scales.action_penalty={action_panelty} task.rewards.scales.reorienttask_obj_dist={reorienttask_obj_dist} task.rewards.scales.reorienttask_obj_rot={reorienttask_obj_rot} task.env.use_relative_control={relative_ctrl} task.rewards.scales.drop_penalty={drop_penalty} seed={seed} wandb_name={wandb_name}"
                            print("------COMMAND-----")
                            print(command)
                            print("------------------")
                            
                            subprocess.run(command, shell=True)
                            # save the weights so that it can be tested later
                            subprocess.run(f"cp runs/Hand_B/nn/Hand_B.pth runs/Hand_B/nn/saves/{wandb_name}.pth", shell=True)
                            
                            # exit()