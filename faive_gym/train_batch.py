import subprocess

# the command that will be kept the same for all runs
base_command = "python train.py max_iterations=1000 num_envs=8000 wandb_activate=True capture_video=True force_render=False"

# The parameters that we want to test (try 2 different random seeds for each parameter):
action_panelties = [0]
x_rotvel_rewards = [0.001,0.005,0.01,0.05,0.1]
drop_penalities = [-1,-5]

# relative control 
relative_control = [False, True]

# iterate through seeds
seeds = [42]

for action_panelty in action_panelties:
    for x_rotvel_reward in x_rotvel_rewards:
        for relative_ctrl in relative_control:
            for seed in seeds:
                for drop_penalty in drop_penalities:
                    wandb_name = f"action_{action_panelty}_x_rotvel_{x_rotvel_reward}_relative_{relative_ctrl}_drop_{drop_penalty}_seed_{seed}"

                    command = f"{base_command} task.rewards.scales.action_penalty={action_panelty} task.rewards.scales.rottask_obj_xrotvel={x_rotvel_reward} task.env.use_relative_control={relative_ctrl} task.rewards.scales.drop_penalty={drop_penalty} seed={seed} wandb_name={wandb_name}"
                    print("------COMMAND-----")
                    print(command)
                    print("------------------")
                    
                    subprocess.run(command, shell=True)
                    # save the weights so that it can be tested later
                    subprocess.run(f"cp runs/Hand_B/nn/Hand_B.pth runs/Hand_B/nn/saves/{wandb_name}.pth", shell=True)