import torch
import glob
import os
from stable_baselines3 import PPO


def checkpoint_load(checkpoint_path):
    checkpoints = glob.glob(f"{checkpoint_path.rstrip('/')}/h*_ofo_chkpt/*/*.zip")
    if len(checkpoints) > 0:
        print(f"Checkpoints found: {checkpoints}")
        # get the newest checkpoint
        newest_checkpoint = max(checkpoints, key=os.path.getctime)
        top_ten = sorted(checkpoints, key=os.path.getctime, reverse=True)[:10]
        print(f"Newest checkpoint: {newest_checkpoint}")
        # torch.load(newest_checkpoint, weights_only=True)
        model = PPO.load(newest_checkpoint)
        # run_model = PPO.load(newest_checkpoint, env=env,)
        # tensorboard_log=tensorboard_log)
        print("\ncheckpoint loaded")
        print(model.get_parameters()["policy"])
    else:
        print("No checkpoints found.")


if __name__ == "__main__":
    checkpoint_path = "/Volumes/Speed/"

    checkpoint_load(checkpoint_path)
