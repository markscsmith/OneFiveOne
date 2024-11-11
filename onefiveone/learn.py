import sys
import time
import os
import datetime
import hashlib
import multiprocessing
from emulator.pyboy_env import PyBoyEnv, PRESS_FRAMES, RELEASE_FRAMES
# Compute and AI libs
import numpy as np
import torch


import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import (
    BaseCallback,
    EveryNTimesteps,
    CheckpointCallback,
)

from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv

# Emulator libs
from pyboy import PyBoy

# Output libs
from timg import Renderer, Ansi24HblockMethod
from PIL import Image, ImageDraw, ImageFont
import glob

LOG_FREQ = 2048



# TODO: Convert to parameter to allow user to toggle between emulating on a CGB or DMG
# (CGB is the Gameboy Color, DMG is the original Gameboy)
CGB = False

# Set the default number of CPU cores to use for training
NUM_CPU = multiprocessing.cpu_count()


def learning_rate_schedule(progress):
    # return 0.025
    # progress starts at 1 and decreases as remaining approaches 0.
    rate = 0.0003
    variation = 0.2 * rate * progress
    # new_rate = rate + np.abs(variation * np.sin(progress * np.pi * 20)) # all positive
    new_rate = rate + variation * np.sin(
        progress * np.pi * 20
    )  # positive and negative adjustments
    # rate = (rate + rate * progress) / 2
    return new_rate
    # return  0.0


def learning_rate_decay_schedule(progress):
    return 0.0003 * (1 - progress)


class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # We set the frequency at which the callback will be called
        # This could be set to be called at each step by setting it to 1
        self.log_freq = LOG_FREQ
        self.buttons_names = "UDLRABS!-udlrabs.-"

    def _on_step(self) -> bool:
        # This method will be called by the model after each call to `env.step()`.
        # Note: self.n_calls is incremented after this method is called.
        if self.n_calls % self.log_freq == 0:
            # Log scalar value (here a random variable)
            rewards = self.locals["rewards"]
            infos = self.locals["infos"]

            for _, info in sorted(enumerate(infos)):
                # TODO: ADD POKEMON CAUGHT TO INFO
                if all(
                    key in info for key in ["actions", "emunum", "reward", "frames"]
                ):
                    actions = info["actions"]
                    emunum = info["emunum"]
                    reward = info["reward"]
                    frames = info["frames"]
                    caught = info["pokemon_caught"]
                    seen = info["pokemon_seen"]
                    pokedex = info["pokedex"]
                    badges = info["badges"]
                    seen_and_capture_events = info["seen_and_capture_events"]

                    # TODO: pad emunumber with 0s to match number of digits in possible emunum
                    self.logger.record(
                        f"actions/{emunum}",
                        f"{actions}:rew={reward}:fra={frames}:caught={caught}:seen={seen}",
                    )
                    self.logger.record(f"caught/{emunum}", f"{caught}")
                    self.logger.record(f"seen/{emunum}", f"{seen}")
                    self.logger.record(f"badges/{emunum}", f"{badges}")
                    self.logger.record(f"reward/{emunum}", f"{reward}")

                    self.logger.record(
                        f"visited/{emunum}", f"{len(info['visited_xy'])}"
                    )
                    self.logger.record(f"pokedex/{emunum}", f"{pokedex}")
                    self.logger.record(
                        f"seen_and_capture/{emunum}", f"{seen_and_capture_events}"
                    )

            # TODO: record each progress/reward separately like I do the actions?

            if len(rewards) > 0:  # Check if rewards list is not empty
                average_reward = sum(rewards) / len(rewards)

                badges = [info["badges"] for info in infos]

                max_reward = max(rewards)
                max_seen = max([info["pokemon_seen"] for info in infos])
                max_caught = max([info["pokemon_caught"] for info in infos])

                self.logger.record("reward/average_reward", average_reward)
                self.logger.record("reward/max_reward", max_reward)
                self.logger.record("reward/max_seen", max_seen)
                self.logger.record("reward/max_caught", max_caught)
                self.logger.record("reward/max_badges", max(badges))

        return True  # Returning True means we will continue training, returning False will stop training


class PokeCaughtCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        rewards = self.training_env.get_attr("total_reward")
        best_env_idx = rewards.index(max(rewards))
        print(self.training_env.env_method("render", best_env_idx)[best_env_idx])
        return True


def add_string_overlay(
    image, display_string, position=(20, 20), font_size=40, color=(255, 0, 0)
):
    """
    Add a number as an overlay on the image.

    Parameters:
    - image: PIL.Image.Image
        The image to which the number will be added
    - number: int or str
        The number to add
    - position: tuple of int, optional (default=(20, 20))
        The (x, y) position at which to add the number
    - font_size: int, optional (default=40)
        The font size of the number
    - color: tuple of int, optional (default=(255, 0, 0))
        The RGB color of the number

    Returns:
    PIL.Image.Image
        The image with the number added
    """
    # Initialize a drawing context
    draw = ImageDraw.Draw(image)

    try:
        # Use a truetype or opentype font file
        # font = ImageFont.truetype("arial.ttf", font_size)
        font = ImageFont.truetype(
            "arial.ttf",
            font_size,
        )
    except IOError:
        # If the font file is not available, the default PIL font is used
        font = ImageFont.load_default()

    # Draw the text
    # print(f"Drawing string {display_string} at {position} in color {color} on image of size {image.size}")
    draw.text(position, str(display_string), font=font, fill=color)

    return image





def make_env(game_path, emunum, max_frames=500_000, device="cpu", state_file=None):
    def _init():
        if state_file is not None and os.path.exists(state_file):
            print(f"Loading state {game_path}.state")
            if CGB:
                ext = ".state"
            else:
                ext = ".ogb_state"

            new_env = PyBoyEnv(
                game_path,
                emunum=emunum,
                save_state_path=game_path + ext,
                max_frames=max_frames,
                device=device,
            )
            new_env.pyboy.load_state(open(game_path + ext, "rb"))
        else:
            
            new_env = PyBoyEnv(
                game_path,
                emunum=emunum,
                max_frames=max_frames,
                device=device,
            )

        return new_env

    return _init


def train_model(
    env,
    total_steps,
    n_steps,
    batch_size,
    episodes,
    file_name,
    save_path="ofo",
    device="cpu",
):

    first_layer_size = 5608
    intermediate_layer_size = 1024
    action_layer_size = 8  # 8 actions
    output_layer_size = 1

    policy_kwargs = dict(
        net_arch=dict(
            pi=[first_layer_size, intermediate_layer_size, intermediate_layer_size, action_layer_size],
            vf=[first_layer_size, intermediate_layer_size, intermediate_layer_size, output_layer_size],
        ),
    )

    # make sure we take care of accidental trailing slashes in the save path which
    # would cause the checkpoint path to be incorrect.
    checkpoint_path = f"{save_path.rstrip('/')}_chkpt"
    env.set_attr("episode", 0)
    tensorboard_log = f"{save_path}/tensorboard/{os.uname()[1]}-{time.time()}"

    run_model = PPO(
        policy="MlpPolicy",
        # Reduce n_steps if too large; ensure not less than some minimum like 2048 for sufficient learning per update.
        n_steps=n_steps,
        # Reduce batch size if it's too large but ensure a minimum size for stability.
        batch_size=batch_size,
        n_epochs=1,
        gamma=0.998,
        gae_lambda=0.98,
        # learning_rate=learning_rate_schedule,
        # learning_rate=learning_rate_decay_schedule,
        ent_coef=0.07,
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
        tensorboard_log=tensorboard_log,
    )
    checkpoints = glob.glob(f"{checkpoint_path.rstrip('/')}/*/*.zip")
    if len(checkpoints) > 0:
        print(f"Checkpoints found: {checkpoints}")
        # get the newest checkpoint
        newest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Newest checkpoint: {newest_checkpoint}")
        run_model.load(newest_checkpoint)

        print("\ncheckpoint loaded")
    else:
        print("No checkpoints found.")

    update_freq = n_steps * num_cpu // 4

    for episode in range(1, episodes + 1):
        print(f"Starting episode {episode}")
        checkpoint_file_path = (
            f"{checkpoint_path.rstrip('/')}/{os.uname()[1]}-{time.time()}-{episode}/"
        )

        print(f"Checkpoint path: {checkpoint_file_path}")
        checkpoint_callback = CheckpointCallback(
            save_freq=total_steps // (NUM_CPU * 2),
            save_path=f"{checkpoint_file_path}",
            name_prefix="poke",
            verbose=2,
        )
        current_stats = EveryNTimesteps(
            n_steps=update_freq,
            callback=PokeCaughtCallback(),
        )
        tbcallback = TensorboardLoggingCallback(tensorboard_log)
        env.set_attr("episode", episode)
        callbacks = [checkpoint_callback, current_stats, tbcallback]
        run_model.learn(
            total_timesteps=total_steps, callback=callbacks, progress_bar=True
        )
        run_model.save(f"{checkpoint_path}/{file_name}-{episode}.zip")

        del callbacks
        del checkpoint_callback
        del current_stats
        del tbcallback

    return run_model


if __name__ == "__main__":
    device = "cpu"
    device = (
        "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else device
    )
    device = "cuda" if torch.cuda.is_available() else device

    import argparse

    parser = argparse.ArgumentParser()
    # TODO: Investigate Pokemon Blue caught = 4 before you have a pokedex. Is this a bug in the game?  Is it a bug in the emulator?  Is it a bug in the memory address? Seems to work fine on Red,
    # only happens during looking at starter pokemon in Oak's lab.

    # TODO: use more sensible defaults for non-me users
    # TODO: sensible defaults should be "current directory, any gb file I can find. If I find more than one, open the newest one. If I find none, error out."
    # TODO: Provide information (at least in logging) about each step of the run to provide better user clarity.
    # TODO: be "quiet" when parameters are passed and work as expected, but "chatty" when the parameter is skipped and the application is doing "defaulty" things.
    # TODO: DIRECTORY CLEANUP INCLUDING LOGROTATINON.

    # TODO: Expirement: If we can train on DIFFERENT pokemon carts, can we train on multiple GB games at a time and build a generally good base "gameboy game" model for training specific games?
    # TODO: Visual gif of map as it exapnds over time, with frames of the game as it is played, so the map is faded gray in the spot the AI isn't currently at.  Should be updated in frame order.  BIG PROJECT.
    
    parser.add_argument("--game_path", type=str, default="./POKEMONY.GBC")
    parser.add_argument("--state_file", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="ofo")
    parser.add_argument("--num_envs", type=int, default=NUM_CPU)
    args = parser.parse_args()

    num_cpu = args.num_envs

    run_env = None

    # TODO: Various hyperparameter tuning:
    # https://stackoverflow.com/questions/76076904/in-stable-baselines3-ppo-what-is-nsteps try using whole batch of n_steps as batch size?

    episodes = 16

    batch_size = 128

    n_steps = 2048

    # approximate hours of play according to in-game timer per env per episode
    # (certain actions stop the in-game timer)
    hours = 4


    # each step is (PRESS_FRAMES + RELEASE_FRAMES) frames long, at 60fps.  
    seconds = hours * 64 * 64
    total_steps = seconds * (60 // (PRESS_FRAMES + RELEASE_FRAMES)) * num_cpu
    

    if num_cpu == 1:
        run_env = DummyVecEnv([make_env(args.game_path, 0, device=device)])
    else:
        run_env = SubprocVecEnv(
            [
                make_env(args.game_path, emunum, device=device, state_file=args.state_file)
                for emunum in range(num_cpu)
            ]
        )

    model_file_name = "model"

    model = train_model(
        env=run_env,
        total_steps=total_steps,
        n_steps=n_steps,
        batch_size=batch_size,
        episodes=episodes,
        file_name=model_file_name,
        save_path=args.output_dir,
        device=device,
    )
