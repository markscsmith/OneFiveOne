import sys
import time
import os
import datetime
import hashlib
import multiprocessing
from emulator.pyboy_env import PyBoyEnv, PRESS_FRAMES, RELEASE_FRAMES
from wrappers import (
    CountBasedExplorationWrapper,
    EpisodicStateNoveltyWrapper,
    pokemon_state_key_fn,
    pokemon_state_hash_fn,
)
# Compute and AI libs
import numpy as np
import torch


from stable_baselines3 import PPO
from stable_baselines3 import DQN
from sb3_contrib import QRDQN

from stable_baselines3.common.callbacks import (
    BaseCallback,
    EveryNTimesteps,
    CheckpointCallback,
)

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# Emulator libs
from pyboy import PyBoy

# Output libs
from timg import Renderer, Ansi24HblockMethod
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


# TODO: better understand why tb logging isn't recording all actions.
class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, n_steps=2048):
        super().__init__(verbose)
        # We set the frequency at which the callback will be called
        # This could be set to be called at each step by setting it to 1
        self.log_freq = n_steps // 4
        self.buttons_names = "UDLRABS!-udlrabs.-"

    def _on_step(self) -> bool:
        # This method will be called by the model after each call to `env.step()`.
        # Note: self.n_calls is incremented after this method is called.
        if self.n_calls % self.log_freq == 0:
            # Log scalar value (here a random variable)
            rewards = self.locals["rewards"]
            infos = self.locals["infos"]
            actions = self.locals["actions"]

            for _, info in sorted(enumerate(infos)):
                # TODO: ADD POKEMON CAUGHT TO INFO
                if all(
                    key in info for key in ["actions", "emunum", "total_reward", "frames"]
                ):
                    actions = info["actions"]
                    emunum = info["emunum"]
                    reward = info["total_reward"]
                    frames = info["frames"]
                    caught = info["pokemon_caught"]
                    seen = info["pokemon_seen"]
                    pokedex = info["pokedex"]
                    seen_and_capture_events = info["seen_and_capture_events"]

                    # TODO: pad emunumber with 0s to match number of digits in possible emunum
                    self.logger.record(
                        f"actions/{emunum}",
                        f"{actions}:rew={reward}:fra={frames}:caught={caught}:seen={seen}",
                    )
                    self.logger.record(f"caught/{emunum}", f"{caught}")
                    self.logger.record(f"seen/{emunum}", f"{seen}")
                    self.logger.record(f"reward/{emunum}", f"{reward}")

                    self.logger.record(
                        f"visited/{emunum}", f"{len(info['visited_xy'])}"
                    )
                    self.logger.record(f"pokedex/{emunum}", f"{pokedex}")
                    self.logger.record(
                        f"seen_and_capture/{emunum}", f"{seen_and_capture_events}"
                    )
            for emunum, actions in sorted(enumerate(actions)):
                self.logger.record(f"actions_debug/{emunum}", f"{actions}")

            # TODO: record each progress/reward separately like I do the actions?

            if len(rewards) > 0:  # Check if rewards list is not empty
                average_reward = sum(rewards) / len(rewards)


                max_reward = max(rewards)
                max_seen = max([info["pokemon_seen"] for info in infos])
                max_caught = max([info["pokemon_caught"] for info in infos])

                self.logger.record("reward/average_reward", average_reward)
                self.logger.record("reward/max_reward", max_reward)
                self.logger.record("reward/max_seen", max_seen)
                self.logger.record("reward/max_caught", max_caught)

        return True  # Returning True means we will continue training, returning False will stop training


class PokeCaughtCallback(BaseCallback):
    def __init__(self, render_interval=100):
        super().__init__()
        self.render_interval = render_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.render_interval == 0:
            rewards = self.training_env.get_attr("total_reward")
            best_env_idx = rewards.index(max(rewards))
            print(self.training_env.env_method("render", best_env_idx)[best_env_idx])
        return True


def make_env(
    game_path,
    emunum,
    num_steps,
    device="cpu",
    state_file=None,
    n_steps=2048,
    # Exploration wrapper options
    use_count_exploration=False,
    count_scale=1.0,
    count_exponent=0.5,
    count_decay=1.0,
    use_novelty_bonus=False,
    novelty_bonus=0.5,
    revisit_novelty_bonus=0.5,
):
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
                num_steps=num_steps,
                device=device,
                n_steps=n_steps,
            )
            new_env.pyboy.load_state(open(game_path + ext, "rb"))
        else:

            new_env = PyBoyEnv(
                game_path,
                emunum=emunum,
                num_steps=num_steps,
                device=device,
            )

        # Apply exploration wrappers (order: novelty first, count on top)
        if use_novelty_bonus:
            new_env = EpisodicStateNoveltyWrapper(
                new_env,
                state_hash_fn=pokemon_state_hash_fn,
                position_key_fn=pokemon_state_key_fn,
                novelty_bonus=novelty_bonus,
                revisit_novelty_bonus=revisit_novelty_bonus,
            )
            print(f"[env {emunum}] Episodic novelty wrapper enabled "
                  f"(bonus={novelty_bonus}, revisit={revisit_novelty_bonus})")

        if use_count_exploration:
            new_env = CountBasedExplorationWrapper(
                new_env,
                state_key_fn=pokemon_state_key_fn,
                scale=count_scale,
                exponent=count_exponent,
                decay_on_reset=count_decay,
            )
            print(f"[env {emunum}] Count-based exploration wrapper enabled "
                  f"(scale={count_scale}, exp={count_exponent}, decay={count_decay})")

        return new_env

    return _init


def create_model(env, total_steps, n_steps, batch_size, device, train_freq, hours, save_path, label=""):
    """
    Create a QRDQN model and load the newest checkpoint if one exists.

    Args:
        env: VecEnv to train on.
        total_steps: Total timesteps per episode.
        n_steps: Rollout length / target update interval.
        batch_size: Training batch size.
        device: Torch device string.
        train_freq: How often to run a training step.
        hours: Simulated gameplay hours (used for buffer size calc).
        save_path: Directory for checkpoints and tensorboard logs.
        label: Optional label for tensorboard (e.g. "exploration" or "control").

    Returns:
        (model, starting_episode, checkpoint_path)
    """
    first_layer_size = 1024
    intermediate_layer_size = 512

    policy_kwargs = dict(
        net_arch = [first_layer_size, first_layer_size, intermediate_layer_size, intermediate_layer_size, intermediate_layer_size, intermediate_layer_size, intermediate_layer_size],
        activation_fn=torch.nn.ReLU,
    )

    checkpoint_path = f"{save_path.rstrip('/')}"
    tb_label = f"-{label}" if label else ""
    tensorboard_log = f"{checkpoint_path}/tensorboard/{os.uname()[1]}-{time.time()}{tb_label}"

    run_model = QRDQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=int(total_steps / hours),
        learning_starts=0,
        tau=0.5,
        gamma=0.997,
        batch_size=batch_size,
        train_freq=train_freq,
        target_update_interval=n_steps,
        exploration_fraction=0.9,
        tensorboard_log=tensorboard_log,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    starting_episode = 1
    checkpoints = glob.glob(f"{checkpoint_path.rstrip('/')}/*.zip")
    if len(checkpoints) > 0:
        print(f"[{label or 'model'}] Checkpoints found: {checkpoints}")
        newest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"[{label or 'model'}] Loading: {newest_checkpoint}")
        starting_episode = int(newest_checkpoint.split("-")[-2]) + 1
        run_model.load(newest_checkpoint)
        print(f"[{label or 'model'}] Checkpoint loaded")
    else:
        print(f"[{label or 'model'}] No checkpoints found, starting fresh.")

    env.set_attr("episode", 0)
    return run_model, starting_episode, checkpoint_path


def train_episode(model, env, episode, total_steps, n_steps, num_envs, render_interval, checkpoint_path, label=""):
    """
    Train a single episode for a model/env pair.

    Args:
        model: The sb3 model to train.
        env: The VecEnv the model is attached to.
        episode: Current episode number.
        total_steps: Total timesteps for this episode.
        n_steps: Used to compute update frequency.
        num_envs: Number of parallel environments (for freq calculations).
        render_interval: How often PokeCaughtCallback renders.
        checkpoint_path: Where to save checkpoints and action logs.
        label: Label for log messages (e.g. "exploration", "control").
    """
    tag = f"[{label}] " if label else ""
    print(f"{tag}Starting episode {episode}")
    episode_path = (
        f"{checkpoint_path.rstrip('/')}/{os.uname()[1]}-{time.time()}-{episode}"
    )
    print(f"{tag}Checkpoint path: {episode_path}")

    checkpoint_callback = CheckpointCallback(
        save_freq=total_steps // (num_envs * 2),
        save_path=f"{episode_path}",
        name_prefix="poke",
        verbose=2,
    )
    update_freq = n_steps * num_envs // 8
    current_stats = EveryNTimesteps(
        n_steps=update_freq,
        callback=PokeCaughtCallback(render_interval),
    )

    env.set_attr("episode", episode)
    callbacks = [checkpoint_callback, current_stats]

    model.learn(
        total_timesteps=total_steps, callback=callbacks, progress_bar=True, log_interval=512
    )
    model.save(f"{episode_path}-model.zip")

    actions_set = env.get_attr("actions")
    global_actions_set = env.get_attr("global_actions")
    for emunum, global_actions in enumerate(global_actions_set):
        with open(f"{episode_path.rstrip('/')}-actions-{emunum}.txt", "w") as f:
            f.write("|".join(global_actions))

    del callbacks, checkpoint_callback, current_stats


def train_model(
    env,
    total_steps,
    n_steps,
    batch_size,
    episodes,
    file_name,
    save_path="ofo",
    device="cpu",
    train_freq=8,
    hours=4,
    num_envs=1,
    render_interval=4,
):
    """Single-model training loop (original behavior)."""
    model, starting_episode, checkpoint_path = create_model(
        env, total_steps, n_steps, batch_size, device, train_freq, hours, save_path
    )

    for episode in range(starting_episode, episodes + 1):
        train_episode(
            model, env, episode, total_steps, n_steps, num_envs,
            render_interval, checkpoint_path,
        )

    return model


def train_ab_test(
    env_exploration,
    env_control,
    total_steps,
    n_steps,
    batch_size,
    episodes,
    save_path="ofo",
    device="cpu",
    train_freq=8,
    hours=4,
    num_envs_exploration=1,
    num_envs_control=1,
    render_interval=4,
):
    """
    A/B test: train two fully independent models side by side.

    Each model gets its own VecEnv, its own weights, its own checkpoints, and
    its own tensorboard log directory. They are trained in alternating episodes
    so they get roughly equal wall-clock time on the GPU/CPU.

    Args:
        env_exploration: VecEnv with exploration wrappers applied.
        env_control: VecEnv without exploration wrappers (control group).
        total_steps: Timesteps per episode per model.
        Other args: same as train_model.
    """
    exploration_path = f"{save_path.rstrip('/')}/exploration"
    control_path = f"{save_path.rstrip('/')}/control"
    os.makedirs(exploration_path, exist_ok=True)
    os.makedirs(control_path, exist_ok=True)

    model_exp, start_exp, cp_exp = create_model(
        env_exploration, total_steps, n_steps, batch_size, device, train_freq,
        hours, exploration_path, label="exploration",
    )
    model_ctrl, start_ctrl, cp_ctrl = create_model(
        env_control, total_steps, n_steps, batch_size, device, train_freq,
        hours, control_path, label="control",
    )

    starting_episode = max(start_exp, start_ctrl)

    for episode in range(starting_episode, episodes + 1):
        print(f"\n{'='*60}")
        print(f"  A/B TEST — Episode {episode}/{episodes}")
        print(f"{'='*60}\n")

        print(f"--- EXPLORATION MODEL (with wrappers) ---")
        train_episode(
            model_exp, env_exploration, episode, total_steps, n_steps,
            num_envs_exploration, render_interval, cp_exp, label="exploration",
        )

        print(f"\n--- CONTROL MODEL (no wrappers) ---")
        train_episode(
            model_ctrl, env_control, episode, total_steps, n_steps,
            num_envs_control, render_interval, cp_ctrl, label="control",
        )

    return model_exp, model_ctrl


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

    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--render_interval", type=int, default=4)
    parser.add_argument("--train_freq", type=int, default=8)

    # Exploration wrapper arguments
    parser.add_argument("--use-count-exploration", action="store_true",
                        help="Enable count-based exploration bonus wrapper")
    parser.add_argument("--count-scale", type=float, default=1.0,
                        help="Scale factor for count-based exploration bonus")
    parser.add_argument("--count-exponent", type=float, default=0.5,
                        help="Decay exponent for visit count bonus (0.5=sqrt)")
    parser.add_argument("--count-decay", type=float, default=1.0,
                        help="Decay rate for visit counts between episodes (1.0=persist, 0.0=reset)")
    parser.add_argument("--use-novelty-bonus", action="store_true",
                        help="Enable episodic state novelty bonus wrapper")
    parser.add_argument("--novelty-bonus", type=float, default=0.5,
                        help="Bonus for encountering a novel state this episode")
    parser.add_argument("--revisit-novelty-bonus", type=float, default=0.5,
                        help="Extra bonus for novel state at a previously-visited position")
    parser.add_argument("--ab-test", action="store_true",
                        help="A/B test: even-numbered envs get exploration wrappers, odd ones don't")

    args = parser.parse_args()

    num_cpu = args.num_envs

    run_env = None

    # TODO: Various hyperparameter tuning:
    # https://stackoverflow.com/questions/76076904/in-stable-baselines3-ppo-what-is-nsteps try using whole batch of n_steps as batch size?

    episodes = args.episodes

    batch_size = args.batch_size

    n_steps = args.n_steps

    # approximate hours of play according to in-game timer per env per episode
    # (certain actions stop the in-game timer)
    hours = args.hours


    # each step is (PRESS_FRAMES + RELEASE_FRAMES) frames long, at 60fps.  
    seconds = hours * 64 * 64
    total_steps = int(seconds * (60 // (PRESS_FRAMES + RELEASE_FRAMES)) * num_cpu)
    

    wrapper_kwargs_on = dict(
        use_count_exploration=args.use_count_exploration,
        count_scale=args.count_scale,
        count_exponent=args.count_exponent,
        count_decay=args.count_decay,
        use_novelty_bonus=args.use_novelty_bonus,
        novelty_bonus=args.novelty_bonus,
        revisit_novelty_bonus=args.revisit_novelty_bonus,
    )
    wrapper_kwargs_off = dict(
        use_count_exploration=False,
        count_scale=0,
        count_exponent=0.5,
        count_decay=1.0,
        use_novelty_bonus=False,
        novelty_bonus=0,
        revisit_novelty_bonus=0,
    )

    def build_vec_env(n_envs, wrapper_kwargs, emunum_offset=0):
        """Build a DummyVecEnv or SubprocVecEnv with the given wrapper config."""
        steps_per_env = total_steps // num_cpu  # keep step budget consistent
        if n_envs == 1:
            return DummyVecEnv([
                make_env(args.game_path, emunum_offset, device=device, n_steps=n_steps,
                         num_steps=steps_per_env, **wrapper_kwargs)
            ])
        return SubprocVecEnv([
            make_env(args.game_path, emunum_offset + i, device=device,
                     state_file=args.state_file, num_steps=steps_per_env, **wrapper_kwargs)
            for i in range(n_envs)
        ])

    if args.ab_test:
        # Split CPUs evenly between the two models
        n_exploration = num_cpu // 2
        n_control = num_cpu - n_exploration
        if n_exploration < 1 or n_control < 1:
            print("ERROR: --ab-test requires at least 2 envs (--num_envs >= 2)")
            sys.exit(1)

        print(f"A/B test mode: {n_exploration} envs for EXPLORATION model, "
              f"{n_control} envs for CONTROL model (total {num_cpu})")

        env_exploration = build_vec_env(n_exploration, wrapper_kwargs_on, emunum_offset=0)
        env_control = build_vec_env(n_control, wrapper_kwargs_off, emunum_offset=n_exploration)

        train_ab_test(
            env_exploration=env_exploration,
            env_control=env_control,
            total_steps=total_steps,
            n_steps=n_steps,
            batch_size=batch_size,
            episodes=episodes,
            save_path=args.output_dir,
            device=device,
            train_freq=args.train_freq,
            hours=hours,
            num_envs_exploration=n_exploration,
            num_envs_control=n_control,
            render_interval=args.render_interval,
        )
    else:
        # Single-model mode (original behavior)
        run_env = build_vec_env(num_cpu, wrapper_kwargs_on)

        train_model(
            env=run_env,
            total_steps=total_steps,
            n_steps=n_steps,
            batch_size=batch_size,
            episodes=episodes,
            file_name="model",
            save_path=args.output_dir,
            device=device,
            train_freq=args.train_freq,
            hours=hours,
            num_envs=num_cpu,
            render_interval=args.render_interval,
        )
