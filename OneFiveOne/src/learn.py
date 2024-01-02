import multiprocessing
import os
from pyboy import PyBoy
from stable_baselines3 import PPO
import gym
import numpy as np
import torch
import torch.nn as nn
from timg import Renderer, Ansi24HblockMethod
from PIL import Image
import io
import base64
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import sys

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.extractor = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

class CustomNetwork(ActorCriticPolicy):
    def __init__(self, observation_space, action_space):
        super(CustomNetwork, self).__init__(
            observation_space,
            action_space,
            lr_schedule=lambda _: 0.001,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs={},
            net_arch=[128, 128],
            activation_fn=nn.ReLU,
        )

class PyBoyEnv(gym.Env):
    def __init__(self, game_path, emunum):
        super(PyBoyEnv, self).__init__()
        self.pyboy = PyBoy(game_path, window_type='headless')
        self.caught_pokemon_start = 0xD2F7
        self.caught_pokemon_end = 0xD309
        self.seen_events = set()
        self.emunum = emunum
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(698,), dtype=np.uint8)

    def step(self, action, ticks=1):
        self.pyboy.send_input(action)
        for i in range(ticks):
            self.pyboy.tick()

        if (self.emunum == 0 and self.pyboy.frame_count % 1000 == 0) or np.random.randint(777) == 0:
            obj = Renderer()
            obj.load_image(Image.fromarray(self.pyboy.botsupport_manager().screen().screen_ndarray()))
            obj.render(Ansi24HblockMethod)
            print("OS:FRAME:", self.pyboy.frame_count)
            print("OS:EMUNUM:", self.emunum)

        flo = io.BytesIO()
        flo.seek(0)
        self.pyboy.save_state(flo)
        flo.seek(0)
        memory_values = np.frombuffer(flo.read(), dtype=np.uint8)
        pokemon_caught = sum(memory_values[self.caught_pokemon_start:self.caught_pokemon_end])
        memory_values = memory_values[0xD5A6:0xD85F]
        observation = np.append(memory_values, pokemon_caught)
        reward = pokemon_caught
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        flo = io.BytesIO()
        flo.seek(0)
        self.pyboy.save_state(flo)
        flo.seek(0)
        memory_values = np.frombuffer(flo.read(), dtype=np.uint8)
        pokemon_caught = sum(memory_values[self.caught_pokemon_start:self.caught_pokemon_end])
        unique_events = memory_values[0xD5A6:0xD85F]
        hashable_strings = "".join([base64.b64encode(bytes(chunk)).decode("utf-8") for chunk in unique_events])

        if hashable_strings not in self.seen_events:
            self.seen_events.add(hashable_strings)

        memory_values = memory_values[0xD5A6:0xD85F]
        observation = np.append(memory_values, pokemon_caught + len(self.seen_events))
        obj = Renderer()
        obj.load_image(Image.fromarray(self.pyboy.botsupport_manager().screen().screen_ndarray()))
        obj.render(Ansi24HblockMethod)
        print("OS:SHAPE:", observation.shape)
        return observation

    def close(self):
        self.pyboy.stop()

def train_model(env, steps, model_path=None):
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path, env)
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),
            activation_fn=torch.nn.ReLU
        )
        device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'
        device = 'cuda' if torch.cuda.is_available() else device
        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, device=device)

    model.learn(total_timesteps=steps)

    if model_path:
        model.save(model_path)

    return model

def run_emulator(game_path, save_state_path, model_path, emunum=0, steps=0, result_queue=None):
    env = PyBoyEnv(game_path, emunum=emunum)
    env.pyboy.load_state(open(save_state_path, 'rb'))
    env.pyboy.set_emulation_speed(0)
    model = train_model(env, steps, model_path)

    obs = env.reset()
    actions = [None] * steps
    ticks = 10
    for i in range(steps):
        action, _ = model.predict(obs)
        actions[i] = action.tolist()
        obs = env.step(action, ticks=ticks)[0]
    print(f"{len(actions)} Actions in thread {emunum}: {''.join([str(x) for x in actions])}")
    env.close()

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    # Ensure 'steps' is an integer
    steps = int(625 * num_cores)
    if len(sys.argv) < 3:
        print("Usage: script.py <game_path> <save_state_path> [<model_path>]")
        sys.exit(1)

    game_path = sys.argv[1]
    save_state_path = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    processes = []

    for i in range(num_cores):
        process = multiprocessing.Process(
            target=run_emulator,
            args=(game_path, save_state_path, model_path, i, steps, result_queue)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print(result_queue.join())
