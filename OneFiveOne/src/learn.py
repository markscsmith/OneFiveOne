import multiprocessing
from pyboy import PyBoy
from stable_baselines3 import PPO
import gym
import os
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
        # Define your custom layers here
        self.extractor = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), features_dim),
            nn.ReLU(),
            # You can add more layers here
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)


class CustomNetwork(ActorCriticPolicy):
    def __init__(self, observation_space, action_space):
        super(CustomNetwork, self).__init__(
            observation_space,
            action_space,
            lr_schedule=lambda _: 0.002,  # Replace with your learning rate schedule
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs={},
            net_arch=[128, 128],  # Define the architecture of the network
            activation_fn=nn.ReLU
        )


# Define your custom Gym environment
class PyBoyEnv(gym.Env):
    def __init__(self, game_path, emunum):
        super(PyBoyEnv, self).__init__()
        self.pyboy = PyBoy(game_path, window_type='headless')

        # Define the memory range for 'number of Pokémon caught'
        self.caught_pokemon_start = 0xD2F7
        self.caught_pokemon_end = 0xD309
        self.seen_events = set()
        self.emunum = emunum

        # Define actioqn_space and observation_space
        self.action_space = gym.spaces.Discrete(8)  # Modify as needed
        # Here we define the observation space to include the game's entire memory
        # and an additional value for the 'number of Pokémon caught'
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(0x10000 + 1,), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(698,), dtype=np.uint8)

    def step(self, action, ticks=1):
        # Apply an action to the emulator

        self.pyboy.send_input(action)
        for i in range(ticks):
            self.pyboy.tick()

        if (self.emunum == 0 and self.pyboy.frame_count % 1000 == 0) or np.random.randint(777) == 0:
            obj = Renderer()
            obj.load_image(Image.fromarray(self.pyboy.botsupport_manager().screen().screen_ndarray()))
            terminal_size = os.get_terminal_size()
            obj.resize(terminal_size.columns, terminal_size.lines * 2 * 160 // 144)
            obj.render(Ansi24HblockMethod)
            print("OS:FRAME:", self.pyboy.frame_count)
            print("OS:EMUNUM:", self.emunum)

        # Read the entire memory and the specific range for Pokémon caught
        # memory_values = np.array([self.pyboy.get_memory_value(address) for address in range(0x10000)])
        flo = io.BytesIO()
        flo.seek(0)
        self.pyboy.save_state(flo)
        flo.seek(0)
        memory_values = np.frombuffer(flo.read(), dtype=np.uint8)
        # pokemon_caught = sum([self.pyboy.get_memory_value(address) for address in range(self.caught_pokemon_start, self.caught_pokemon_end + 1)])
        pokemon_caught = sum(memory_values[self.caught_pokemon_start:self.caught_pokemon_end])

        memory_values = memory_values[0xD5A6:0xD85F]
        observation = np.append(memory_values, pokemon_caught)
        # if self.pyboy.frame_count % 10000 == 0:
        #     obj = Renderer()
        #     obj.load_image(Image.fromarray(self.pyboy.botsupport_manager().screen().screen_ndarray()))
        #     obj.render(Ansi24HblockMethod)
        # Define your reward and done condition based on your specific game logic
        reward = pokemon_caught  # Modify as per your game logic
        done = False  # Define your game's ending condition

        info = {}
        return observation, reward, done, info

    def reset(self):
        print("OS:SHAPE:", self.observation_space.shape)
        # memory_values = self.pyboy.mb.ram.internal_ram0.append(self.pyboy.mb.ram.internal_ram1)
        # memory_values = np.array([self.pyboy.get_memory_value(address) for address in range(0x10000)])
        flo = io.BytesIO()
        flo.seek(0)
        self.pyboy.save_state(flo)
        flo.seek(0)
        memory_values = np.frombuffer(flo.read(), dtype=np.uint8)
        pokemon_caught = sum(memory_values[self.caught_pokemon_start:self.caught_pokemon_end])
        unique_events = memory_values[0xD5A6:0xD85F]
        hashable_strings = "".join([base64.b64encode(bytes(chunk)).decode('utf-8') for chunk in unique_events])

        if hashable_strings not in self.seen_events:
            self.seen_events.add(hashable_strings)
            # print("OS:EVENTS:", hashable_strings)

        memory_values = memory_values[0xD5A6:0xD85F]
        seed = 0
        observation = np.append(memory_values, pokemon_caught + len(self.seen_events))
        # use timg to output the current screen

        
        obj = Renderer()
        
        obj.load_image(Image.fromarray(self.pyboy.botsupport_manager().screen().screen_ndarray()))
        terminal_size = os.get_terminal_size()
        obj.resize(terminal_size.columns, terminal_size.lines * 2 * 160 // 144)
        obj.render(Ansi24HblockMethod)
        print("OS:SHAPE:", observation.shape, seed)
        return observation

    def close(self):
        self.pyboy.stop()


def train_model(env, steps):
    # You can adjust the policy keyword arguments as needed
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),  # Define the architecture of the policy and value networks
        activation_fn=torch.nn.ReLU  # Activation function
    )
    # Autodetect mps, cuda, rocm
    device = 'cpu'
    device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else device
    device = 'cuda' if torch.cuda.is_available() else device

    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, device=device)  # Use 'cpu' if GPU is not available
    model.learn(total_timesteps=steps)
    return model


# Function to run the emulator with the Gym environment
def run_emulator(game_path, save_state_path, emunum=0, steps=0, result_queue=None):
    env = PyBoyEnv(game_path, emunum=emunum)
    env.pyboy.load_state(open(save_state_path, 'rb'))
    env.pyboy.set_emulation_speed(0)
    model = train_model(env, steps)

    obs = env.reset()
    actions = [None] * steps  # Create a pre-sized list of length "steps"
    ticks = 10
    for i in range(steps):
        action, _ = model.predict(obs)
        # print("OS:ACTION:", action)

        actions[i] = action.tolist()
        obs = env.step(action, ticks=ticks)[0]
    print(f"{len(actions)} Actions in thread {emunum}:", "".join([str(x) for x in actions]))
    env.close()

if __name__ == '__main__':
    steps = 20000
    if len(sys.argv) < 2:
        print("Please provide the path to the game as a command line argument.")
        sys.exit(1)

    game_path = sys.argv[1]
    save_state_path = sys.argv[2]

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    # games = glob.glob(game_path)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    processes = []

    for i in range(num_cores):
        process = multiprocessing.Process(target=run_emulator, args=(game_path, save_state_path, i, steps, result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print(result_queue.join())