import multiprocessing
import glob
from pyboy import PyBoy
from stable_baselines3 import PPO
import gym
import numpy as np
import torch
import torch.nn as nn
from timg import Renderer, Ansi24HblockMethod
from PIL import Image

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
            lr_schedule=lambda _: 0.001,  # Replace with your learning rate schedule
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs={},
            net_arch=[128, 128],  # Define the architecture of the network
            activation_fn=nn.ReLU
        )

# Define your custom Gym environment
class PyBoyEnv(gym.Env):
    def __init__(self, game_path):
        super(PyBoyEnv, self).__init__()
        self.pyboy = PyBoy(game_path, window_type='headless')
        
        # Define the memory range for 'number of Pokémon caught'
        self.caught_pokemon_start = 0xD2F7
        self.caught_pokemon_end = 0xD309

        # Define action_space and observation_space
        self.action_space = gym.spaces.Discrete(8)  # Modify as needed
        # Here we define the observation space to include the game's entire memory
        # and an additional value for the 'number of Pokémon caught'
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(0x10000 + 1,), dtype=np.uint8)

    def step(self, action):
        # Apply an action to the emulator
        self.pyboy.send_input(action)

        # Read the entire memory and the specific range for Pokémon caught
        memory_values = np.array([self.pyboy.get_memory_value(address) for address in range(0x10000)])
        pokemon_caught = sum([self.pyboy.get_memory_value(address) for address in range(self.caught_pokemon_start, self.caught_pokemon_end + 1)])
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
        self.pyboy
        memory_values = np.array([self.pyboy.get_memory_value(address) for address in range(0x10000)])
        pokemon_caught = sum([self.pyboy.get_memory_value(address) for address in range(self.caught_pokemon_start, self.caught_pokemon_end + 1)])
        observation = np.append(memory_values, pokemon_caught)
        # use timg to output the current screen
        obj = Renderer()
        obj.load_image(Image.fromarray(self.pyboy.botsupport_manager().screen().screen_ndarray()))
        obj.render(Ansi24HblockMethod)
        return observation


    def close(self):
        self.pyboy.stop()


# Function to create and train the PPO model
# def train_model(env, steps):
#     policy_kwargs = dict(
#         net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])],
#         activation_fn=torch.nn.ReLU
#     )
#     model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, device='cuda')
#     model.policy = CustomNetwork(env.observation_space, env.action_space)  # Use the custom network
#     model.learn(total_timesteps=steps)
#     return model

def train_model(env, steps):
    # You can adjust the policy keyword arguments as needed
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),  # Define the architecture of the policy and value networks
        activation_fn=torch.nn.ReLU  # Activation function
    )
    # Autodetect mps, cuda, rocm
    device = 'cpu'
    device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, device=device)  # Use 'cpu' if GPU is not available
    model.learn(total_timesteps=steps)
    return model

# Function to run the emulator with the Gym environment
def run_emulator(game_path, emunum=0, steps=0, result_queue=None):
    env = PyBoyEnv(game_path)
    model = train_model(env, steps)

    obs = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

    env.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the path to the game as a command line argument.")
        sys.exit(1)

    game_path = sys.argv[1]

    num_cores = multiprocessing.cpu_count()
    # games = glob.glob(game_path)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    processes = []

    for i in range(num_cores):
        process = multiprocessing.Process(target=run_emulator, args=(game_path, i, 20000, result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    num_cores = multiprocessing.cpu_count()
    

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    processes = []

    for i in range(num_cores):
        process = multiprocessing.Process(target=run_emulator, args=(game_path, i, 20000, result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()