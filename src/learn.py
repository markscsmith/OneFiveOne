import multiprocessing
from pyboy import PyBoy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from timg import Renderer, Ansi24HblockMethod, Ansi8HblockMethod
from PIL import Image
from os.path import exists
import gymnasium as gym
import numpy as np
import base64
import torch
import io
import os

class PokeCaughtCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PokeCaughtCallback, self).__init__(verbose)
        self.timg_render = Renderer()

    def _on_step(self) -> bool:
        # Retrieve pokemon_caught from each environment
        all_pokemon_caught = self.training_env.get_attr('last_pokemon_count')
        visiteds = self.training_env.get_attr('visited_xy')
        rewards = self.training_env.get_attr('last_score')
        frames = self.training_env.get_attr('frames')
        renders = self.training_env.env_method('generate_image')
        stationary_frames = self.training_env.get_attr('stationary_frames')
        xs = self.training_env.get_attr('last_player_x')
        ys = self.training_env.get_attr('last_player_y')
        xbs = self.training_env.get_attr('last_player_x_block')
        ybs = self.training_env.get_attr('last_player_y_block')
        # Example: print or process the retrieved values

        # Find the best performing environment
        # best_env_idx = np.argmax(rewards)
        best_env_idx = np.argmax(stationary_frames)

        

        # for env_num, (pokemon_count, visited, steps, reward)  in enumerate(zip(all_pokemon_caught, visiteds, frames, rewards)):
        # for env_num, (pokemon_count, visited, steps, reward)  in enumerate(zip(all_pokemon_caught, visiteds, frames, rewards)):
        #     print(f"{str(env_num).zfill(3)} 🟣 {pokemon_count} 🎬 {steps} 🌎 {len(visited)} 🏆 {reward}")

        best_image = renders[best_env_idx]
        
        terminal_size = os.get_terminal_size()
        # convert best image to grayscale
        self.timg_render.load_image(best_image)
        
        if terminal_size.columns < 160 or terminal_size.lines < 144 / 2:
            self.timg_render.resize(terminal_size.columns,
                                    terminal_size.lines * 2)
            
        self.timg_render.render(Ansi24HblockMethod)
        print(f"Best: {best_env_idx} 🟢 {all_pokemon_caught[best_env_idx]} 🎬 {frames[best_env_idx]} 🌎 {len(visiteds[best_env_idx])} 🏆 {rewards[best_env_idx]} 🦶 {stationary_frames[best_env_idx]} X: {xs[best_env_idx]} Y: {ys[best_env_idx]} XB: {xbs[best_env_idx]} YB: {ybs[best_env_idx]}")
        
        
        return True

class PyBoyEnv(gym.Env):
    def __init__(self, game_path, emunum, save_state_path=None):
        super(PyBoyEnv, self).__init__()
        self.pyboy = PyBoy(game_path, window_type="headless")
        self.renderer = Renderer()
        # Define the memory range for 'number of Pokémon caught'
        self.caught_pokemon_start = 0xD2F7
        self.caught_pokemon_end = 0xD309
        self.player_x_mem = 0xD361
        self.player_y_mem = 0xD362
        self.player_x_block_mem = 0xD363
        self.player_y_block_mem = 0xD364 
        self.seen_events = set()
        self.emunum = emunum
        self.save_state_path = save_state_path
        self.visited_xy = set()
        self.last_score = 0
        self.last_pokemon_count = 0
        self.frames = 0
        self.stationary_frames = 0
        self.last_player_x = 0
        self.last_player_y = 0
        self.last_player_x_block = 0
        self.last_player_y_block = 0
        

        

        # Define actioqn_space and observation_space
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(698,),
                                                 dtype=np.uint8)
    
    def generate_image(self):
        return Image.fromarray(self.pyboy.botsupport_manager().screen().screen_ndarray())

    def render(self):
        terminal_size = os.get_terminal_size()
        self.renderer.load_image(
             Image.fromarray(
                 self.pyboy.botsupport_manager().screen().screen_ndarray())
        )
        if terminal_size.columns < 160 or terminal_size.lines < 144 / 2:
            self.renderer.resize(terminal_size.columns,
                                 terminal_size.lines * 2 * 160 // 144)
        self.renderer.render(Ansi24HblockMethod)
        
        

    def step(self, action):
        self.frames = self.pyboy.frame_count
        ticks = 24
        self.pyboy.send_input(action)
        for _ in range(ticks):
            self.pyboy.tick()
        # flo = io.BytesIO()
        # self.pyboy.save_state(flo)
        # flo.seek(0)
        # memory_values = np.frombuffer(flo.read(), dtype=np.uint8)
        memory_values = [self.pyboy.get_memory_value(i) for i in range(0x10000)]
        pokemon_caught = sum(
            memory_values[self.caught_pokemon_start: self.caught_pokemon_end]
        )
        

        px = memory_values[self.player_x_mem]
        py = memory_values[self.player_y_mem]
        pbx = memory_values[self.player_x_block_mem]
        pby = memory_values[self.player_y_block_mem]
        if self.last_player_x == px and self.last_player_y == py and self.last_player_x_block == pbx and self.last_player_y_block == pby:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0
            self.last_player_x = px
            self.last_player_y = py
            self.last_player_x_block = pbx
            self.last_player_y_block = pby

        self.visited_xy.add((px, py, pbx, pby))

        memory_values = memory_values[0xD5A6:0xD85F]
        observation = np.append(memory_values, (pokemon_caught + 1) * len(self.visited_xy))

        reward = (pokemon_caught * 100) + (len(self.visited_xy) - (self.stationary_frames // 10))

        # if np.random.randint(777) == 0 or self.last_pokemon_count != pokemon_caught or self.last_score - reward > 100:
        #     self.render()
        #
        self.last_score = reward
        terminated = False
        truncated = False
        info = {}
        self.last_pokemon_count = pokemon_caught
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        print("OS:SHAPE:", self.observation_space.shape)
        # memory_values = self.pyboy.mb.ram.internal_ram0.append(self.pyboy.mb.ram.internal_ram1)
        # memory_values = np.array([self.pyboy.get_memory_value(address) for address in range(0x10000)])
        memory_values = [self.pyboy.get_memory_value(i) for i in range(0x10000)]
        pokemon_caught = sum(
            memory_values[self.caught_pokemon_start: self.caught_pokemon_end]
        )
        unique_events = memory_values[0xD5A6:0xD85F]
        hashable_strings = "".join(
            [base64.b64encode(bytes(chunk)).decode("utf-8")
             for chunk in unique_events]
        )

        if hashable_strings not in self.seen_events:
            self.seen_events.add(hashable_strings)
            # print("OS:EVENTS:", hashable_strings)

        memory_values = memory_values[0xD5A6:0xD85F]
        seed = 0
        observation = np.append(
            memory_values, pokemon_caught + len(self.seen_events))
        # use timg to output the current screen

        # obj = Renderer()

        # obj.load_image(
        #     Image.fromarray(
        #         self.pyboy.botsupport_manager().screen().screen_ndarray())
        # )
        # terminal_size = os.get_terminal_size()
        # if terminal_size.columns < 160 or terminal_size.lines < 144 / 2:
        #     obj.resize(terminal_size.columns,
        #                terminal_size.lines * 2 * 160 // 144)
        # obj.render(Ansi24HblockMethod)
        print("OS:SHAPE:", observation.shape, seed)
        return observation, {"seed": seed}
    



def make_env(game_path, emunum):
    def _init():
        env = PyBoyEnv(game_path, emunum=emunum)
        
        if os.path.exists(game_path + ".state"):
             env.pyboy.load_state(open(game_path + ".state", "rb"))
        env.pyboy.set_emulation_speed(0)
        return env
    return _init


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str)
    args = parser.parse_args()
    num_cpu = multiprocessing.cpu_count()

    current_stats = EveryNTimesteps(n_steps=1000, callback=PokeCaughtCallback())


    # num_cpu = 1
    env = SubprocVecEnv([make_env(args.game_path,
                                  emunum) for emunum in range(num_cpu)])
    
    steps = 20000 * num_cpu
    file_name = "model"
    def train_model(env, num_steps):
        device = "cpu"
        device = (
            "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else device
        )
        device = "cuda" if torch.cuda.is_available() else device
        
        if exists(file_name + '.zip'):
            print('\nloading checkpoint')
            model = PPO.load(file_name, env=env, device=device)
#           model.n_steps = steps
            model.n_envs = num_cpu
#            model.rollout_buffer.buffer_size = steps
            model.rollout_buffer.n_envs = num_cpu
            model.rollout_buffer.reset()
            
        else:
            model = PPO("MlpPolicy", env, verbose=1, device=device)
        # TODO: Progress callback that collects data from each frame for stats
        model.learn(total_timesteps=num_steps, progress_bar=True, callback=current_stats)
        return model
    runsteps = 3000000 * 6 # hrs
    model = train_model(env, runsteps)
    model.save(f"{file_name}.zip")