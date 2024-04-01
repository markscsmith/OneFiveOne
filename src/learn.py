import sys
import time
import os
import datetime
import hashlib
from os.path import exists

# Compute and AI libs
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy

# multiprocessing environment for parallel training
import multiprocessing
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv

# Emulator libs
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Output libs
from timg import Renderer, Ansi24HblockMethod
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Memory ranges to read in Pokemon Red/Blue (+ Yellow?)
# MEM_START = 0xCC3C
MEM_START = 0xD2F7
MEM_END = 0xDEE1

class PokeCart():
    def __init__(self, cart_data) -> None:
        # calculate checksum of cart_data
        self.cart_data = cart_data
        self.checksum = hashlib.md5(cart_data).hexdigest()

    def identify_cart(self):
        # identify cart
        carts = {"a6924ce1f9ad2228e1c6580779b23878":  "POKEMONG.GBC",
                 "9f2922b235a5eeb78d65594e82ef5dde":  "PMCRYSTA.GBC",
                 # TODO: Add Pokemon Yellow logic to keep Pikachu happy. 🌩️🐭  Address is 0xD46F    1    Pikachu's happiness per https://datacrystal.tcrf.net/wiki/Pokémon_Yellow/RAM_map
                 "d9290db87b1f0a23b89f99ee4469e34b":  "POKEMONY.GBC",
                 "50927e843568814f7ed45ec4f944bd8b":  "POKEMONB.GBC",
                 "3e098020b56c807393cc2ebae5e1857a":  "POKEMONS.GBC",
                 "3d45c1ee9abd5738df46d2bdda8b57dc":  "POKEMONR.GBC", }
        if self.checksum in carts:
            return carts[self.checksum]
        else:
            print("Unknown cart:", self.checksum)
            return None

    def cart_offset(self):
        # Pokemon Yellow has offset -1 vs blue and green
        # TODO: Pokemon Gold Silver and Crystal
        carts = {
            "POKEMONR.GBC": MEM_START,
            "POKEMONB.GBC": MEM_START,
            # I now suddenly understand what was meant by this comment from https://datacrystal.tcrf.net/wiki/Pokémon_Yellow/RAM_map: "The RAM map for this game has an offset of -1 from the one on Red and Blue."
            # I think I tried this before, but I didn't grok it at the time due to other memory read glitches and bugs I introduced
            "POKEMONY.GBC": MEM_START + 1,
            "POKEMONG.GBC": 0,
            "PMCRYSTA.GBC": 0,
            "POKEMONS.GBC": 0,
        }
        if self.identify_cart() in carts:
            return carts[self.identify_cart()]
        else:
            print("Unknown cart:", self.checksum)
            return 0x00000000


def learning_rate_schedule(progress):
    # return 0.025
    # progress starts at 1 and decreases as remaining approaches 0.
    rate = 0.0003
    variation = 0.2 * rate * progress
    # new_rate = rate + np.abs(variation * np.sin(progress * np.pi * 20)) # all positive
    new_rate = rate + variation * np.sin(progress * np.pi * 20) # positive and negative adjustments
    # rate = (rate + rate * progress) / 2
    print(f"LR: {new_rate}", file=sys.stderr)
    return new_rate
    # return  0.0


class CustomNetwork(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomNetwork, self).__init__(*args, **kwargs)
        self.lr_schedule = learning_rate_schedule


class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardLoggingCallback, self).__init__(verbose)
        # We set the frequency at which the callback will be called
        # This could be set to be called at each step by setting it to 1
        self.log_freq = 1000
        self.buttons_names = "UDLRABS!-udlrabs.-"

    def _on_step(self) -> bool:
        # This method will be called by the model after each call to `env.step()`.
        # Note: self.n_calls is incremented after this method is called.
        if self.n_calls % self.log_freq == 0:
            # Log scalar value (here a random variable)
            rewards = self.locals['rewards']
            infos = self.locals['infos']
            # todo: record each progress/reward separately like I do the actions?
            if len(rewards) > 0:  # Check if rewards list is not empty
                average_reward = sum(rewards) / len(rewards)
                max_reward = max(rewards)
                self.logger.record('reward/average_reward', average_reward)
                self.logger.record('reward/max_reward', max_reward)
            for _, info in sorted(enumerate(infos)):
                # TODO: ADD POKEMON CAUGHT TO INFO
                if all(key in info for key in ['actions', 'emunum', 'reward', 'frames']):
                    actions = info['actions']
                    emunum = info['emunum']
                    reward = info['reward']
                    frames = info['frames']
                    caught = info['pokemon_caught']
                    seen = info['pokemon_seen']
                    # TODO: pad emunumber with 0s to match number of digits in possible emunum
                    self.logger.record(
                        f"actions/{emunum}", f"{actions[-self.log_freq:-self.log_freq].lower()}{actions[-self.log_freq:]}:rew={reward}:fra={frames}:caught={caught}:seen={seen}")
                    self.logger.record(f"caught/{emunum}", f"{caught}")
                    self.logger.record(f"seen/{emunum}", f"{seen}")
                    self.logger.record(f"reward/{emunum}", f"{reward}")
                    self.logger.record(f"visited/{emunum}", f"{len(info['visited_xy'])}")
                    self.logger.record(f"items/{emunum}", f"{info['items']}")

                    
                    
                    
                    

        return True  # Returning True means we will continue training, returning False will stop training


class PokeCaughtCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(PokeCaughtCallback, self).__init__(verbose)

        self.timg_render = Renderer()
        self.filename_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.progress = 0
        self.total_timesteps = total_timesteps
        self.progress_bar = tqdm(
            total=self.total_timesteps, position=0, leave=True)

    def generate_gif_and_actions(self):
        actions = self.training_env.get_attr('actions')
        rewards = self.training_env.get_attr('last_score')
        frames = self.training_env.get_attr('frames')
        visiteds = self.training_env.get_attr('visited_xy')
        pokemon_caughts = self.training_env.get_attr('last_pokemon_count')

        if self.model.num_timesteps % 100000 == 0:
            hostname = os.uname()[1]
            file_name = f"{hostname}"
            # Generate a CSV of data
            # TODO: make path a parameter and use more sensible defaults for non-me users
            with open(f"/Volumes/Scratch/ofo/{file_name}.csv", "w", encoding="utf-8") as f:
                f.write("env_num,caught,actions,rewards,frames,visiteds\n")
                for env_num, (action, caught, reward, frame, visited) in enumerate(zip(actions, pokemon_caughts, rewards, frames, visiteds)):
                    f.write(
                        f"{env_num},{caught},{''.join(action)},{reward},{frame},\"{visited}\"\n")

    def _on_step(self) -> bool:
        # Retrieve pokemon_caught from each environment
        # all_pokemon_caught = self.training_env.get_attr('last_pokemon_count')
        # visiteds = self.training_env.get_attr('visited_xy')

        # frames = self.training_env.get_attr('frames')
        # stationary_frames = self.training_env.get_attr('stationary_frames')
        # xs = self.training_env.get_attr('last_player_x')
        # ys = self.training_env.get_attr('last_player_y')
        # xbs = self.training_env.get_attr('last_player_x_block')
        # ybs = self.training_env.get_attr('last_player_y_block')
        # map_ids = self.training_env.get_attr('last_player_map')
        # actions = self.training_env.get_attr('actions')
        rewards = self.training_env.get_attr('last_score')
        best_env_idx = np.argmax(rewards)
        self.training_env.env_method('render', best_env_idx)
        self.progress = self.model.num_timesteps
        self.progress_bar.update(self.progress - self.progress_bar.n)
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


class PyBoyEnv(gym.Env):
    def __init__(self, game_path, emunum, save_state_path=None, max_frames=500_000, **kwargs):
        super(PyBoyEnv, self).__init__()
        self.pyboy = PyBoy(game_path, window="null", cgb=True)
        self.game_path = game_path
        self.menu_value = None
        self.n = 21600  # 15 minutes of game time in frames
        self.last_n_frames = [self.pyboy.screen.ndarray] * self.n
        self.renderer = Renderer()
        self.actions = ""
        self.screen_images = []
        self.reset_unlocked = False
        # Define the memory range for 'number of Pokémon caught'
        self.max_frames = max_frames
        self.cart = PokeCart(open(game_path, "rb").read())
        self.caught_pokemon_start = 0xD2F7 - self.cart.cart_offset()
        self.caught_pokemon_end = 0xD309 - self.cart.cart_offset()
        self.seen_pokemmon_start = 0xD30A - self.cart.cart_offset()
        self.seen_pokemmon_end = 0xD31C - self.cart.cart_offset()
        self.player_x_mem = 0xD361 - self.cart.cart_offset()
        self.player_y_mem = 0xD362 - self.cart.cart_offset()
        self.player_x_block_mem = 0xD363 - self.cart.cart_offset()
        self.player_y_block_mem = 0xD364 - self.cart.cart_offset()
        self.player_map_mem = 0xD35E - self.cart.cart_offset()
        self.seen_events = set()
        self.emunum = emunum
        self.save_state_path = save_state_path
        self.visited_xy = set()
        self.last_score = 0
        self.last_pokemon_count = 0
        self.last_seen_pokemon_count = 0
        self.frames = 0
        self.stationary_frames = 0
        self.last_player_x = None
        self.last_player_y = None
        self.last_player_x_block = None
        self.last_player_y_block = None
        self.last_player_map = None
        self.screen_image_arrays = set()
        self.screen_image_arrays_list = []
        self.unchanged_frames = 0
        self.reset_penalty = 0
        self.player_maps = set()
        self.backtrack_bonus = 0
        self.item_points = {}
        self.last_items = []
        self.pokedex = {}

        self.speed_bonus = self.max_frames / (self.frames + 1)

        self.last_memory_update_frame = 0
        self.current_memory = self.get_memory_range()

        self.buttons = {
            0: (WindowEvent.PASS, "-"),
            1: (WindowEvent.PRESS_ARROW_UP, "U"),
            2: (WindowEvent.PRESS_ARROW_DOWN, "D"),
            3: (WindowEvent.PRESS_ARROW_LEFT, "L"),
            4: (WindowEvent.PRESS_ARROW_RIGHT, "R"),
            5: (WindowEvent.PRESS_BUTTON_A, "A"),
            6: (WindowEvent.PRESS_BUTTON_B, "B"),
            7: (WindowEvent.PRESS_BUTTON_START, "S"),
            8: (WindowEvent.PASS, "."),
            9: (WindowEvent.RELEASE_ARROW_UP, "u"),
            10: (WindowEvent.RELEASE_ARROW_DOWN, "d"),
            11: (WindowEvent.RELEASE_ARROW_LEFT, "l"),
            12: (WindowEvent.RELEASE_ARROW_RIGHT, "r"),
            13: (WindowEvent.RELEASE_BUTTON_A, "a"),
            14: (WindowEvent.RELEASE_BUTTON_B, "b"),
            15: (WindowEvent.RELEASE_BUTTON_START, "s"),

        }

        self.buttons_names = "UDLRABS!_udlrabs.-"

        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Format the datetime as a string suitable for a Unix filename
        self.filename_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Define actioqn_space and observation_space
        # self.action_space = gym.spaces.Discrete(256)
        # self.action_space = gym.spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        # self.action_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(8, start=0)
        size = MEM_END - MEM_START + 2
        # size = MEM_START MEM_END + 2
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size,), dtype=np.uint16)

    def generate_image(self):
        return self.pyboy.screen.ndarray

    def generate_screen_ndarray(self):
        return self.pyboy.screen.ndarray

    def calculate_reward(self):
        # calculate total bits from the memory values
        self.get_memory_range()
        offset = self.cart.cart_offset() - MEM_START
        pokemon_caught = sum(
            [bin(values).count('1')
             for values in self.current_memory[self.caught_pokemon_start + offset: self.caught_pokemon_end + offset]]
        )
        pokemon_seen = sum(
            [bin(values).count('1')
             for values in self.current_memory[self.seen_pokemmon_start + offset: self.seen_pokemmon_end + offset]]
        )

        self.pokedex = {i: bin(values).count('1')
                   for i, values in enumerate(self.current_memory[0xD31E + offset: 0xD345 + offset])}

        items = [bin(values).count('1')
                 for values in self.current_memory[0xD31E + offset: 0xD345 + offset]]
        item_types = [items[i] for i in range(0, len(items), 2)]
        item_counts = [items[i] for i in range(1, len(items), 2)]
        # calculate points of items based on the numer of items added per step

        
        item_diff = [np.abs(item_counts[i] - self.last_items[i]) for i in range(len(item_counts))]
            # create tuple of item type and points
        item_and_points = [(item_types[i], item_diff[i]) for i in range(len(item_diff))]
        if len(item_and_points) > 0:
            self.item_points = {item: 0 for item in item_and_points}
            
        for item, points in item_and_points:
            self.item_points[item] += points


        px = self.current_memory[self.player_x_mem]
        py = self.current_memory[self.player_y_mem]
        pbx = self.current_memory[self.player_x_block_mem]
        pby = self.current_memory[self.player_y_block_mem]
        map_id = self.current_memory[self.player_map_mem]
        self.player_maps.add(map_id)
        if self.last_player_x == px and self.last_player_y == py and self.last_player_x_block == pbx and self.last_player_y_block == pby and self.last_player_map == map_id:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0
            self.last_player_x = px
            self.last_player_y = py
            self.last_player_x_block = pbx
            self.last_player_y_block = pby
            self.last_player_map = map_id

        # convert binary chunks into a single string
        chunk_id = f"{px}:{py}:{pbx}:{pby}:{map_id}"

        self.visited_xy.add(chunk_id)

        self.last_pokemon_count = pokemon_caught
        # reward = pokemon_caught * 1000 + len(self.visited_xy) * 10 - self.stationary_frames * 10 - self.unchanged_frames * 10 - self.reset_penalty
        # More caught pokemon = more leeway for standing still
        # reward = int(pokemon_caught * 32000 // 152) + ((len(self.player_maps)) * (32000 // 255) * (2000  * (pokemon_caught + 1) - self.stationary_frames) / 2000 * (pokemon_caught + 1))

        if pokemon_seen == 0:
            pokemon_caught = 0

        if pokemon_caught > self.last_pokemon_count:
            # Give a backtrack bonus and reset the explored list
            self.backtrack_bonus = len(self.visited_xy)
            self.visited_xy = set()
            
        self.last_pokemon_count = pokemon_caught
        self.last_seen_pokemon_count = pokemon_seen
        reward = (len(self.player_maps) * 1000 + (self.backtrack_bonus + len(self.visited_xy)) // 10) // 10
        reward = reward + (reward * (pokemon_caught * 2) + (pokemon_seen)) // 150 + sum(self.item_points.values()) * 10

        self.speed_bonus = self.max_frames / (self.frames + 1)

        reward = (reward) - (reward *
                             (self.stationary_frames / (self.frames + 1))) + ((reward * self.speed_bonus) // 10)

        return reward

    def render(self, target_index=None, reset=False):
        if target_index is not None and target_index == self.emunum or reset:
            terminal_size = os.get_terminal_size()
            terminal_offset = 6

            image = self.pyboy.screen.image
            w = 160
            h = 144
            if terminal_size.columns != w or terminal_size.lines < h / 2:
                image_aspect_ratio = w / h
                terminal_aspect_ratio = terminal_size.columns / \
                    (terminal_size.lines - terminal_offset)

                if image_aspect_ratio > terminal_aspect_ratio:
                    new_width = int(w / image_aspect_ratio)
                elif image_aspect_ratio < terminal_aspect_ratio:
                    new_width = int(w * image_aspect_ratio)
                else:
                    new_width = w

                new_height = h

                replacer = Image.new("RGB", (new_width, new_height), (0, 0, 0))
                # in center of image
                replacer.paste(image, ((new_width - image.width) // 2, 0))
                image = replacer

            self.renderer.load_image(image)
            self.renderer.resize(terminal_size.columns,
                                 terminal_size.lines * 2 - terminal_offset)
            self.renderer.render(Ansi24HblockMethod)
            item_score = sum(self.item_points.values())
            if target_index is not None:
                print(
                    f"Best:  {target_index:2d} 🟢 {self.last_pokemon_count:3d} 👀 {self.last_seen_pokemon_count:3d} 🎬 {self.frames:6d} 🌎 {len(self.visited_xy):3d}:{len(self.player_maps):3d} 🏆 {self.last_score:7.2f} 🎒 {item_score:3d} 🐆 {self.speed_bonus:7.2f}🦶 {self.stationary_frames:3d} X: {self.last_player_x:3d} Y: {self.last_player_y:3d} XB: {self.last_player_x_block:3d} YB: {self.last_player_y_block:3d}, Map: {self.last_player_map:3d} Actions {' '.join(self.actions[-6:])} {len(self.actions)}")
            if reset:
                print(
                    f"Reset: {self.emunum:2d} 🟢 {self.last_pokemon_count:3d} 👀 {self.last_seen_pokemon_count:3d} 🎬 {self.frames:6d} 🌎 {len(self.visited_xy):3d}:{len(self.player_maps):3d}🏆 {self.last_score:7.2f} 🎒 {item_score:3d} 🐆 {self.speed_bonus:7.2f} 🦶 {self.stationary_frames:3d} X: {self.last_player_x:3d} Y: {self.last_player_y:3d} XB: {self.last_player_x_block:3d} YB: {self.last_player_y_block:3d}, Map: {self.last_player_map:3d} Actions {' '.join(self.actions[-6:])} {len(self.actions)}")

    # TODO: build expanding pixel map to show extents of game travelled. (minimap?) Use 3d numpy array to store visited pixels. performance?

    def step(self, action):
        self.frames = self.pyboy.frame_count
        button_1, button_name_1 = self.buttons[action]
        button_2, _ = self.buttons[action + 8]
        self.pyboy.send_input(button_1)
        self.pyboy.tick()
        #ticks = 1
        #for _ in range(ticks):
        self.pyboy.send_input(button_2)
        self.actions = f"{self.actions}{button_name_1}"
        #self.actions = self.actions + (f"{button_name_1}")
        # Grab less frames to append if we're standing still.

        reward = round(self.calculate_reward(), 3)

        self.last_score = reward

        truncated = False
        # if self.frames >= self.max_frames:
        #     terminated = True
        # else:
        terminated = False

        info = {"reward": reward,
                "actions": self.actions,
                "emunum": self.emunum,
                "frames": self.frames,
                "pokemon_caught": self.last_pokemon_count,
                "pokemon_seen": self.last_seen_pokemon_count, 
                "visited_xy": self.visited_xy,
                "stationary_frames": self.stationary_frames,
                "items": self.item_points,}

        self.current_memory = self.get_memory_range()
        observation = np.append(self.current_memory, reward)
        return observation, reward, terminated, truncated, info

    def get_memory_range(self):
        memory_values = self.pyboy.memory[MEM_START: MEM_END + 1]
        return memory_values
    

    # def get_reward_memory_range(self):
    #     freq = 24
    #     # self.current_memory = self.get_memory_range()
    #     if self.last_memory_update_frame <= self.frames - 24 or self.last_memory_update_frame == 0:
    #         self.last_memory_update_frame = self.frames
    #         self.current_memory = self.get_memory_range()
    #     else:
    #         self.current_memory[self.caught_pokemon_start:self.caught_pokemon_end] = self.pyboy.memory[self.caught_pokemon_start + MEM_START: self.caught_pokemon_end + MEM_START]
    #         self.current_memory[self.seen_pokemmon_start:self.seen_pokemmon_end] = self.pyboy.memory[self.seen_pokemmon_start + MEM_START: self.seen_pokemmon_end + MEM_START]

    #     # FIXME TODO: Workaround for Pokemon Blue bug where the number of pokemon caught shoots up to 4 before any pokemon are seen or caught.


    #         self.current_memory[self.player_x_mem] = self.pyboy.memory[self.player_x_mem + MEM_START]
    #         self.current_memory[self.player_y_mem] = self.pyboy.memory[self.player_y_mem + MEM_START]
    #         self.current_memory[self.player_x_block_mem] = self.pyboy.memory[self.player_x_block_mem + MEM_START]
    #         self.current_memory[self.player_x_block_mem] = self.pyboy.memory[self.player_y_block_mem + MEM_START]
    #         self.current_memory[self.player_map_mem] = self.pyboy.memory[self.player_map_mem + MEM_START]
    #     return self.current_memory
    

    def reset(self, seed=0, **kwargs):
        # reward = self.calculate_reward()
        # observation = np.append(
        #     self.get_memory_range(), reward)

        self.stationary_frames = 0
        self.unchanged_frames = 0
        self.speed_bonus = self.max_frames / (self.frames + 1)
        # print("OS:RESET:", self.emunum, seed)
        super().reset(seed=seed, **kwargs)
        self.last_memory_update_frame = 0
        self.visited_xy = set()
        self.player_maps = set()
        self.reset_penalty = 0
        self.screen_image_arrays = set()
        self.screen_image_arrays_list = []
        self.last_player_x = 0
        self.last_player_y = 0
        self.last_player_x_block = 0
        self.last_player_y_block = 0
        self.menu_value = 0
        self.pokedex = {}
        self.pyboy = PyBoy(self.game_path, window="null", cgb=True)
        # self.last_n_frames = [self.pyboy.screen.ndarray] * self.n

        if self.save_state_path is not None:
            self.pyboy.load_state(open(self.save_state_path, "rb"))
        else:
            print(
                f"Error: No state file found for {self.save_state_path}", file=sys.stderr)
            exit(1)

        self.actions = ""
        self.screen_image_arrays = set()
        self.screen_image_arrays_list = []
        self.visited_xy = set()
        self.stationary_frames = 0
        self.last_score = 0
        self.last_pokemon_count = 0
        self.frames = 0
        self.last_player_x = 0
        self.last_player_y = 0
        self.last_player_x_block = 0
        self.last_player_y_block = 0
        reward = self.calculate_reward()
        observation = np.append(
            self.get_memory_range(), reward)
        print("RESET:OS:SHAPE:", observation.shape, seed, file=sys.stderr)
        return observation, {"seed": seed}


def make_env(game_path, emunum):
    def _init():
        if os.path.exists(game_path + ".state"):
            print(f"Loading state {game_path}.state")
            new_env = PyBoyEnv(game_path, emunum=emunum,
                               save_state_path=game_path + ".state")
            new_env.pyboy.load_state(open(game_path + ".state", "rb"))
        else:
            print(f"Error: No state file found for {game_path}.state")
            exit(1)

        new_env.pyboy.set_emulation_speed(0)
        return new_env
    return _init



def train_model(env, total_steps, steps, episode, file_name):
    policy_kwargs = dict(
        # features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs={},
        net_arch=dict(pi=[2048, 1024, 1024, 512], vf=[2048, 1024, 1024, 512]),
        activation_fn=nn.ReLU,
    )

    device = "cpu"
    device = (
        "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else device
    )
    device = "cuda" if torch.cuda.is_available() else device
    tensorboard_log = f"/Volumes/Scratch/ofo/tensorboard/{os.uname()[1]}-{time.time()-episode}"
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        run_model = PPO.load(file_name, env=env, device=device, tensorboard_log=tensorboard_log)
        run_model.rollout_buffer.reset()
        # run_model.tensorboard_log=f"/Volumes/Scratch/ofo/tensorboard/{os.uname()[1]}-{time.time()}",

    else:
        # n_steps = steps * num_cpu
        tensorboard_log = f"/Volumes/Scratch/ofo/tensorboard/{os.uname()[1]}-{time.time()}-{episode}"
        run_model = PPO(policy="MlpPolicy",
                        
                        # Reduce n_steps if too large; ensure not less than some minimum like 2048 for sufficient learning per update.
                        n_steps=steps,
                        # Reduce batch size if it's too large but ensure a minimum size for stability.
                        batch_size=steps // 8,
                        # Adjusted foor potentially more stable learning across batches.
                        n_epochs=3,
                        # Increased to give more importance to future rewards, can help escape repetitive actions.
                        gamma=0.9998,
                        # Adjusted for a better balance between bias and variance in advantage estimation.
                        gae_lambda=0.998,
                        learning_rate=learning_rate_schedule,  # Standard starting point for PPO, adjust based on performance.
                        # learning_rate=0.0002,
                        env=env,
                        # Ensure this aligns with the complexities of your environment.
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        device=device,
                        # Reduced for less aggressive exploration after initial learning, adjust based on needs.
                        ent_coef=0.01,
                        tensorboard_log=tensorboard_log,
                        # vf_coef=0.5,  # Adjusted to balance value function loss importance.
                        )

    # model_merge_callback = EveryNTimesteps(n_steps=steps * num_cpu * 1024, callback=ModelMergeCallback(args.num_hosts))
    # TODO: Progress callback that collects data from each frame for stats

    checkpoint_callback = None
    current_stats = None
    tbcallback = None

    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=f"/Volumes/Scratch/ofo_chkpt/{os.uname()[1]}-{time.time()}.zip", name_prefix="poke")
    current_stats = EveryNTimesteps(
        n_steps=10000, callback=PokeCaughtCallback(total_steps))
    tbcallback = TensorboardLoggingCallback(tensorboard_log)
    callbacks = [checkpoint_callback, current_stats, tbcallback]
    run_model.learn(total_timesteps=total_steps,
                    progress_bar=False, callback=callbacks)
    return run_model

if __name__ == "__main__":
    # TODO: make path a parameter and use more sensible defaults for non-me users
    
    import argparse
    parser = argparse.ArgumentParser()
    # TODO: Investigate Pokemon Blue caught = 4 before you have a pokedex. Is this a bug in the game?  Is it a bug in the emulator?  Is it a bug in the memory address? Seems to work fine on Red,
    # only happens during looking at starter pokemon in Oak's lab.

    # TODO: use more sensible defaults for non-me users
    # TODO: sensible defaults should be "current directory, any gb file I can find. If I find more than one, open the newest one. If I find none, error out."
    # TODO: make sure the output indicates what is being done and why. e.g. "No directory specified.  Checking current directory for .gb files. Found 3 files. Using the newest one: PokemonYellow.gb"
    # TODO: be "quiet" when parameters are passed and work as expected, but "chatty" when the parameter is skipped and the application is doing "defaulty" things.
    # TODO: DIRECTORY CLEANUP INCLUDING LOGROTATINON.
    parser.add_argument("--game_path", type=str,
                        default="/home/mscs/PokemonYellow.gb")
    # TODO: fix multi-host model merge.  Can we train across multiple instances of the same cart? Can we train across DIFFERENT pokemon carts?
    # TODO: Expirement: If we can train on DIFFERENT pokemon carts, can we train on multiple GB games at a time and build a generally good base "gameboy game" model for training specific games?

    # TODO: Visual gif of map as it exapnds over time, with frames of the game as it is played, so the map is faded gray in the spot the AI isn't currently at.  Should be updated in frame order.  BIG PROJECT.
    # TODO: 5529600 frames is roughly 10 seconds of gametime (144h * 160w * 24fps * 10) and about 5.2mb of data. 10m of data is about 317MB. Math OK? 144 * 160 * 24 * 60 * 10 / 1024 / 1024

    parser.add_argument("--num_hosts", type=int, default=1)
    args = parser.parse_args()

    num_cpu = multiprocessing.cpu_count()

    hrs = 10  # number of hours (in-game) to run for.
    # hrs = 1 # temporarily shorter duration.
    runsteps = int(3200000 * (hrs))

    # num_cpu = 1
    run_env = None
    if num_cpu == 1:
        run_env = DummyVecEnv([make_env(args.game_path, 0)])
    else:
        run_env = SubprocVecEnv([make_env(args.game_path,
                                      emunum) for emunum in range(num_cpu)])

    model_file_name = "model"

    episodes = 13
    for e in range(0, episodes):
        model = train_model(run_env, runsteps, steps=8192, episode=e, file_name=model_file_name)
        model.save(f"{model_file_name}.zip")
