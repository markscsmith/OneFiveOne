import multiprocessing
from pyboy import PyBoy, WindowEvent
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from timg import Renderer, Ansi24HblockMethod
from PIL import Image, ImageDraw, ImageFont


from os.path import exists
import gymnasium as gym

from gym.spaces import Box, Dict, Discrete
import numpy as np
import base64
import torch
import torch.nn as nn
import time
import os
import datetime
import hashlib
import glob
from tqdm import tqdm

MEM_START = 0xCC3C
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
                 "d9290db87b1f0a23b89f99ee4469e34b":  "POKEMONY.GBC",
                 "50927e843568814f7ed45ec4f944bd8b":  "POKEMONB.GBC",
                 "3e098020b56c807393cc2ebae5e1857a":  "POKEMONS.GBC",
                 "3d45c1ee9abd5738df46d2bdda8b57dc":  "POKEMONR.GBC",}
        if self.checksum in carts:
            return carts[self.checksum]
        else:
            print("Unknown cart:", self.checksum)
            return None

    def cart_offset(self):
        # Pokemon Yellow has offset -1 vs blue and green
        # TODO: Pokemon Gold Silver and Crystal
        carts ={
                "POKEMONR.GBC": MEM_START,
                "POKEMONB.GBC": MEM_START,
                "POKEMONY.GBC": MEM_START,
                "POKEMONG.GBC": MEM_START,
                "PMCRYSTA.GBC": MEM_START,                 
                "POKEMONS.GBC": MEM_START,
                }
        if self.identify_cart() in carts:
            return carts[self.identify_cart()]
        else:
            print("Unknown cart:", self.checksum)
            return 0x00000000


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, **kwargs):
        super(CustomFeatureExtractor, self).__init__(observation_space=observation_space, features_dim=features_dim, **kwargs)

        # Define your custom layers here
        self.extractor = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)


    def generate_gif_and_actions(self):
        actions = self.training_env.get_attr('actions')
        rewards = self.training_env.get_attr('last_score')
        frames = self.training_env.get_attr('frames')
        visiteds = self.training_env.get_attr('visited_xy')
        pokemon_caughts = self.training_env.get_attr('last_pokemon_count')

        hostname = os.uname()[1]

        base_file_name = f"{hostname}-{self.filename_datetime}-{self.model.num_timesteps}"
        # Generate a CSV of data
        with open(f"/Volumes/Scratch/ofo/{base_file_name}.csv", "w", encoding="utf-8") as f:
            f.write("env_num,caught,actions,rewards,frames,visiteds\n")
            for env_num, (action, caught, reward, frame, visited) in enumerate(zip(actions, pokemon_caughts, rewards, frames, visiteds)):
                f.write(f"{env_num},{caught},{''.join(action)},{reward},{frame},\"{visited}\"\n")



    def scan_models(self):
        # Force LS of directory to refresh cacche:
        os.system("ls /Volumes/Scratch/ofo")
        model_files = glob.glob("/Volumes/Scratch/ofo/*-*.zip")

        found_models = []
        for model_file in model_files:
            if int(model_file.split("-")[-1].split(".")[0]) >= (self.model.num_timesteps - self.model.num_timesteps * 0.34):
                found_models.append(model_file)
        return found_models


    def merge_models(self, model_files):
        models = [PPO.load(model_file, device='cpu') for model_file in model_files]

        params = [model.policy.state_dict() for model in models]
        average_params = {key: sum([param[key] for param in params]) / len(params) for key in params[0].keys()}
        self.model.policy.load_state_dict(average_params)
        print("Models merged.")
        return True



def learning_rate_schedule(progress):
    return 0.025
    #return  0.0


class CustomNetwork(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomNetwork, self).__init__(*args, **kwargs)
        self.lr_schedule = learning_rate_schedule

class PokeCaughtCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0 ):
        super(PokeCaughtCallback, self).__init__(verbose)
        self.timg_render = Renderer()
        self.filename_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.progress = 0
        self.total_timesteps = total_timesteps
        self.progress_bar = tqdm(total=self.total_timesteps, position=0, leave=True)

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
            with open(f"/Volumes/Scratch/ofo/{file_name}.csv", "w", encoding="utf-8") as f:
                f.write("env_num,caught,actions,rewards,frames,visiteds\n")
                for env_num, (action, caught, reward, frame, visited) in enumerate(zip(actions, pokemon_caughts, rewards, frames, visiteds)):
                    f.write(f"{env_num},{caught},{''.join(action)},{reward},{frame},\"{visited}\"\n")

    def _on_step(self) -> bool:
        # Retrieve pokemon_caught from each environment
        all_pokemon_caught = self.training_env.get_attr('last_pokemon_count')
        visiteds = self.training_env.get_attr('visited_xy')
        rewards = self.training_env.get_attr('last_score')
        frames = self.training_env.get_attr('frames')
        stationary_frames = self.training_env.get_attr('stationary_frames')
        xs = self.training_env.get_attr('last_player_x')
        ys = self.training_env.get_attr('last_player_y')
        xbs = self.training_env.get_attr('last_player_x_block')
        ybs = self.training_env.get_attr('last_player_y_block')
        map_ids = self.training_env.get_attr('last_player_map')
        actions = self.training_env.get_attr('actions')
        
        # filename_datetimes = self.training_env.get_attr('filename_datetime')

        # all_frames = self.training_env.get_attr('screen_images')


        # for env_num, (rendered_frames, visited, steps, reward, filename_datetime)  in enumerate(zip(all_frames, visiteds, frames, rewards, filename_datetimes)):
        #     print(f"{str(env_num).zfill(3)} 🟣 {all_pokemon_caught[env_num]} 🎬 {steps} 🌎 {len(visited)} 🏆 {reward}")
        #     # Combine frames into gif
        #     rendered_frames[0].save(f"/Volumes/Scratch/frames/{env_num}-{filename_datetime}.gif", save_all=True, append_images=rendered_frames[1:], optimize=False, duration=100, loop=0)
            # for frame_num, frame in enumerate(rendered_frames):
            #     frame.save(f"/Volumes/Scratch/frames/{env_num}-{frame_num}-{filename_datetime}.png")
        # Example: print or process the retrieved values

        # Find the best performing environment
        best_env_idx = np.argmax(rewards)
        # best_env_idx = np.argmax(stationary_frames)


        # for env_num, (pokemon_count, visited, steps, reward)  in enumerate(zip(all_pokemon_caught, visiteds, frames, rewards)):
        # for env_num, (pokemon_count, visited, steps, reward)  in enumerate(zip(all_pokemon_caught, visiteds, frames, rewards)):
        #     print(f"{str(env_num).zfill(3)} 🟣 {pokemon_count} 🎬 {steps} 🌎 {len(visited)} 🏆 {reward}")
        
        self.training_env.env_method('render', best_env_idx)
        self.progress = self.model.num_timesteps
        self.progress_bar.update(self.progress - self.progress_bar.n)

        

        # if terminal_size.columns < 160 or terminal_size.lines < 144 / 2:
        
        # print(f"Best: {best_env_idx} 🟢 {all_pokemon_caught[best_env_idx]} 🎬 {frames[best_env_idx]} 🌎 {len(visiteds[best_env_idx])} 🏆 {rewards[best_env_idx]} 🦶 {stationary_frames[best_env_idx]} X: {xs[best_env_idx]} Y: {ys[best_env_idx]} XB: {xbs[best_env_idx]} YB: {ybs[best_env_idx]}, Map: {map_ids[best_env_idx]} Actinos {actions[best_env_idx][-6:]}")

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
        self.pyboy = PyBoy(game_path, window_type="headless", cgb=True)
        self.game_path = game_path
        self.menu_value =  None
        self.n = 21600
        self.last_n_frames = [None] * self.n
        self.renderer = Renderer()
        self.actions = []
        self.screen_images = []
        self.reset_unlocked = False
        # Define the memory range for 'number of Pokémon caught'
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
        self.max_frames = max_frames
        # self.buttons = [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT,
        #                 WindowEvent.PRESS_ARROW_RIGHT,WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
        #                 WindowEvent.PRESS_BUTTON_START, WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_ARROW_UP,
        #                 WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT,
        #                 WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_START,
        #                 WindowEvent.RELEASE_BUTTON_SELECT]

        self.buttons = [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT,
                        WindowEvent.PRESS_ARROW_RIGHT,WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
                        WindowEvent.PRESS_BUTTON_START, WindowEvent.PASS, WindowEvent.RELEASE_ARROW_UP,
                        WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT,
                        WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_START,
                        WindowEvent.PASS]


        self.buttons_names = "UDLRABS!udlrabs."



        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Format the datetime as a string suitable for a Unix filename
        self.filename_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")



        # Define actioqn_space and observation_space
        # self.action_space = gym.spaces.Discrete(256)
        # self.action_space = gym.spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        size = MEM_END - MEM_START + 2
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size,), dtype=np.uint16)


    def generate_image(self):
        return self.pyboy.botsupport_manager().screen().screen_image()


    def generate_screen_ndarray(self):
        return self.pyboy.botsupport_manager().screen().screen_ndarray()

    def calculate_reward(self, memory_values, button_mash = 0):
        # calculate total bits from the memory values
        pokemon_caught = sum(
             [bin(values).count('1') for values in memory_values[self.caught_pokemon_start: self.caught_pokemon_end]]
        )
        pokemon_seen = sum(
             [bin(values).count('1') for values in memory_values[self.seen_pokemmon_start: self.seen_pokemmon_end]]
        )
        
        self.last_seen_pokemon_count = pokemon_seen

        px = memory_values[self.player_x_mem]
        py = memory_values[self.player_y_mem]
        pbx = memory_values[self.player_x_block_mem]
        pby = memory_values[self.player_y_block_mem]
        map_id = memory_values[self.player_map_mem]
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
            reward = len(self.player_maps) * 1000 + len(self.visited_xy) // 10
        else:
            reward = pokemon_caught * 10000  + pokemon_seen * 5000 + len(self.player_maps) * 1000 + len(self.visited_xy) // 10
        
        # reduce the reward by the % of frames the player has been stationary, allowing for longer events later in the game
        # reward = reward - int(reward * (self.stationary_frames / (self.frames + 1)))  * 10
        # if reward < -50000:
        #     self.reset()
        # elif reward < 0:
        #     reward = 0

        return reward


    def render(self, target_index = None, reset = False):
        if target_index is not None and target_index == self.emunum or reset:
            terminal_size = os.get_terminal_size()
            terminal_offset = 4
            
                
            image = Image.fromarray(self.pyboy.botsupport_manager().screen().screen_ndarray())
            w = 160
            h = 144
            if terminal_size.columns != w or terminal_size.lines < h / 2:
                image_aspect_ratio = w / h
                terminal_aspect_ratio = terminal_size.columns / (terminal_size.lines - terminal_offset)

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
            self.renderer.resize(terminal_size.columns, terminal_size.lines * 2 - terminal_offset)
            self.renderer.render(Ansi24HblockMethod)
            if target_index is not None:
                print("Best:  {} 🟢 {} 👀 {} 🎬 {} 🌎 {} 🏆 {} 🦶 {} X: {} Y: {} XB: {} YB: {}, Map: {} Actinos {} {}".format(target_index, self.last_pokemon_count, self.last_seen_pokemon_count, self.frames, len(self.visited_xy), self.last_score, self.stationary_frames, self.last_player_x, self.last_player_y, self.last_player_x_block, self.last_player_y_block, self.last_player_map, ":".join(self.actions[-6:]), len(self.actions)))
            if reset:
                print("Reset: {} 🟢 {} 👀 {} 🎬 {} 🌎 {} 🏆 {} 🦶 {} X: {} Y: {} XB: {} YB: {}, Map: {} Actinos {} {}".format(self.emunum, self.last_pokemon_count, self.last_seen_pokemon_count, self.frames, len(self.visited_xy), self.last_score, self.stationary_frames, self.last_player_x, self.last_player_y, self.last_player_x_block, self.last_player_y_block, self.last_player_map, ":".join(self.actions[-6:]), len(self.actions)))
            

    def step(self, action):
        self.frames = self.pyboy.frame_count
        ticks = 24
        frame_checks = 24
        # rethink the action space as the binary state of all 8 buttons:
        # U = UP, D = DOWN, L = LEFT, R = RIGHT, A = A, B = B, * = "STAR"T, > = SELECT
        #           """  U  D  L  R  A  B  *  > """
        button_states = [0, 0, 0, 0, 0, 0, 0, 0]
        for i, _ in enumerate(button_states):
            if action[i] > 0.75:
                button_states[i] = 1
        button_states_raw = "".join(str(i) for i in button_states)
        button_mash = 0
        # count how many arrow buttons are pushed by checking the first 4 bits




        for i, state in enumerate(button_states_raw):
            if int(state) > 0:
                button_states[i] = 1
        # Fixed select button to 0 to prevent hard resets
        duration = 0
        # duration_bits = action[8:]
        # if duration_bits[0] > 0.5:
        #     duration += 8
        # if duration_bits[1] > 0.5:
        #     duration += 4
        # if duration_bits[2] > 0.5:
        #     duration += 2
        # if duration_bits[3] > 0.5:
        #     duration += 1
        temp_reset_penalty = 0
        if self.reset_unlocked:
            pass
        else:
            button_states[7] = 0




        # ticks += duration
        # self.pyboy.send_input(self.buttons[action])
        # self.actions.append(self.buttons_names[action])
        for i, state in enumerate(button_states):
            if state:
                self.pyboy.send_input(self.buttons[i])
            else:
                self.pyboy.send_input(self.buttons[i + 8])
        self.actions.append(button_states_raw + ":" + str(duration))
        for _ in range(ticks):
            self.pyboy.tick()
        # Grab less frames to append if we're standing still.

        # screen_bytes = self.generate_screen_ndarray().tobytes()
        # if screen_bytes not in self.last_n_frames:
        #     self.last_n_frames.append(screen_bytes)
        #     self.last_n_frames.pop(0)
        #     self.screen_image_arrays.add(screen_bytes)
        #     self.unchanged_frames -= 1
        # else:
        #     self.unchanged_frames += 1



        # flo = io.BytesIO()
        # self.pyboy.save_state(flo)
        # flo.seek(0)
        # memory_values = np.frombuffer(flo.read(), dtype=np.uint8)

        memory_values = self.get_memory_range()
        reward = self.calculate_reward(memory_values=memory_values, button_mash = button_mash)
        

        # Don't count the frames where the player is still in the starting menus. Pokemon caught gives more leeway on standing still
                                                                                                       # Minutes standing still * 10

        self.last_score = reward

        # if np.random.randint(777) == 0 or self.last_pokemon_count != pokemon_caught or self.last_score - reward > 100:
        #     self.render()
        #
        truncated = False
        if self.frames >= self.max_frames:
            terminated = True
        else:
            terminated = False
        if reward < -10000 or (self.stationary_frames > 10000 > self.frames / 5) and self.frames > 10000:
            self.reset_unlocked = True
            # self.pyboy.load_state(open(self.save_state_path, "rb"))
        else:
            truncated = False
            self.reset_unlocked = False
        if reward < 0:
            reward = 0

        info = {}
        observation = np.append(memory_values, reward)
        return observation, reward, terminated, truncated, info

    import numba

    @numba.jit(forceobj=True)
    def get_memory_range(self):    
        memory_values = [self.pyboy.get_memory_value(i) for i in range(MEM_START, MEM_END + 1)]
        return memory_values

    def reset(self, seed=0, **kwargs):
        self.stationary_frames = 0
        self.unchanged_frames = 0
        hard_reset = False
        if self.reset_penalty < -5000:
            hard_reset = True
            self.actions += ["HARD RESET"]
            self.render(reset=True)
        else:
            self.actions += ["RESET"]

        # print("OS:RESET:", self.emunum, seed)
        super().reset(seed=seed, **kwargs)

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
        self.last_n_frames = [None] * self.n
        self.pyboy = PyBoy(self.game_path, window_type="headless", cgb=True)


        if self.save_state_path is not None:
            self.pyboy.load_state(open(self.save_state_path, "rb"))
        else:
            print(f"Error: No state file found for {self.save_state_path}")
            exit(1)
        print("OS:SHAPE:", self.observation_space.shape)
        # memory_values = self.pyboy.mb.ram.internal_ram0.append(self.pyboy.mb.ram.internal_ram1)
        # memory_values = np.array([self.pyboy.get_memory_value(address) for address in range(0x10000)])
        memory_values = self.get_memory_range()
        # pokemon_caught = sum(
        #     memory_values[self.caught_pokemon_start: self.caught_pokemon_end]
        # )
        # unique_events = memory_values[0xD5A6:0xD85F]
        # hashable_strings = "".join(
        #     [base64.b64encode(bytes(chunk)).decode("utf-8")
        #      for chunk in unique_events]
        # )
        self.actions = []
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
        


        # if hashable_strings not in self.seen_events:
        #     self.seen_events.add(hashable_strings)
        #     # print("OS:EVENTS:", hashable_strings)

        reward = self.calculate_reward(memory_values=memory_values)
        observation = np.append(
            memory_values, reward)
        # use timg to output the current screen

        print("OS:SHAPE:", observation.shape, seed)
        return observation, {"seed": seed}


def make_env(game_path, emunum):
    def _init():
        if os.path.exists(game_path + ".state"):
            print(f"Loading state {game_path}.state")
            new_env = PyBoyEnv(game_path, emunum=emunum, save_state_path=game_path + ".state")
            new_env.pyboy.load_state(open(game_path + ".state", "rb"))
        else:
            print(f"Error: No state file found for {game_path}.state")
            exit(1)
            new_env = PyBoyEnv(game_path, emunum=emunum, save_state_path=None)

        new_env.pyboy.set_emulation_speed(0)
        return new_env
    return _init


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_path", type=str, default="/home/mscs/PokemonYellow.gb")
    parser.add_argument("--num_hosts", type=int, default=1)
    args = parser.parse_args()

    num_cpu = multiprocessing.cpu_count()
    
    hrs = 10 # number of hours to run for.
    runsteps = int(5000  * (hrs) * num_cpu)
    # num_cpu = 1
    # Hostname and timestamp



    # num_cpu = 1
    if num_cpu == 1:
        env =  DummyVecEnv([make_env(args.game_path, 0)])
    else:
        env = SubprocVecEnv([make_env(args.game_path,
                                  emunum) for emunum in range(num_cpu)])


    file_name = "model"
    def train_model(env, num_steps, steps, episodes):
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs={},
            net_arch=dict(pi=[256, 128, 32], vf=[256, 128, 32]),
            activation_fn=nn.ReLU,
        )

        device = "cpu"
        device = (
            "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else device
        )
        device = "cuda" if torch.cuda.is_available() else device

        if exists(file_name + '.zip'):
            print('\nloading checkpoint')
            run_model = PPO.load(file_name, env=env, device=device)
            run_model.n_steps = steps * num_cpu
            run_model.n_envs = num_cpu
            run_model.device = device
            run_model.rollout_buffer.n_envs = num_cpu
            run_model.rollout_buffer.reset()

        else:
            n_steps = steps * num_cpu

            run_model = PPO(policy="CnnPolicy", n_steps=n_steps, batch_size=n_steps * num_cpu,  n_epochs=13, gamma=0.92, gae_lambda=0.92, learning_rate=0.0025, env=env, policy_kwargs=policy_kwargs, verbose=1, device=device, ent_coef=0.9, vf_coef=0.99,)
        # model_merge_callback = EveryNTimesteps(n_steps=steps * num_cpu * 1024, callback=ModelMergeCallback(args.num_hosts))
        # TODO: Progress callback that collects data from each frame for stats
        
        for _ in range(0, episodes):
            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=f"/Volumes/Scratch/ofo_chkpt/{os.uname()[1]}-{time.time()}.zip", name_prefix="poke")
            current_stats = EveryNTimesteps(n_steps=3000, callback=PokeCaughtCallback(runsteps))
            callbacks = [checkpoint_callback, current_stats]
            run_model.learn(total_timesteps=num_steps, progress_bar=False, callback=callbacks)
        return run_model

    model = train_model(env, runsteps, steps=1024, episodes=13)
    model.save(f"{file_name}.zip")
