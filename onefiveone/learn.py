import sys
import time
import os
import datetime
import hashlib
import multiprocessing

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
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv

# Emulator libs
from pyboy import PyBoy

# Output libs
from timg import Renderer, Ansi24HblockMethod
from PIL import Image, ImageDraw, ImageFont
import glob

# Memory ranges to read in Pokemon Red/Blue (+ Yellow?)
# MEM_START = 0xCC3C
MEM_START = 0xD2F7
MEM_END = 0xDEE1
SPRITE_MAP_START = 0xC100
SPRITE_MAP_END = 0xC2FF

FRAME_BUFFER_SIZE = 3600

LOG_FREQ = 1000

PRESS_FRAMES = 10
RELEASE_FRAMES = 20

CGB = False
NUM_CPU = multiprocessing.cpu_count()


class PokeCart:
    def __init__(self, cart_data) -> None:
        # calculate checksum of cart_data
        self.cart_data = cart_data
        self.offset = None
        self.checksum = hashlib.md5(cart_data).hexdigest()
        self.carts = {
            "a6924ce1f9ad2228e1c6580779b23878": ("POKEMONG.GBC", 0),
            "9f2922b235a5eeb78d65594e82ef5dde": ("PMCRYSTA.GBC", 0),
            # TODO: Add Pokemon Yellow logic to keep Pikachu happy. 🌩️🐭  Address is 0xD46F    1    Pikachu's happiness per https://datacrystal.tcrf.net/wiki/Pokémon_Yellow/RAM_map
            "d9290db87b1f0a23b89f99ee4469e34b": ("POKEMONY.GBC", -1),
            "50927e843568814f7ed45ec4f944bd8b": ("POKEMONB.GBC", 0),
            "3e098020b56c807393cc2ebae5e1857a": ("POKEMONS.GBC", 0),
            "3d45c1ee9abd5738df46d2bdda8b57dc": ("POKEMONR.GBC", 0),
        }

    def identify_cart(self):
        # identify cart

        if self.checksum in self.carts:
            print(f"Identified cart: {self.carts[self.checksum]} with offset {self.carts[self.checksum][1]}")
            return self.carts[self.checksum][0]
        else:
            print(f"Unknown cart: {self.checksum}")
            return None

    def cart_offset(self):
        # Pokemon Yellow has offset -1 vs blue and green
        # TODO: Pokemon Gold Silver and Crystal
        if self.offset is not None:
            return self.offset
        elif self.identify_cart() is not None:
            self.offset = self.carts[self.checksum][1]
            return self.offset
        print("Unknown cart:", self.checksum)
        self.offset = 0
        return self.offset

def diff_flags(s1, s2):
    return [i for i, (c1, c2) in enumerate(zip(s1, s2)) if c1 != c2]

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
    return  0.0003 * (1 - progress)

class CustomNetwork(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomNetwork, self).__init__(*args, **kwargs)
        self.lr_schedule = learning_rate_schedule


class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardLoggingCallback, self).__init__(verbose)
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
            max_item_points = 0
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
                    self.logger.record(f'badges/{emunum}', f"{badges}")
                    self.logger.record(f"reward/{emunum}", f"{reward}")
                    
                    self.logger.record(
                        f"visited/{emunum}", f"{len(info['visited_xy'])}"
                    )
                    # self.logger.record(f"items/{emunum}", f"{info['items']}")
                    # max_item_points = max(max_item_points, sum(info["items"].values()))
                    self.logger.record(f"pokedex/{emunum}", f"{pokedex}")
                    self.logger.record(f"seen_and_capture/{emunum}", f"{seen_and_capture_events}")

            # todo: record each progress/reward separately like I do the actions?
            

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
                # self.logger.record("reward/max_items", max_item_points)

        return True  # Returning True means we will continue training, returning False will stop training


class PokeCaughtCallback(BaseCallback):
    def __init__(self, total_timesteps, multiplier=1, verbose=0):
        super(PokeCaughtCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.timg_render = Renderer()
        self.filename_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.multiplier = multiplier
        self.step_count = 0
        # self.progress_bar = tqdm(
        #     total=total_timesteps, desc="Frames", leave=False, dynamic_ncols=True
        # )

    def _on_step(self) -> bool:
        rewards = self.training_env.get_attr("last_score")
        self.step_count += 1
        
        best_env_idx = rewards.index(max(rewards))
        print(self.training_env.env_method("render", best_env_idx)[
                best_env_idx
            ])
        # self.progress_bar.update(self.multiplier)

        # render_string = self.training_env.env_method("render", best_env_idx)[
        #     best_env_idx
        # ]

        # # render_string = self.training_env.env_method("render", best_env_idx)
        # # sys.stdout.write(render_string)
        # print(render_string)
        # self.progress_bar.update(self.multiplier)

        # if self.progress_bar.n >= self.total_timesteps:
        #     self.progress_bar.close()

        # self.progress = self.model.num_timesteps
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
    def __init__(
        self,
        game_path,
        emunum,
        save_state_path=None,
        device="cpu",
        episode=0,
        **kwargs,
    ):
        super(PyBoyEnv, self).__init__()
        self.pyboy = PyBoy(game_path, window="null", cgb=CGB)
        self.game_path = game_path
        self.menu_value = None
        self.n = 8 # number of frames to store
        # self.last_n_frames = [self.pyboy.memory[SPRITE_MAP_START:SPRITE_MAP_END].copy() for _ in range(self.n)]
        # self.last_n_frames = [self.pyboy.memory[MEM_START:MEM_END].copy() for _ in range(self.n)]
        self.screen_image = np.copy(self.pyboy.screen.ndarray)
        self.last_n_frames = [self.screen_image] * self.n
        self.renderer = Renderer()
        
        self.actions = ""
        self.reset_unlocked = False
        # Define the memory range for 'number of Pokémon caught'
        self.cart = PokeCart(open(game_path, "rb").read())
        offset = self.cart.cart_offset()
        # End needs to have +8 to include the last byte
        self.caught_pokemon_start = 0xD2F7 + offset
        self.caught_pokemon_end = 0xD309 + 1 + offset
        self.seen_pokemon_start = 0xD30A + offset
        self.seen_pokemon_end = 0xD31C + 1 + offset
        self.player_x_mem = 0xD361 + offset
        self.player_y_mem = 0xD362 + offset
        self.player_x_block_mem = 0xD363 + offset
        self.player_y_block_mem = 0xD364 + offset
        self.player_map_mem = 0xD35E + offset
        self.party_exp_reward = 0
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
        self.my_pokemon = None
        self.step_count = 0
        self.backtrack_reward = 0
        self.last_chunk_id = None
        self.screen_image = None

        self.recent_frames = []
        
        
        self.badges = 0
        
        
        
        self.player_maps = set()
        self.pokedex = "-" * 151
        self.seen_pokedex = []
        self.caught_pokedex = []
        self.last_carried_item_total = 0
        self.last_stored_item_total = 0
        self.device = device
        self.episode = episode
        self.seen_and_capture_events = {}
        self.travel_reward = 0


        self.last_memory_update_frame = 0
        self.current_memory = None

        self.party_exp = [0, 0, 0, 0, 0, 0]
        # self.buttons = {
        #     0: (utils.WindowEvent.PASS, "-"),
        #     1: (utils.WindowEvent.PRESS_ARROW_UP, "U"),
        #     2: (utils.WindowEvent.PRESS_ARROW_DOWN, "D"),
        #     3: (utils.WindowEvent.PRESS_ARROW_LEFT, "L"),
        #     4: (utils.WindowEvent.PRESS_ARROW_RIGHT, "R"),
        #     5: (utils.WindowEvent.PRESS_BUTTON_A, "A"),
        #     6: (utils.WindowEvent.PRESS_BUTTON_B, "B"),
        #     7: (utils.WindowEvent.PRESS_BUTTON_START, "S"),
        #     8: (utils.WindowEvent.PASS, "."),
        #     9: (utils.WindowEvent.RELEASE_ARROW_UP, "u"),
        #     10: (utils.WindowEvent.RELEASE_ARROW_DOWN, "d"),
        #     11: (utils.WindowEvent.RELEASE_ARROW_LEFT, "l"),
        #     12: (utils.WindowEvent.RELEASE_ARROW_RIGHT, "r"),
        #     13: (utils.WindowEvent.RELEASE_BUTTON_A, "a"),
        #     14: (utils.WindowEvent.RELEASE_BUTTON_B, "b"),
        #     15: (utils.WindowEvent.RELEASE_BUTTON_START, "s"),
        # }

        self.buttons = {
            0: ("", "-"),
            1: ("up", "U"),
            2: ("down", "D"),
            3: ("left", "L"),
            4: ("right", "R"),
            5: ("a", "A"),
            6: ("b", "B"),
            7: ("start", "S"),
        }

        self.buttons_names = "UDLRABS!_udlrabs.-"

        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Format the datetime as a string suitable for a Unix filename
        self.filename_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        # size = (self.n * 359) + 1
        # size = MEM_END - MEM_START + 1
        size = 1327

        # Define actioqn_space and observation_space
        # self.action_space = gym.spaces.Discrete(256)
        # self.action_space = gym.spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        

        # self.observation_space = Box(low=0, high=255, shape=(size,), dtype=np.float32)
        # use screen as input


        
        # single frame
        # self.observation_space = Box(low=0, high=255, shape=(144,160,4), dtype=np.uint8)
        # multiple frames

        self.observation_space = Box(low=0, high=255, shape=(144,160,4*self.n), dtype=np.uint8)

        self.action_space = Discrete(8, start=0)
        # size = SPRITE_MAP_END - SPRITE_MAP_START + 1

        # size = MEM_START MEM_END + 2



    def get_mem_block(self, offset):
        pokemart = [] # self.pyboy.memory[0xCF7B + offset:0xCF85 + offset + 1]
        my_pokemon = self.pyboy.memory[0xD16B + offset:0xD272 + offset + 1]
        pokedex = self.pyboy.memory[0xD2F7 + offset:0xD31C + offset + 1]
        items = [] # self.pyboy.memory[0xD31D + offset:0xD346 + offset + 1]
        money = [] # self.pyboy.memory[0xD347 + offset:0xD349 + offset + 1]
        badges = [self.pyboy.memory[0xD356 + offset]]
        location = self.pyboy.memory[0xD35E+offset:0xD365+offset + 1]
        stored_items = []#  self.pyboy.memory[0xD53A + offset:0xD59F + offset + 1]
        coins = [] # self.pyboy.memory[0xD5A4 + offset: 0xD5A5 + offset + 1]
        missable_object_flags = [] # self.pyboy.memory[0xD5A6 + offset: 0xD5C5 + offset + 1]
        event_flags = [] # self.pyboy.memory[0xD72E + offset: 0xD7EE + offset + 1]
        ss_anne = [] # [self.pyboy.memory[0xD803] + offset]
        mewtwo = [] # [self.pyboy.memory[0xD85F] + offset]
        opponent_pokemon = [] # self.pyboy.memory[0xD8A4 + offset: 0xD9AB + offset + 1]

        return  [pokemart ,
            my_pokemon ,
            pokedex ,
            items ,
            money ,
            badges ,
            location ,
            stored_items ,
            coins ,
            missable_object_flags ,
            event_flags ,
            ss_anne ,
            mewtwo ,
            opponent_pokemon ,]

    def get_screen_tiles(self):
        return self.pyboy.memory[SPRITE_MAP_START:SPRITE_MAP_END].copy()


    def generate_image(self):
        return self.pyboy.screen.ndarray

    def generate_screen_ndarray(self):
        return self.pyboy.screen.ndarray
    
    
    def get_pokedex_status_string(self, data_seen, data_owned):
        def get_status(data, poke_num):
            byte_index = poke_num // 8
            bit_index = poke_num % 8
            return (data[byte_index] >> bit_index) & 1
        
        status_string = ""
        for poke_num in range(151):
            seen = get_status(data_seen, poke_num)
            owned = get_status(data_owned, poke_num)
            if owned and seen:
                status_string += 'O'
            elif seen:
                status_string += 'S'
            elif owned:
                status_string += '?'
            else:
                status_string += '-'
        return status_string
        

    def calculate_reward(self):

        offset = self.cart.cart_offset()  # + MEM_START
        mem_block = self.get_mem_block(offset)
        reward = 0


        travel_reward = self.travel_reward

        pokemart, my_pokemon, pokedex, items, money, badges, location, stored_items, coins, missable_object_flags, event_flags, ss_anne, mewtwo, opponent_pokemon = mem_block
        

        map_id = location[0]
        
        px = location[1]
        py = location[2]
        pbx = location[3]
        pby = location[4]
        
        
        self.my_pokemon = my_pokemon

        caught_pokemon_start = self.caught_pokemon_start
        caught_pokemon_end = self.caught_pokemon_end
        seen_pokemon_start = self.seen_pokemon_start
        seen_pokemon_end = self.seen_pokemon_end


        # Calculate badge reward total
        badge_count = sum([bin(badge).count("1") for badge in badges])
        self.badges = badge_count
        badge_reward = badge_count * 10

        # Calculate reward from flags / missable objects / events

        # Calculate reward from exploring the game world by counting maps, doesn't need to store counter
        if self.last_player_map != map_id:
            if map_id not in self.player_maps:
                self.player_maps.add(map_id)

        chunk_id = f"{px}:{py}:{pbx}:{pby}:{map_id}"
        
        visited_score = 0
        if chunk_id not in self.visited_xy and self.last_chunk_id != chunk_id:
            self.visited_xy.add(chunk_id)
            visited_score = 0.100
        self.last_chunk_id = chunk_id

        travel_reward += visited_score


        # convert binary chunks into a single string
        
        full_dex = pokedex
        caught_pokedex = list(full_dex[:caught_pokemon_end - caught_pokemon_start])
        seen_pokedex = list(full_dex[seen_pokemon_start - caught_pokemon_start:])
        self.seen_pokedex = seen_pokedex
        self.caught_pokedex = caught_pokedex
        last_dex = self.pokedex
        new_dex = self.get_pokedex_status_string(seen_pokedex, caught_pokedex)

        # compare the last pokedex to the current pokedex
        if last_dex != new_dex:
            poke_nums = diff_flags(last_dex, new_dex)
            poke_pairs = zip(poke_nums, [new_dex[p] for p in poke_nums])
            self.seen_and_capture_events[self.pyboy.frame_count] = list(poke_pairs)
            self.visited_xy = set()


        self.pokedex = new_dex

        pokemon_owned =  self.pokedex.count("O")
        pokemon_seen = self.pokedex.count("S") + pokemon_owned

        last_poke = self.last_pokemon_count
        last_poke_seen = self.last_seen_pokemon_count

        if pokemon_seen == 0:
            pokemon_owned = 0

        if pokemon_owned > last_poke:
            self.seen_and_capture_events[self.pyboy.frame_count] = (pokemon_owned, pokemon_seen)
            self.last_pokemon_count = pokemon_owned
            

        if pokemon_seen > last_poke_seen:
            self.last_seen_pokemon_count = pokemon_seen

        self.last_pokemon_count = pokemon_owned
        self.last_seen_pokemon_count = pokemon_seen
        # [84, 0, 19, 0, 0, 23, 23, 163, 84, 45, 0, 0, 133, 139, 0, 0, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 72, 30, 40, 0, 0, 5, 0, 19, 0, 11, 0, 8, 0, 14, 0, 10
        #   1  2   3  4  5   6   7    8   9  10 11 12   13   14 15 16   17 18 19 20 21 22 23 24 25 26 27  28  29  30  31 32 33 34 35  36 37  38 39 40 41  42 43  44
        #      |hp  | x  x   T   T    H  m1  m2 m3 m4 |  tid  |  exp | a | d | spd | spc|     
        party = [my_pokemon[0:44],
                 my_pokemon[44:88],
                 my_pokemon[88:132],
                 my_pokemon[132:176],
                 my_pokemon[176:220],
                 my_pokemon[220:264],]
        
        # level = 32 in per pokemon
        poke_levels = [poke[33] for poke in party]
        party_exp_reward = self.party_exp_reward
        party_exp = poke_levels
        # for poke in party:
        #     upper = int(poke[14]) << 8
        #     lower = int(poke[15])
        #     exp = upper + lower
        #     party_exp.append(exp)

        if sum(party_exp) > sum(self.party_exp):
            party_exp_reward += (sum(party_exp) - sum(self.party_exp))
            # self.render()
            #print("Party EXP:", party_exp, self.party_exp, party_exp_reward)
        self.party_exp = party_exp
        reward = (
            len(self.player_maps) + ((pokemon_owned * 2) * 100 + pokemon_seen * 100)
        )  + badge_reward + party_exp_reward + travel_reward
        self.party_exp_reward = party_exp_reward
        self.travel_reward = travel_reward
        
        # reward -= (reward * (self.stationary_frames / (self.frames + 1)))

        self.last_player_x = px
        self.last_player_y = py
        self.last_player_x_block = pbx
        self.last_player_y_block = pby
        self.last_player_map = map_id

        return round(reward, 4), mem_block

    # TODO: Refactor so returns image instead of immediately rendering so PokeCaughtCallback can render instead.
    def render(self, target_index=None, reset=False):
        if target_index is not None and target_index == self.emunum or reset:
            terminal_size = os.get_terminal_size()
            terminal_offset = 7

            image = self.pyboy.screen.image
            w = 160
            h = 144
            memories = self.last_n_frames
            # convert list of ndarrays into a single ndarray
            
            # convert memories into an image
            
            new_image = Image.new("RGB", (image.width, image.height + image.height // 2))
            for i, memory in enumerate(memories):
                # shrink image to 1/4 size
                memory = Image.fromarray(memory)
                memory = memory.resize((w // 4, h // 4))
                if i < 4:
                    new_image.paste(memory, ((i * w // 4), h))
                else:
                    new_image.paste(memory, (((i - 4) * w // 4), h + h // 4))
            

            new_image.paste(image, (0, 0))
            image = new_image
            
            # if terminal_size.columns != w or terminal_size.lines < h / 2:
            #     image_aspect_ratio = image.width / image.height
            #     terminal_aspect_ratio = terminal_size.columns / (
            #         terminal_size.lines - terminal_offset
            #     )

            #     if image_aspect_ratio > terminal_aspect_ratio:
            #         new_width = int(image.width / image_aspect_ratio)
            #     elif image_aspect_ratio < terminal_aspect_ratio:
            #         new_width = int(image.width * image_aspect_ratio)
            #     else:
            #         new_width = image.width

            #     height_offset = new_width - image.width
            #     new_height = image.height + height_offset
            #     replacer = Image.new("RGB", (new_width, new_height), (0, 0, 0))
            #     # in center of image
            #     replacer.paste(
            #         image, ((new_width - image.width) // 2, height_offset // 2)
            #     )
            #     image = replacer
            # create new larger image with memories along the bottom at the same width as the original image maintaining aspect ratio
            

            self.renderer.load_image(image)
            self.renderer.resize(
                terminal_size.columns, terminal_size.lines * 2 - terminal_offset
            )

            fc = self.pyboy.frame_count
            game_seconds = fc // 60
            game_minutes = game_seconds // 60
            game_hours = game_minutes // 60
            # use the proper clock face for the hours:
            clock_faces = "🕛🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚"
            game_time_string = f"{clock_faces[game_hours % 12]} {game_hours:02d}:{game_minutes % 60:02d}:{game_seconds % 60:02d}"
            image_string = self.renderer.to_string(Ansi24HblockMethod)
            if target_index is not None:
                render_string = f"{image_string}🧳 {self.episode} 🧠: {target_index:2d} 🥾 {self.step_count:10d} 🟢 {self.last_pokemon_count:3d} 👀 {self.last_seen_pokemon_count:3d} 🎉 {self.party_exp} 🌎 {len(self.visited_xy):3d}:{len(self.player_maps):3d} 🏆 {self.last_score:7.2f} 💪 {self.party_exp_reward} \n[{self.last_player_x:3d},{self.last_player_y:3d},{self.last_player_x_block:3d},{self.last_player_y_block:3d}], 🗺️: {self.last_player_map:3d} Actions {' '.join(self.actions[-6:])} 🎬 {self.frames:6d} {game_time_string} {len(self.actions)}"
            else:
                render_string = f"{image_string}🧳 {self.episode} 🛠️: {self.emunum:2d} 🥾 {self.step_count:10d} 🟢 {self.last_pokemon_count:3d} 👀 {self.last_seen_pokemon_count:3d} 🎉 {self.party_exp} 🌎 {len(self.visited_xy):3d}:{len(self.player_maps):3d} 🏆 {self.last_score:7.2f} 💪 {self.party_exp_reward} \n[{self.last_player_x:3d},{self.last_player_y:3d},{self.last_player_x_block:3d},{self.last_player_y_block:3d}], 🗺️: {self.last_player_map:3d} Actions {' '.join(self.actions[-6:])} 🎬 {self.frames:6d} {len(self.actions)}"

            return render_string
#🧠: 19 🟢  64 👀  64 🌎  27:  4 🏆 19270.00 🎒   1 🐆   20.00
    # TODO: build expanding pixel map to show extents of game travelled. (minimap?) Use 3d numpy array to store visited pixels. performance?

    def step(self, action):
        self.step_count += 1
        self.frames = self.pyboy.frame_count

        button = self.buttons[action]
        if action != 0:
            self.pyboy.button_press(button[0])
        for _ in range(PRESS_FRAMES):
            self.pyboy.tick()
        if action != 0:
            self.pyboy.button_release(button[0])
        for _ in range(RELEASE_FRAMES):
            self.pyboy.tick()
        self.screen_image = np.copy(self.pyboy.screen.ndarray)
        # .5 seconds = 1 step
        # 5 seconds = 10 steps
        # 10 seconds = 20 steps
        # 30 seconds = 60 steps
        # 60 seconds = 120 steps
        n = self.n
        stm = 4

        
        i = 4
        if self.step_count % 120 == 0:
            i += 3
        elif self.step_count % 60 == 0:
            i += 2
        elif self.step_count % 30 == 0:
            i+= 1
              
        # 0 1 2 3 4 5 6 7
        # 0 1 2 3 = 4 5 6 7
        self.last_n_frames[:-(n-i)] = self.last_n_frames[1:i+1]
        self.last_n_frames[-1] = self.screen_image

        # if it's the same button it's held.  If it's a different button it's a different button.
        # In theory this means it'll figure out how to hold buttons down and how to not
        # press buttons when it's not useful to do so
        self.actions = f"{self.actions}{button[1]}"
        # self.actions = self.actions + (f"{button_name_1}")
        # Grab less frames to append if we're standing still.

        
        sprites = self.get_screen_tiles()
        reward , mem_block = self.calculate_reward()
        self.last_score = reward

        truncated = False
        terminated = False
        

        info = {
            "reward": reward,
            "actions": self.actions,
            "emunum": self.emunum,
            "frames": self.frames,
            "pokemon_caught": self.last_pokemon_count,
            "pokemon_seen": self.last_seen_pokemon_count,
            "visited_xy": self.visited_xy,
            "pokedex": self.pokedex,
            "seen_and_capture_events": self.seen_and_capture_events,
            "badges": self.badges
        }
        # screen = self.pyboy.memory[MEM_START:MEM_END].copy()
        # self.last_n_frames[:-1] = self.last_n_frames[1:]
        # self.last_n_frames[-1] = screen
        mem_block.append(sprites)
        # flat_mem_block = [item for sublist in mem_block for item in sublist]
        # observation = np.append(flat_mem_block, reward)

        # convert observation into float32s
        # if self.device == "mps":
        # observation = observation.astype(np.float32)
        # else:
        #     observation = observation.astype(np.float64)
        observation = np.concatenate(self.last_n_frames, axis=-1)

        return observation, reward, terminated, truncated, info
    
    def set_episode(self, episode):
        self.episode = episode

    def reset(self, seed=0, **kwargs):
        
        super().reset(seed=seed, **kwargs)
        self.last_memory_update_frame = 0
        
        self.travel_reward = 0
        self.last_chunk_id = None
        
        self.party_exp_reward = 0
        self.party_exp = [0, 0, 0, 0, 0, 0]
        self.step_count = 0
        self.visited_xy = set()
        self.player_maps = set()
        self.last_player_x = 0
        self.last_player_y = 0
        self.last_player_x_block = 0
        self.last_player_y_block = 0
        self.menu_value = 0
        self.pokedex = "-" * 151
        self.pyboy = PyBoy(
            self.game_path,
            window="null",
            cgb=CGB,
        )

        self.screen_image = np.copy(self.pyboy.screen.ndarray)

        self.last_n_frames = [self.screen_image] * self.n

        if self.save_state_path is not None:
            self.pyboy.load_state(open(self.save_state_path, "rb"))
        else:
            print(
                f"Error: No state file found for {self.save_state_path}",
                file=sys.stderr,
            )

        self.actions = ""
        self.visited_xy = set()
        self.last_score = 0
        self.last_pokemon_count = 0
        self.frames = 0
        self.last_player_x = 0
        self.last_player_y = 0
        self.last_player_x_block = 0
        self.last_player_y_block = 0
    
        # self.last_n_frames = [self.pyboy.memory[MEM_START:MEM_END].copy() for _ in range(self.n)]
        # screen = self.pyboy.memory[MEM_START:MEM_END].copy()
        # observation = np.append(screen, reward)
        sprites = self.get_screen_tiles()
        reward, mem_block = self.calculate_reward()
        mem_block.append(sprites)
        # flat_mem_block = [item for sublist in mem_block for item in sublist]
        # observation = np.append(flat_mem_block, reward)
        # observation = observation.astype(np.float32)
        observation = np.concatenate(self.last_n_frames, axis=-1)

        # convert observation into float32s
        # if self.device == "mps":
        # observation = observation.astype(np.uint8)
        # else:
        #  observation = observation.astype(np.float64)
        # if self.device == "mps":
        #     observation = observation.astype(np.float32)
        # else:
        #     observation = observation.astype(np.float64)
        # print("RESET:OS:SHAPE:", observation.size, seed, file=sys.stderr)
        return observation, {"seed": seed}


def make_env(game_path, emunum, max_frames=500_000, device="cpu"):
    def _init():
        if os.path.exists(game_path + ".state"):
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
            print(f"Error: No state file found for {game_path}.state")
            exit(1)

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
    
    # first_layer_size = (24 * 359) + 1
    first_layer_size = 1024
    policy_kwargs = dict(
        # features_extractor_class=CustomFeatureExtractor,
        # features_extractor_kwargs={},
        net_arch=dict(
            pi=[first_layer_size, first_layer_size, first_layer_size],
            vf=[first_layer_size, first_layer_size, first_layer_size],
        ),
        
        
    )
    
    # make sure we take care of accidental trailing slashes in the save path which
    # would cause the checkpoint path to be incorrect.
    checkpoint_path = f"{save_path.rstrip('/')}_chkpt"
    env.set_attr("episode", 0)
    tensorboard_log = f"{save_path}/tensorboard/{os.uname()[1]}-{time.time()}"

    run_model = PPO(
        policy="CnnPolicy",
        # Reduce n_steps if too large; ensure not less than some minimum like 2048 for sufficient learning per update.
        n_steps=n_steps,
        # Reduce batch size if it's too large but ensure a minimum size for stability.
        batch_size=batch_size,
        n_epochs=3,
        gamma=0.998,
        # gae_lambda=0.95,
        # learning_rate=learning_rate_schedule,
        # learning_rate=learning_rate_decay_schedule,
        ent_coef=0.05,
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

        # run_model = PPO.load(newest_checkpoint, env=env,)
        # tensorboard_log=tensorboard_log)
        print("\ncheckpoint loaded")
    else:
        print("No checkpoints found.")

    # model_merge_callback = EveryNTimesteps(n_steps=steps * num_cpu * 1024, callback=ModelMergeCallback(args.num_hosts))
    # TODO: Progress callback that collects data from each frame for stats

    # wiill this eliminate the progress bar left hanging out?

    # TODO checkpoints not being saved

    # update_freq = n_steps * num_cpu
    update_freq = n_steps
    
    
    
    # callbacks = [current_stats, tbcallback]
    for episode in range(1, episodes + 1):
        print(f"Starting episode {episode}")
        checkpoint_file_path = (
        f"{checkpoint_path.rstrip('/')}/{os.uname()[1]}-{time.time()}-{episode}/"
    )

        print(f"Checkpoint path: {checkpoint_file_path}")
        checkpoint_callback = CheckpointCallback(
        # save_freq=total_steps // 64,
        save_freq=total_steps // (NUM_CPU * 2),
        save_path=f"{checkpoint_file_path}",
        name_prefix="poke",
        verbose=2,
    )
        current_stats = EveryNTimesteps(
        n_steps=update_freq,
        callback=PokeCaughtCallback(total_steps + (update_freq * 16), multiplier=update_freq, verbose=1),
    )
        tbcallback = TensorboardLoggingCallback(tensorboard_log)
        env.set_attr("episode", episode)
        callbacks = [checkpoint_callback, current_stats, tbcallback]
        run_model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)
        run_model.save(f"{checkpoint_path}/{file_name}-{episode}.zip")
        
        del callbacks
        del checkpoint_callback
        del current_stats
        del tbcallback
        
    # run_model.save(f"{checkpoint_path}/{file_name}-{episode}.zip")



    return run_model


if __name__ == "__main__":
    # TODO: make path a parameter and use more sensible defaults for non-me users
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
    # TODO: make sure the output indicates what is being done and why. e.g. "No directory specified.  Checking current directory for .gb files. Found 3 files. Using the newest one: PokemonYellow.gb"
    # TODO: be "quiet" when parameters are passed and work as expected, but "chatty" when the parameter is skipped and the application is doing "defaulty" things.
    # TODO: DIRECTORY CLEANUP INCLUDING LOGROTATINON.
    parser.add_argument("--game_path", type=str, default="/home/mscs/PokemonYellow.gb")
    # TODO: fix multi-host model merge.  Can we train across multiple instances of the same cart? Can we train across DIFFERENT pokemon carts?
    # TODO: Expirement: If we can train on DIFFERENT pokemon carts, can we train on multiple GB games at a time and build a generally good base "gameboy game" model for training specific games?

    # TODO: Visual gif of map as it exapnds over time, with frames of the game as it is played, so the map is faded gray in the spot the AI isn't currently at.  Should be updated in frame order.  BIG PROJECT.
    # TODO: 5529600 frames is roughly 10 seconds of gametime (144h * 160w * 24fps * 10) and about 5.2mb of data. 10m of data is about 317MB. Math OK? 144 * 160 * 24 * 60 * 10 / 1024 / 1024
    parser.add_argument("--output_dir", type=str, default="ofo")
    parser.add_argument("--num_hosts", type=int, default=1)
    args = parser.parse_args()

    num_cpu = NUM_CPU

    # hrs = 10  # number of hours (in-game) to run for.
    # hrs = 5 # temporarily shorter duration.
    # runsteps = int(3200000 * (hrs))
    # runsteps = int(32000 * (hrs))
    # runsteps = int(3600 * (hrs))

    run_env = None
    # max_frames = PRESS_FRAMES + RELEASE_FRAMES * runsteps

    # episodes = 13
    episodes = 13

    # batch_size = 512 // 4
    batch_size = 64
    n_steps = 2048
    # n_steps = 512
    # total_steps = n_steps * 1024 * 6
    # total_steps = (
    #     60 * 60 * (60 // (PRESS_FRAMES + RELEASE_FRAMES))
    # )  # 8 hours * 60 minutes * 60 seconds * 60 frames per second * 32 // (PRESS_FRAMES + RELEASE_FRAMES)

    # total_steps = num_cpu * n_steps * batch_size * 4
    total_steps = num_cpu * n_steps * batch_size // 8
    

    if num_cpu == 1:
        run_env = DummyVecEnv([make_env(args.game_path, 0, device=device)])
    else:
        run_env = SubprocVecEnv(
            [
                make_env(args.game_path, emunum, device=device)
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
