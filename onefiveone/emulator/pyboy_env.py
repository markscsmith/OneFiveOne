import sys
import os
import datetime
import hashlib

# Compute and AI libs
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium import spaces

# Emulator libs
from pyboy import PyBoy

# Output libs
from timg import Renderer, Ansi24HblockMethod
from PIL import Image, ImageDraw, ImageFont

import hashlib



# This allows the bot to press a button and wait for the game state to "settle" before pressing another button.
# 30 frames seems to be a good balance between throughput of the game and allowing the AI to still progress quickly.
PRESS_FRAMES = 10 # Press button for this many frames
RELEASE_FRAMES = 20 # Wait this many frames before pressing again


def diff_flags(s1, s2):
    return [i for i, (c1, c2) in enumerate(zip(s1, s2)) if c1 != c2]

def add_string_overlay(
    image : Image, display_string, position=(20, 20), font_size=40, color=(255, 0, 0)
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
        num_steps,
        save_state_path=None,
        device="cpu",
        episode=0,
        cgb=False,
        **kwargs,
    ):
        super(PyBoyEnv, self).__init__()
        self.max_steps = num_steps
        # Configuring the emulator environment
        self.game_path = game_path
        self.save_state_path = save_state_path

        self.renderer = Renderer()
        self.cart = PokeCart(open(self.game_path, "rb").read())
        self.pyboy = PyBoy(self.game_path, window="null", cgb=cgb)

        self.cgb = cgb
        
        # Configuring the details of this instance of the emulator environment
        self.n = 8  # number of frames to store in the observation space
        self.emunum = emunum
        self.device = device
        self.episode = episode

        # Format the datetime as a string suitable for a Unix filename
        current_datetime = datetime.datetime.now()
        self.filename_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Configure observation and action spaces to match size of memory block used to calculate reward
        offset = self.cart.cart_offset()
        block = self.get_mem_block(offset=offset)[-1]
        
        self.memory_space = Box(
            low=0, high=255, shape=(self.n, len(block)), dtype=np.uint8
        )
        self.screen_space = Box(low=0, high=255, shape=(144, 160, 4), dtype=np.uint8)

        self.location_space = Box(low=0, high=255, shape=(3,), dtype=np.uint8)

        self.observation_space = spaces.Dict({"m":self.memory_space, "s":self.screen_space, "l":self.location_space})

        self.action_space = Discrete(8, start=0) # 8 buttons to press, only one pressed at a time
        
        # Define the mapping of those buttons to the 8 discrete actions
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

        # Tracking variabels for progress
        self.frames = None # Total frames elapsed
        self.step_count = None
        self.flags = None
        

        self.last_n_memories = None
        self.poke_levels = None
        

        self.actions = None # Actions pressed in this episode (string)
        self.last_pokemon_count = None # Last count of pokemon caught
        self.last_seen_pokemon_count = None # Last count of pokemon seen
        self.seen_and_capture_events = None # List of seen and captured pokemon and when they occured

        # Score data
        self.flag_score = None # Total reward given for flags
        
        self.party_exp_reward = None # Total reward given for party exp
        self.total_travel_reward = None # Total reward given for travel
        self.attack_reward = None # Total reward given damage done to opponent pokemon
        
        self.total_reward = None # Tracking total reward accumulated this run

        
        # Player location data
        self.last_player_x = None
        self.last_player_y = None
        self.last_player_x_block = None
        self.last_player_y_block = None
        self.last_player_map = None
        self.visited_xy = None # Set of visited XY coordinates
        self.player_maps = None # Set of visited maps
        self.last_chunk_id = None # Last chunk of visited XY coordinates
        
        # Player data
        self.money = None # Current money in wallet
        self.pokedex = None
        

        # Item data
        self.last_total_items = None
        self.last_items = None
        self.item_points = None
        self.total_item_points = None
        self.last_carried_item_total = None
        self.last_stored_item_total = None

        # Pokemon Data
        self.total_poke_exp = None # Total exp across pokemon in party

        # Opponent data
        self.opponent_pokemon_total_hp = None # total amount of damage done to opponent pokemon
        
        self.no_improvement_limit = 4096
        self.last_improvement_step = 0
        self.best_total_reward = 0
        
        self.last_action = None
        self.consecutive_moves = 0
        
        self.reset()

    def reset(self, seed=0, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.pyboy.stop(save=False)
        del self.pyboy
        self.pyboy = PyBoy(self.game_path, window="null", cgb=self.cgb)
        if self.save_state_path is not None:
            self.pyboy.load_state(open(self.save_state_path, "rb"))
        else:
            print(
                f"No state file. Starting from title screen.",
                file=sys.stderr,
            )
        # Tracking variables for progress
        self.frames = 0
        self.step_count = 0
        self.flags = []
        self.poke_levels = [0, 0, 0, 0, 0, 0]


        self.actions = [None] * self.max_steps
        self.last_pokemon_count = 0
        self.last_seen_pokemon_count = 0
        self.seen_and_capture_events = {}
        

        # Score Data
        self.flag_score = 0

        self.party_exp_reward = 0
        self.total_travel_reward = 0
        self.attack_reward = 0

        self.total_reward = 0
        
        # Player Location Data
        self.last_player_x = 0
        self.last_player_y = 0
        self.last_player_x_block = 0
        self.last_player_y_block = 0
        self.last_player_map = 0
        self.visited_xy = set()
        self.player_maps = set()
        self.last_chunk_id = None

        # Player Data
        self.money = 0
        self.pokedex = "-" * 151

        # Item Data
        self.last_total_items = 0
        self.last_items = []
        self.item_points = {}
        self.total_item_points = 0
        self.last_carried_item_total = 0
        self.last_stored_item_total = 0


        # Pokemon Data
        self.total_poke_exp = 0
        
        # Opponent Data
        self.opponent_pokemon_total_hp = 0
        self.last_n_memories = [self.get_mem_block(self.cart.cart_offset())[-1]] * self.n
        _, observation = self.calculate_reward()

        self.last_improvement_step = 0
        self.best_total_reward = 0

        return observation, {"seed": seed}


    def step(self, action):
        max_steps = self.max_steps
        self.frames = self.pyboy.frame_count

        button = self.buttons[action]
        if action != 0:
            self.pyboy.button(button[0], delay=2)

        self.pyboy.tick(PRESS_FRAMES + RELEASE_FRAMES, True)
       
        reward, observation = self.calculate_reward(action)
        
        reward = round(reward * multiplier, 4)
        self.actions[self.step_count] = f"{button[1]}:{self.step_count}:{self.total_reward:.2f}:C{self.last_pokemon_count}:S{self.last_seen_pokemon_count}:X{self.last_player_x}:Y{self.last_player_y}:M{self.last_player_map}"
        truncated = False
        terminated = False

        if self.total_reward > self.best_total_reward:
            self.best_total_reward = self.total_reward
            self.last_improvement_step = self.step_count

        if (self.step_count - self.last_improvement_step) >= self.no_improvement_limit:
            truncated = True

        info = {
            "reward": reward,
#            "total_reward": self.total_reward,
 #           "actions": self.actions,
            "emunum": self.emunum,
            "frames": self.frames,
            "pokemon_caught": self.last_pokemon_count,
            "pokemon_seen": self.last_seen_pokemon_count,
 #           "visited_xy": self.visited_xy,
            "pokedex": self.pokedex,
  #          "seen_and_capture_events": self.seen_and_capture_events,
        }

        self.step_count += 1
        return observation, reward, terminated, truncated, info
    
    def speed_step(self, action):
        self.step_count += 1
        self.frames = self.pyboy.frame_count

        button = self.buttons[action]
        if action != 0:
            self.pyboy.button(button[0], delay=2)

        self.pyboy.tick(PRESS_FRAMES + RELEASE_FRAMES, True)
        return self.pyboy.screen.image


    def get_mem_block(self, offset):
        # Magic addresses sourced from: https://datacrystal.tcrf.net/wiki/Pokémon_Red_and_Blue/RAM_map
        event_addresses = [
                            0xD5AB, # Starters Back?
                            0xD5C0, # 0=Mewtwo appears, 1=Doesn't (See D85F)
                            0xD5F3, # Have Town map?
                            0xD60D, # Have Oak's Parcel?
                            0xD710, # Fossilized Pokémon?
                            0xD72E, # Did you get Lapras Yet?
                            0xD751, # Fought Giovanni Yet?
                            0xD755, # Fought Brock Yet?
                            0xD75E, # Fought Misty Yet?
                            0xD773, # Fought Lt. Surge Yet?
                            0xD77C, # Fought Erika Yet?
                            0xD782, # Fought Articuno Yet?
                            0xD792, # Fought Koga Yet?
                            0xD79A, # Fought Blaine Yet?
                            0xD7B3, # Fought Sabrina Yet?
                            0xD7D4, # Fought Zapdos Yet?
                            0xD7D8, # Fought Snorlax Yet (Vermilion)
                            0xD7E0, # Fought Snorlax Yet? (Celadon)
                            0xD7EE, # Fought Moltres Yet?
                            0xD803, # Is SS Anne here?
                            0xD85F, # Mewtwo can be caught if bit 2 clear # Needs D5C0 bit 1 clear, too
                            ]

        # Player data
        map = [self.pyboy.memory[0xD35E + offset]]
        location = map + self.pyboy.memory[0xD361 + offset : 0xD365 + offset + 1]

        # Player Objectives
        pokedex = self.pyboy.memory[0xD2F7 + offset : 0xD31C + offset + 1]
        # badges = [self.pyboy.memory[0xD356 + offset]]  # Currently not used

        # Pokemon party data for player and opponent
        my_pokemon = self.pyboy.memory[0xD16B + offset : 0xD272 + offset + 1]
        opponent_pokemon = self.pyboy.memory[0xCFE6 + offset : 0xCFE7 + offset + 1]
        
        
        # Items and money
        items = self.pyboy.memory[0xD31D + offset : 0xD346 + offset + 1]
        stored_items = self.pyboy.memory[0xD53A + offset:0xD59F + offset + 1]

        # pokemart = self.pyboy.memory[0xCF7B + offset:0xCF85 + offset + 1]
        
        money_bytes = self.pyboy.memory[0xD347 + offset : 0xD349 + offset + 1]
        money = int(''.join(f'{byte:02x}' for byte in money_bytes))
        
        # Casino coins
        # chips = self.pyboy.memory[0xD5A4 + offset: 0xD5A5 + offset + 1]
        
        # Flags for various events:
        missable_object_flags = self.pyboy.memory[0xD5A6 + offset: 0xD5C5 + offset + 1]
        event_flags = [self.pyboy.memory[address + offset] for address in event_addresses]
        ss_anne = [event_flags[-2]]
        mewtwo = [event_flags[1] + event_flags[-1]]

        combined_memory = (
            # list(pokemart) +
            list(my_pokemon) +
            list(pokedex) +
            list(items) +
            list(money_bytes) +
            [money] +
            # list(badges) +
            list(location) +
            list(stored_items) +
            # list(chips) +
            list(missable_object_flags) +
            list(event_flags) +
            list(ss_anne) +
            list(mewtwo) +
            list(opponent_pokemon)
            # flatten to a single list:
        )

        pokemart, badges, chips = None, None, None

        return [
            pokemart,
            my_pokemon,
            pokedex,
            items,
            money,
            badges,
            location,
            stored_items,
            chips,
            missable_object_flags,
            event_flags,
            ss_anne,
            mewtwo,
            opponent_pokemon,
            combined_memory,
        ]

    def calculate_reward(self, action=None):
        offset = self.cart.cart_offset()
        mem_block = self.get_mem_block(offset).copy()
        reward = 0
        old_money = self.money
        travel_reward = 0

        (
            pokemart,
            my_pokemon,
            pokedex,
            items,
            money,
            badges,
            location,
            stored_items,
            coins,
            missable_object_flags,
            event_flags,
            ss_anne,
            mewtwo,
            opponent_pokemon,
            combined_memory,
        ) = mem_block
        self.last_n_memories = self.last_n_memories[1:] + [combined_memory]

        # Calculate opponent_pokemon_total_hp
        opponent_pokemon_total_hp = int.from_bytes(opponent_pokemon, byteorder='big')

        # ---- Location data to calculate travel reward ----

        map_id = location[0]
        px = location[1]
        py = location[2]
        pbx = location[3]
        pby = location[4]
        
        self.my_pokemon = my_pokemon

        # Calculate reward from exploring the game world by counting maps, doesn't need to store counter
        if self.last_player_map != map_id:
            if map_id not in self.player_maps:
                travel_reward += 10  # was 5
                self.player_maps.add(map_id)
        self.player_maps.add(map_id)
        event_reward = 0

        chunk_id = f"{px}:{py}:{pbx}:{pby}:{map_id}"

        visited_score = 0
        # if self.last_chunk_id != chunk_id:
        #     if chunk_id in self.visited_xy:
        #         visited_score = 0.01
        #     else:
        #         self.visited_xy.add(chunk_id)
        #         visited_score =  0.1

        self.last_chunk_id = chunk_id

        self.last_player_x = px
        self.last_player_y = py
        self.last_player_x_block = pbx
        self.last_player_y_block = pby
        self.last_player_map = map_id
        
        travel_reward += visited_score
        # Startup values before getting in-game
        # 0000 map 0
        # 6101 map 38
        # 6301 map 38
        self.total_travel_reward += travel_reward


        # ----Pokedex data to calculate reward for pokemon seen and caught----

        # convert binary chunks into a single string
        dex_blocks = 19 # (152 possible bloocks, but only 151 pokemon.)

        full_dex = pokedex
        caught_pokedex = list(full_dex[:dex_blocks])
        seen_pokedex = list(full_dex[dex_blocks:])
            
        last_dex = self.pokedex
        new_dex = self.get_pokedex_status_string(seen_pokedex, caught_pokedex)

        # compare the last pokedex to the current pokedex
        if last_dex != new_dex:
            poke_nums = diff_flags(last_dex, new_dex)
            poke_pairs = zip(poke_nums, [new_dex[p] for p in poke_nums])
            self.seen_and_capture_events[self.pyboy.frame_count] = list(poke_pairs)
            self.visited_xy = set()
            self.player_maps = set()

        self.pokedex = new_dex

        pokemon_owned = self.pokedex.count("O")
        pokemon_seen = self.pokedex.count("S") + pokemon_owned

        last_poke = self.last_pokemon_count
        last_poke_seen = self.last_seen_pokemon_count

        if pokemon_owned > last_poke:
            self.seen_and_capture_events[self.pyboy.frame_count] = (
                pokemon_owned,
                pokemon_seen,
            )
            reward += (pokemon_owned - last_poke) * 300  # was 200

        if pokemon_seen > last_poke_seen:
            self.last_seen_pokemon_count = pokemon_seen
            reward += (pokemon_seen - last_poke_seen) * 300  # was 100

        self.last_pokemon_count = pokemon_owned
        self.last_seen_pokemon_count = pokemon_seen

        party = [
            my_pokemon[0:44],
            my_pokemon[44:88],
            my_pokemon[88:132],
            my_pokemon[132:176],
            my_pokemon[176:220],
            my_pokemon[220:264],
        ]
           
        poke_levels = [poke[33] for poke in party]
        poke_party_bytes = [poke[16:18] for poke in party]
        # TODO: Add HP score to reward calculation: losing HP shouldn't be a reward, but gaining HP should be.
        # Area in memory where the HP is stored is 18:20 for each pokemon. (NEEDS VERIFICATION)
        poke_total_exp = 0
        for party_byte in poke_party_bytes:
            poke_total_exp += int.from_bytes(party_byte, byteorder='big')

        party_exp_reward = 0
        old_exp = self.total_poke_exp
        
        party_exp_reward = np.abs(poke_total_exp - old_exp) / 100
        self.total_poke_exp = poke_total_exp
        
        self.poke_levels = poke_levels
        self.party_exp_reward += party_exp_reward


        # ---- Opponent data to calculate attack rewards ----
        attack_reward = 2.0 * (self.opponent_pokemon_total_hp - opponent_pokemon_total_hp)  # was 1.5
        
        self.attack_reward += max(attack_reward, 0)
            
        self.opponent_pokemon_total_hp = opponent_pokemon_total_hp
     
        # ---- Item data to calculate reward for items collected ----

        item_counts = items[1 + 1::2]
        item_types = items[0 + 1::2]

        stored_item_counts = stored_items[1 + 1::2]
        
        carried_item_total = sum(item_counts)
        stored_item_total = sum(stored_item_counts)

        self.last_stored_item_total = stored_item_total
        self.last_carried_item_total = carried_item_total
        
        self.last_total_items = carried_item_total + stored_item_total

        # calculate points of items based on the number of items added per step
        last_items = self.last_items
        if len(last_items) == 0:
            last_items = [0] * len(item_counts)
        item_diff = [
            np.abs(item_counts[i] - last_items[i]) for i in range(len(item_counts))
        ]
        self.last_items = item_counts

        # create tuple of item type and points
        new_item_points = zip(item_types, item_diff)

        for item, points in new_item_points:
            self.item_points[item] = points

        item_points = sum(self.item_points.values())
        self.total_item_points += item_points

        # ---- Event data to calculate reward for flags ----
        if len(self.flags) == 0 or sum(self.flags) == 0:
            self.flags = event_flags + missable_object_flags
        else:
            flag_diff = diff_flags(self.flags, event_flags + missable_object_flags)
            event_reward += 1 * len(flag_diff)
            self.flags = event_flags + missable_object_flags
            self.flag_score += event_reward



        # ---- Money data to calculate reward for money collected ----

        self.money = money
        money_reward = 0
        if old_money is not None and old_money != money:
            money_divider = 1000
            if money > old_money:
                money_divider = 500
                money_reward = np.abs(money - old_money) / money_divider

        
        reward = (
            party_exp_reward
            + item_points
            + travel_reward
            + attack_reward
            + event_reward
            + money_reward
        )
        # Scale the reward to reduce risk of clipping:
        reward *= 0.1
        # Calculate movement multiplier
        if action is not None:
            if self.last_action == action and action in [1, 2, 3, 4]:  # Only consider directional actions
                self.consecutive_moves += 1
            else:
                self.consecutive_moves = 1
            
            movement_multiplier = 1 + (self.consecutive_moves - 1) * 0.1
            reward *= movement_multiplier
            
            self.last_action = action

        self.total_reward += reward
        return round(reward, 4), {"m":self.last_n_memories, "s":self.pyboy.screen.ndarray.copy(), "l":[px, py, map_id]}


    def render(self, target_index=None, reset=False):
        if target_index is not None and target_index == self.emunum or reset:
            terminal_size = os.get_terminal_size()
            terminal_offset = 7

            image = self.pyboy.screen.image
            w = 160
            h = 144

            new_image = Image.new(
                "RGB", (image.width, image.height + image.height // 2)
            )
            new_image.paste(image, (0, 0))
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
            action_string = [x[0] for x in self.actions[max(0,self.step_count - 6):self.step_count]]

            if target_index is not None:
                render_string = f"{image_string}🧳 {self.episode} 🧠: {int(target_index):2d} 🥾 {int(self.step_count):10d} 🟢 {int(self.last_pokemon_count):3d} 👀 {int(self.last_seen_pokemon_count):3d} 🎒 {int(self.total_item_points):3d} 🌎 {len(self.visited_xy):3d}:{len(self.player_maps):3d} 🏆 {self.total_reward:7.2f} 💪 {self.party_exp_reward:7.2f} 🥊 {int(self.attack_reward):7d} 💰 {int(self.money):7d} 📫 {self.flag_score} \n 🚀 {self.total_travel_reward:4.2f} [{int(self.last_player_x):3d},{int(self.last_player_y):3d}], 🗺️: {int(self.last_player_map):3d} Actions {' '.join(action_string)}:{len(self.actions)} 🎉 {self.poke_levels} 🎬 {int(self.frames):6d} {game_time_string}"
            else:
                render_string = f"{image_string}🧳 {self.episode} 🛠️: {int(self.emunum):2d} 🥾 {int(self.step_count):10d} 🟢 {int(self.last_pokemon_count):3d} 👀 {int(self.last_seen_pokemon_count):3d} 🎒 {int(self.total_item_points):3d} 🌎 {len(self.visited_xy):3d}:{len(self.player_maps):3d} 🏆 {self.total_reward:7.2f} 💪 {self.party_exp_reward:7.2f} 🥊 {int(self.attack_reward):7d}💰 {int(self.money):7d} 📫 {self.flag_score} \n 🚀 {self.total_travel_reward:4.2f} [{int(self.last_player_x):3d},{int(self.last_player_y):3d}], 🗺️: {int(self.last_player_map):3d} Actions {' '.join(action_string)}:{len(self.actions)} 🎉 {self.poke_levels} 🎬 {int(self.frames):6d}"

            return render_string
        
    def render_screen_image(self, target_index=None, reset=False, frame=None, max_frame=None, action=None, other_info=None):
        if target_index is not None and target_index == self.emunum or reset:
            image = Image.fromarray(self.pyboy.screen.ndarray)
            if frame is not None and max_frame is not None:
                image = add_string_overlay(image, f"{frame}/{max_frame}", position=(20, 20))
            if action is not None:
                image = add_string_overlay(image, action, position=(20, 60))
            player_coords = f"{self.last_player_x:3d},{self.last_player_y:3d},{self.last_player_x_block:3d},{self.last_player_y_block:3d},{self.last_player_map:3d}"
            image = add_string_overlay(image, f"{self.total_reward:7.2f}", position=(20, 40), color=(255, 0, 255))
            image = add_string_overlay(image, player_coords, position=(20, 80))
            seen = self.last_seen_pokemon_count
            owned = self.last_pokemon_count
            image = add_string_overlay(image, f"{seen:3d}/{owned:3d}", position=(20, 100))
            image = add_string_overlay(image, f"{self.episode}, {other_info}", position=(20, 120))
            return image
        return Image.new("RGB", (160, 144))

    # 🧠: 19 🟢  64 👀  64 🌎  27:  4 🏆 19270.00 🎒   1 🐆   20.00
    # TODO: build expanding pixel map to show extents of game travelled. (minimap?) Use 3d numpy array to store visited pixels. performance?

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
                status_string += "O"
            elif seen:
                status_string += "S"
            elif owned:
                status_string += "?"
            else:
                status_string += "-"
        return status_string


class PokeCart:
    def __init__(self, cart_data) -> None:
        # calculate checksum of cart_data
        self.cart_data = cart_data
        self.offset = None
        self.checksum = hashlib.md5(cart_data).hexdigest()
        # TODO: Pokemon Gold Silver and Crystal
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
        # identify cart based on checksum
        if self.checksum in self.carts:
            print(
                f"Identified cart: {self.carts[self.checksum]} with offset {self.carts[self.checksum][1]}"
            )
            return self.carts[self.checksum][0]
        else:
            print(f"Unknown cart: {self.checksum}")
            return None

    def cart_offset(self):
        # Pokemon Yellow has offset -1 vs blue and green in memory addresses
        if self.offset is not None:
            return self.offset
        elif self.identify_cart() is not None:
            self.offset = self.carts[self.checksum][1]
            return self.offset
        print("Unknown cart:", self.checksum)
        self.offset = 0
        return self.offset