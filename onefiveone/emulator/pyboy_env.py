import sys
import os
import datetime
import hashlib

# Compute and AI libs
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# Emulator libs
from pyboy import PyBoy

# Output libs
from timg import Renderer, Ansi24HblockMethod
from PIL import Image

import hashlib
from pyboy import PyBoy


PRESS_FRAMES = 10 # Press button for this many frames
RELEASE_FRAMES = 20 # Wait this many frames before pressing again


def diff_flags(s1, s2):
    return [i for i, (c1, c2) in enumerate(zip(s1, s2)) if c1 != c2]

class PyBoyEnv(gym.Env):
    def __init__(
        self,
        game_path,
        emunum,
        save_state_path=None,
        device="cpu",
        episode=0,
        cgb=False,
        **kwargs,
    ):
        super(PyBoyEnv, self).__init__()
        self.pyboy = PyBoy(game_path, window="null", cgb=cgb, log_level="CRITICAL")
        self.game_path = game_path
        self.n = 8  # number of frames to store
        
        self.renderer = Renderer()
        self.cgb = cgb
        self.actions = ""

        self.cart = PokeCart(open(game_path, "rb").read())
        offset = self.cart.cart_offset()

        # Pokedex memory locations

        self.party_exp_reward = None

        self.emunum = emunum
        self.save_state_path = save_state_path
        
        self.visited_xy = None
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
        self.money = None
        self.total_poke_exp = None
        self.last_total_items = 0
        self.last_items = []
        self.item_points = {}
        self.total_item_points = 0
        self.opponent_party = []
        self.total_reward = 0

        self.opponent_pokemon_total_hp = 0

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
        self.attack_reward = 0
        self.flags = []
        self.flag_score = 0

        self.last_memory_update_frame = 0
        self.current_memory = None
        self.last_n_memories = []

        self.party_exp = [0, 0, 0, 0, 0, 0]
        self.poke_levels = [0, 0, 0, 0, 0, 0]

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

        size = 1327

    
        block = self.get_mem_block(offset=offset)[-1]

        self.observation_space = Box(
            low=0, high=255, shape=(self.n, len(block)), dtype=np.uint8
        )
        self.observation_space

        self.action_space = Discrete(8, start=0)

    def get_mem_block(self, offset):

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

        pokemart = self.pyboy.memory[0xCF7B + offset:0xCF85 + offset + 1]
        my_pokemon = self.pyboy.memory[0xD16B + offset : 0xD272 + offset + 1]
        pokedex = self.pyboy.memory[0xD2F7 + offset : 0xD31C + offset + 1]
        items = self.pyboy.memory[0xD31D + offset : 0xD346 + offset + 1]
        money_bytes = self.pyboy.memory[0xD347 + offset : 0xD349 + offset + 1]
        money = int(''.join(f'{byte:02x}' for byte in money_bytes))
        badges = [self.pyboy.memory[0xD356 + offset]]
        location = self.pyboy.memory[0xD35E + offset : 0xD365 + offset + 1]
        stored_items = self.pyboy.memory[0xD53A + offset:0xD59F + offset + 1]
        coins = self.pyboy.memory[0xD5A4 + offset: 0xD5A5 + offset + 1]
        missable_object_flags = self.pyboy.memory[0xD5A6 + offset: 0xD5C5 + offset + 1]
        event_flags = [self.pyboy.memory[address + offset] for address in event_addresses]
        ss_anne =[self.pyboy.memory[0xD803] + offset]
        mewtwo = [self.pyboy.memory[0xD85F] + offset]
        opponent_pokemon = self.pyboy.memory[0xCFE6 + offset : 0xCFE7 + offset + 1]
        
        combined_memory = []
        combined_memory.extend(pokemart)
        combined_memory.extend(my_pokemon)
        combined_memory.extend(pokedex)
        combined_memory.extend(items)
        combined_memory.extend(money_bytes)
        combined_memory.append(money)
        combined_memory.extend(badges)
        combined_memory.extend(location)
        combined_memory.extend(stored_items)
        combined_memory.extend(coins)
        combined_memory.extend(missable_object_flags)
        combined_memory.extend(event_flags)
        combined_memory.extend(ss_anne)
        combined_memory.extend(mewtwo)
        combined_memory.extend(opponent_pokemon)

        return [
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
        ]

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
                status_string += "O"
            elif seen:
                status_string += "S"
            elif owned:
                status_string += "?"
            else:
                status_string += "-"
        return status_string

    def calculate_reward(self):
        offset = self.cart.cart_offset()  # + MEM_START
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
        if len(self.last_n_memories) == 0:
            self.last_n_memories = [combined_memory] * self.n
        else:
            self.last_n_memories = self.last_n_memories[1:] + [combined_memory]

        
        
        self.opponent_party = opponent_pokemon
        self.money = money
        map_id = location[0]

        px = location[1]
        py = location[2]
        pbx = location[3]
        pby = location[4]

        self.my_pokemon = my_pokemon

        # Calculate reward from exploring the game world by counting maps, doesn't need to store counter
        if self.last_player_map != map_id:
            if map_id not in self.player_maps:
                travel_reward += 0.02
                self.player_maps.add(map_id)
            else:
                travel_reward += 0
        
        event_reward = 0

        if len(self.flags) == 0 or sum(self.flags) == 0:
            self.flags = event_flags + missable_object_flags
        else:
            flag_diff = diff_flags(self.flags, event_flags + missable_object_flags)
            if len(flag_diff) > 0:
                event_reward += 1 * len(flag_diff)
                self.flags = event_flags + missable_object_flags
                self.flag_score += event_reward
        

        

        chunk_id = f"{px}:{py}:{pbx}:{pby}:{map_id}"

        visited_score = 0
        if self.last_chunk_id != chunk_id:
            if chunk_id in self.visited_xy:
                # TODO: restore negative and positive rewards for visiting new chunks and revisiting cold ones
                # Targeted for after issue #10 is resolved.
                # visited_score = -0.1
                pass
            else:
                self.visited_xy.add(chunk_id)
                # visited_score =  0.1
                pass

        self.last_chunk_id = chunk_id

        travel_reward += visited_score

        # convert binary chunks into a single string
        dex_blocks = 19 # (152 possible bloocks, but only 151 pokemon.)

        full_dex = pokedex
        caught_pokedex = list(full_dex[:dex_blocks])
        seen_pokedex = list(full_dex[dex_blocks:])
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
            self.player_maps = set()

        self.pokedex = new_dex

        pokemon_owned = self.pokedex.count("O")
        pokemon_seen = self.pokedex.count("S") + pokemon_owned

        last_poke = self.last_pokemon_count
        last_poke_seen = self.last_seen_pokemon_count

        if pokemon_seen == 0:
            pokemon_owned = 0

        if pokemon_owned > last_poke:
            self.seen_and_capture_events[self.pyboy.frame_count] = (
                pokemon_owned,
                pokemon_seen,
            )
            reward += (pokemon_owned - last_poke) * 10
            self.last_pokemon_count = pokemon_owned

        if pokemon_seen > last_poke_seen:
            self.last_seen_pokemon_count = pokemon_seen
            reward += (pokemon_seen - last_poke_seen) * 10

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


        attack_reward = 0
        opponent_pokemon_total_hp = int.from_bytes(opponent_pokemon, byteorder='big')
        if opponent_pokemon_total_hp > 0 and self.opponent_pokemon_total_hp > opponent_pokemon_total_hp:
            attack_reward = (self.opponent_pokemon_total_hp - opponent_pokemon_total_hp)
        
        self.attack_reward += attack_reward
            
        self.opponent_pokemon_total_hp = opponent_pokemon_total_hp
        
        poke_levels = [poke[33] for poke in party]
        poke_party_bytes = [poke[16:18] for poke in party]
        poke_total_exp = 0
        for party_byte in poke_party_bytes:
            poke_total_exp += int.from_bytes(party_byte, byteorder='big')



        exp_reward = 0
        if self.total_poke_exp is None:
            self.total_poke_exp = poke_total_exp
        else:
            old_exp = self.total_poke_exp
            if poke_total_exp != old_exp:
                exp_reward = np.abs(poke_total_exp - old_exp) / 100
                self.total_poke_exp = poke_total_exp
        

        party_exp_reward = exp_reward
        self.poke_levels = poke_levels


        item_counts = items[1 + 1::2]
        item_types = items[0 + 1::2]

        stored_item_counts = stored_items[1 + 1::2]
        
        carried_item_total = sum(item_counts)
        stored_item_total = sum(stored_item_counts)

        last_carried_item_total = self.last_carried_item_total
        last_stored_item_total = self.last_stored_item_total

        self.last_stored_item_total = stored_item_total
        self.last_carried_item_total = carried_item_total

        last_total_items = self.last_total_items
        if carried_item_total + stored_item_total != last_total_items:
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
            if item == 0 or item == 255:
                pass
            else:
                self.item_points[item] = points

        item_points = sum(self.item_points.values())
        self.total_item_points += item_points

        reward += (
            
            party_exp_reward / 500
            + item_points
            + travel_reward
            + attack_reward
            + event_reward
        )

        if old_money is not None and old_money != money:
            money_divider = 1000
            if money > old_money:
                money_divider = 500

            reward += np.abs(money - old_money) / money_divider

        self.party_exp_reward += party_exp_reward / 500
        self.travel_reward = travel_reward

        self.last_player_x = px
        self.last_player_y = py
        self.last_player_x_block = pbx
        self.last_player_y_block = pby
        self.last_player_map = map_id

        self.total_reward += reward

        return round(reward, 4), self.last_n_memories

    # TODO: Refactor so returns image instead of immediately rendering so PokeCaughtCallback can render instead.
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
            if target_index is not None:
                render_string = f"{image_string}🧳 {self.episode} 🧠: {target_index:2d} 🥾 {self.step_count:10d} 🟢 {self.last_pokemon_count:3d} 👀 {self.last_seen_pokemon_count:3d} 🎒 {self.total_item_points:3d} 🌎 {len(self.visited_xy):3d}:{len(self.player_maps):3d} 🏆 {self.total_reward:7.2f} 💪 {self.party_exp_reward:7.2f} 🥊 {self.attack_reward:7d} 💰 {self.money:7d} 📫 {self.flag_score} \n[{self.last_player_x:3d},{self.last_player_y:3d},{self.last_player_x_block:3d},{self.last_player_y_block:3d}], 🗺️: {self.last_player_map:3d} Actions {' '.join(self.actions[-6:])} 🎉 {self.poke_levels} 🎬 {self.frames:6d} {game_time_string} {len(self.actions)}"
            else:
                render_string = f"{image_string}🧳 {self.episode} 🛠️: {self.emunum:2d} 🥾 {self.step_count:10d} 🟢 {self.last_pokemon_count:3d} 👀 {self.last_seen_pokemon_count:3d} 🎒 {self.total_item_points:3d} 🌎 {len(self.visited_xy):3d}:{len(self.player_maps):3d} 🏆 {self.total_reward:7.2f} 💪 {self.party_exp_reward:7.2f} 🥊 {self.attack_reward:7d}💰 {self.money:7d} 📫 {self.flag_score} \n[{self.last_player_x:3d},{self.last_player_y:3d},{self.last_player_x_block:3d},{self.last_player_y_block:3d}], 🗺️: {self.last_player_map:3d} Actions {' '.join(self.actions[-6:])} 🎉 {self.poke_levels} 🎬 {self.frames:6d} {len(self.actions)}"

            return render_string

    # 🧠: 19 🟢  64 👀  64 🌎  27:  4 🏆 19270.00 🎒   1 🐆   20.00
    # TODO: build expanding pixel map to show extents of game travelled. (minimap?) Use 3d numpy array to store visited pixels. performance?

    def step(self, action):
        self.step_count += 1
        self.frames = self.pyboy.frame_count

        button = self.buttons[action]
        if action != 0:
            self.pyboy.button(button[0], delay=2)

        self.pyboy.tick(PRESS_FRAMES + RELEASE_FRAMES, True)

        self.actions = f"{self.actions}{button[1]}"
       
        reward, observation = self.calculate_reward()
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
            "badges": self.badges,
        }

        return observation, reward, terminated, truncated, info

    def set_episode(self, episode):
        self.episode = episode

    def reset(self, seed=0, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.last_memory_update_frame = 0
        self.last_total_items = 0
        self.last_items = []
        self.item_points = {}
        self.travel_reward = 0
        self.last_chunk_id = None
        self.total_poke_exp = None
        self.party_exp_reward = 0
        self.party_exp = [0, 0, 0, 0, 0, 0]
        self.poke_levels = [0, 0, 0, 0, 0, 0]
        self.step_count = 0
        self.visited_xy = set()
        self.player_maps = set()
        self.last_player_x = 0
        self.last_player_y = 0
        self.last_player_x_block = 0
        self.last_player_y_block = 0
        self.money = None
        self.pokedex = "-" * 151
        self.opponent_pokemon_total_hp = 0
        self.attack_reward = 0
        self.total_reward = 0
        self.flag_score = 0
        self.flags = []
        self.pyboy = PyBoy(
            self.game_path,
            window="null",
            cgb=self.cgb,
            log_level="CRITICAL",
        )
        self.opponent_party = []

        if self.save_state_path is not None:
            self.pyboy.load_state(open(self.save_state_path, "rb"))
        else:
            print(
                f"No state file. Starting from title screen.",
                file=sys.stderr,
            )

        self.actions = ""
        self.last_score = 0
        self.last_pokemon_count = 0
        self.frames = 0
        self.last_player_x = 0
        self.last_player_y = 0
        self.last_player_x_block = 0
        self.last_player_y_block = 0
        self.total_item_points = 0


        _, observation = self.calculate_reward()

        return observation, {"seed": seed}

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