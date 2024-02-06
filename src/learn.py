import multiprocessing
from pyboy import PyBoy, WindowEvent
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from timg import Renderer, Ansi24HblockMethod
from PIL import Image, ImageDraw, ImageFont
from os.path import exists
import gymnasium as gym
import numpy as np
import base64
import torch
import torch.nn as nn
import time
import os
import datetime
import hashlib
import glob
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
        carts ={ "POKEMONG.GBC": 0x00000000,
                 "PMCRYSTA.GBC": 0x00000000,
                 "POKEMONY.GBC": 0x00000000 - 1,
                 "POKEMONB.GBC": 0x00000000,
                 "POKEMONS.GBC": 0x00000000,
                 "POKEMONR.GBC": 0x00000000,}
        if self.identify_cart() in carts:
            return carts[self.identify_cart()]
        else:
            print("Unknown cart:", self.checksum)
            return 0x00000000


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define your custom layers here
        self.extractor = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

class ModelMergeCallback(BaseCallback):
    def __init__(self, num_hosts, verbose=0):
        super(ModelMergeCallback, self).__init__(verbose)
        self.filename_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.num_hosts = num_hosts

    def _on_step(self) -> bool:
        hostname = os.uname()[1]
        file_name = f"{hostname}-{self.filename_datetime}-{self.model.num_timesteps}"
        self.model.save(f"/Volumes/Mag/ofo/{file_name}.zip")
        found_models = self.scan_models()
        time.sleep(1)
        retries = 0
        max_retries = 30
        while len(found_models) < self.num_hosts and retries < max_retries:
            retries += 1
            found_models = self.scan_models()
            time.sleep(1)

            if len(found_models) == self.num_hosts:
                print("Merging models")
                try:
                    self.merge_models(found_models)
                
                    print(f"Model Merge Complete! Merged {len(found_models)} models. New model size: {self.model.num_timesteps}")
                    # Flashy emoji to show model merge clearly on screen!
                    print("""
                      🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
                      🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
                      🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
                      🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
                      🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
                    """)
                    retries = max_retries
                except Exception as e:
                    print(f"Model merge failed: {e}")
                    retries += 1
                    time.sleep(1)

            else:
                print(f"Model error! Only found {len(found_models)} models. Expected {self.num_hosts} models.")
                print("""
                      ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
                      ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
                      ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
                      ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
                      ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
                    """)
                time.sleep(10)



        self.generate_gif_and_actions()

        return True

    def generate_gif_and_actions(self):
        actions = self.training_env.get_attr('actions')
        rewards = self.training_env.get_attr('last_score')
        frames = self.training_env.get_attr('frames')
        visiteds = self.training_env.get_attr('visited_xy')
        pokemon_caughts = self.training_env.get_attr('last_pokemon_count')

        hostname = os.uname()[1]
        file_name = f"{hostname}-{self.filename_datetime}-{self.model.num_timesteps}"
        # Generate a CSV of data
        with open(f"/Volumes/Mag/ofo/{file_name}.csv", "w") as f:
            f.write("env_num,caught,actions,rewards,frames,visiteds\n")
            for env_num, (action, caught, reward, frame, visited) in enumerate(zip(actions, pokemon_caughts, rewards, frames, visiteds)):
                f.write(f"{env_num},{caught},{''.join(action)},{reward},{frame},\"{visited}\"\n")



    def scan_models(self):
        # Force LS of directory to refresh cacche:
        os.system("ls /Volumes/Mag/ofo")
        model_files = glob.glob(f"/Volumes/Mag/ofo/*-*.zip")

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




class CustomNetwork(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomNetwork, self).__init__(*args, **kwargs)
        self.lr_schedule = 0.025

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
        actions = self.training_env.get_attr('actions')
        filename_datetimes = self.training_env.get_attr('filename_datetime')

        all_frames = self.training_env.get_attr('screen_images')


        # for env_num, (rendered_frames, visited, steps, reward, filename_datetime)  in enumerate(zip(all_frames, visiteds, frames, rewards, filename_datetimes)):
        #     print(f"{str(env_num).zfill(3)} 🟣 {all_pokemon_caught[env_num]} 🎬 {steps} 🌎 {len(visited)} 🏆 {reward}")
        #     # Combine frames into gif
        #     rendered_frames[0].save(f"/Volumes/Mag/frames/{env_num}-{filename_datetime}.gif", save_all=True, append_images=rendered_frames[1:], optimize=False, duration=100, loop=0)
            # for frame_num, frame in enumerate(rendered_frames):
            #     frame.save(f"/Volumes/Mag/frames/{env_num}-{frame_num}-{filename_datetime}.png")
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
        print(f"Best: {best_env_idx} 🟢 {all_pokemon_caught[best_env_idx]} 🎬 {frames[best_env_idx]} 🌎 {len(visiteds[best_env_idx])} 🏆 {rewards[best_env_idx]} 🦶 {stationary_frames[best_env_idx]} X: {xs[best_env_idx]} Y: {ys[best_env_idx]} XB: {xbs[best_env_idx]} YB: {ybs[best_env_idx]}, Actinos {actions[best_env_idx][-3:]}")

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
    def __init__(self, game_path, emunum, save_state_path=None):
        super(PyBoyEnv, self).__init__()
        self.pyboy = PyBoy(game_path, window_type="headless", cgb=True)
        self.renderer = Renderer()
        self.actions = []
        self.screen_images = []
        # Define the memory range for 'number of Pokémon caught'
        self.cart = PokeCart(open(game_path, "rb").read())
        self.caught_pokemon_start = 0xD2F7 - self.cart.cart_offset()
        self.caught_pokemon_end = 0xD309 - self.cart.cart_offset()
        self.player_x_mem = 0xD361 - self.cart.cart_offset()
        self.player_y_mem = 0xD362 - self.cart.cart_offset()
        self.player_x_block_mem = 0xD363 - self.cart.cart_offset()
        self.player_y_block_mem = 0xD364 - self.cart.cart_offset()
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
        self.screen_image_arrays = set()
        self.screen_image_arrays_list = []
        
        self.buttons = [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT,
                        WindowEvent.PRESS_ARROW_RIGHT,WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
                        WindowEvent.PRESS_BUTTON_START, WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_ARROW_UP,
                        WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT,
                        WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_START,
                        WindowEvent.RELEASE_BUTTON_SELECT, WindowEvent.PASS]


        self.buttons_names = "UDLRABS!udlrabs. "


        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Format the datetime as a string suitable for a Unix filename
        self.filename_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")



        # Define actioqn_space and observation_space
        self.action_space = gym.spaces.Discrete(17)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(698,),
                                                 dtype=np.uint8)


    def generate_image(self):
        return self.pyboy.botsupport_manager().screen().screen_image()


    def generate_screen_ndarray(self):
        return self.pyboy.botsupport_manager().screen().screen_ndarray()


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
        ticks = 1
        frame_checks = 24
        self.pyboy.send_input(self.buttons[action])
        self.actions.append(self.buttons_names[action])
        for _ in range(ticks):
            self.pyboy.tick()
        if self.frames % frame_checks == 0:
            self.screen_image_arrays.add(self.generate_screen_ndarray().tobytes)


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

        # Don't count the frames where the player is still in the starting menus. Pokemon caught gives more leeway on standing still
        reward = (pokemon_caught + 1) + len(self.visited_xy) + (self.stationary_frames / (pokemon_caught + 1))
        # if np.random.randint(777) == 0 or self.last_pokemon_count != pokemon_caught or self.last_score - reward > 100:
        #     self.render()
        #
        self.last_score = reward
        terminated = False
        truncated = False
        info = {}
        self.last_pokemon_count = pokemon_caught
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        super().reset(**kwargs)
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
        
        if hashable_strings not in self.seen_events:
            self.seen_events.add(hashable_strings)
            # print("OS:EVENTS:", hashable_strings)

        memory_values = memory_values[0xD5A6:0xD85F]
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
        new_env = PyBoyEnv(game_path, emunum=emunum)

        if os.path.exists(game_path + ".state"):
            new_env.pyboy.load_state(open(game_path + ".state", "rb"))
        new_env.pyboy.set_emulation_speed(0)
        return new_env
    return _init


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str)
    parser.add_argument("--num_hosts", type=int, default=1)
    args = parser.parse_args()
    num_cpu = multiprocessing.cpu_count()
    # Hostname and timestamp
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=f"/Volumes/Mag/{os.uname()[1]}-{time.time()}.zip", name_prefix="poke")


    current_stats = EveryNTimesteps(n_steps=10000, callback=PokeCaughtCallback())

    model_merge_callback = EveryNTimesteps(n_steps=250000, callback=ModelMergeCallback(args.num_hosts))


    # num_cpu = 1
    env = SubprocVecEnv([make_env(args.game_path,
                                  emunum) for emunum in range(num_cpu)])


    file_name = "model"
    def train_model(env, num_steps, steps):
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs={},
            net_arch=[dict(pi=[128, 128, 32], vf=[128, 128, 32])],
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
            run_model = PPO(policy=CustomNetwork, n_steps=steps * num_cpu, learning_rate=0.025,
                        env=env, policy_kwargs=policy_kwargs, verbose=1, device=device)

        # TODO: Progress callback that collects data from each frame for stats
        callbacks = [checkpoint_callback, model_merge_callback, current_stats]
        run_model.learn(total_timesteps=runsteps, progress_bar=True, callback=callbacks)
        return run_model
    runsteps = 200000 * 8 # hrs
    model = train_model(env, runsteps, steps=64)
    model.save(f"{file_name}.zip")
