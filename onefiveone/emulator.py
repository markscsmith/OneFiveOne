import pyboy
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import argparse
import pyboy
import timg
from timg import Renderer 
from timg.methods import *

import sys

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("game", help="Path to the game ROM file")
parser.add_argument("save", help="Path to the save file")
parser.add_argument("inputs", nargs="*", help="Button presses per frame")
args = parser.parse_args()

# Create pyboy instance
game_path = args.game
save_path = args.save
inputs = args.inputs

if not inputs:
    # Read inputs from stdin
#    inputs = sys.stdin.read().splitlines()
    # get out of house
    inputs = ["ruuuuurrr---ddddddlllld--"]
    # get to the grass
    inputs += ["rrrrruuuuuuu"]
    # wait for oak to come and talk
    inputs += ["------------a---a"]
    # pokemon battle with oak catching it
    inputs += ["-" * 16] 
    inputs += ["a" + "-" * 19 + "a" + "-" * 6 + "a" + "-" * 3]
    # Dialogues
    inputs += ["a" + "-" * 5 + "a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5 + "a" + "-" * 5]
    # Oak walks you to lab, walks away, you walk up to oak
    inputs += ["a" + "-" * 20 + "uuuuuu"]
    # Oak talks to you
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    # Gary complains and Oak is confused
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "ddrrua" "-" * 5]
    # Gary snatches pokemon
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "d" + "lllluua" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    # leave nickname for pikachu
    inputs += ["d" + "a" + "d" * 4 + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    # gary walks over
    inputs += ["a" + "-" * 5 + "a" + "-" * 21]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    # use thundershock!
    inputs += ["a" + "-" * 13 + "a" + "-" * 2 + "a" + "-" * 5]
    
    # use thundershock!
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 13 + "a" + "-" * 2 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 10]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["a" + "-" * 5 + "a" + "-" * 5]
    inputs += ["addddddd--"] 
    # first battle won.

    
    

pyboy_instance = pyboy.PyBoy(game_path, window="null")
pyboy_instance.load_state(open(save_path, 'rb'))
# Process inputs per tick
OFFSET = -1


POKEDEX_START = 0xD2F7
POKEDEX_END = 0xD31C

# fast way to get screen?
# CURRENT_SELECTED_MENU_ITEM = 0xC100
# SELECTED_MENU_ITEM = 0xC1FF

CURRENT_SELECTED_MENU_ITEM = 0xC3A0
SELECTED_MENU_ITEM = 0xC507 + 1


player_x_mem = 0xD361 + OFFSET
player_y_mem = 0xD362 + OFFSET
player_x_block_mem = 0xD363 + OFFSET
player_y_block_mem = 0xD364 + OFFSET
player_map_mem = 0xD35E + OFFSET
item_start = 0xD31E + OFFSET
item_end = 0xD345 + OFFSET
stored_item_start = 0xD53B  + OFFSET
stored_item_end = 0xD59E + OFFSET
input_batches = len(inputs)
renderer = timg.Renderer()

input_batch = 0
last_pokedex = []
pokedex = []
for frame_inputs in inputs:
    current_input = 0
    input_batch += 1
    next_input = []
    
    

    # convert input from u d r l a b s . to pyboy format
    for input_char in frame_inputs:
        if input_char == "u":
            next_input.append((WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP))
        elif input_char == "d":
            next_input.append((WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN))
        elif input_char == "l":
            next_input.append((WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT))
        elif input_char == "r":
            next_input.append((WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT))
        elif input_char == "a":
            next_input.append((WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A))
        elif input_char == "b":
            next_input.append((WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B))
        elif input_char == "s":
            next_input.append((WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START))
        elif input_char == ".":
            next_input.append((WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT))
        else:
            next_input.append((WindowEvent.PASS, WindowEvent.PASS))
    
    input_block_length = len(next_input)
    for input, release_input in next_input:
        current_input += 1
        pyboy_instance.send_input(input)
        
        
        for i in range(8):
            pyboy_instance.tick()
    # Render output using timg
        pyboy_instance.send_input(release_input)
        for i in range(16):
            pyboy_instance.tick()
        last_pokedex = pokedex
        pokedex = [pyboy_instance.memory[POKEDEX_START:POKEDEX_END]]
        pokedex = ["".join(format(byte, '08b') for byte in chunk) for chunk in pokedex]
        render = False
        menu = pyboy_instance.memory[CURRENT_SELECTED_MENU_ITEM: SELECTED_MENU_ITEM]
        if 122 in menu:
            pass
            # render = True
        # convert menu into 20 x 18 rows
        menu = [menu[i:i+20] for i in range(0, len(menu), 20)]
        print(f"{pyboy_instance.frame_count:4d} {current_input} Player X: {pyboy_instance.memory[player_x_mem]} Player Y: {pyboy_instance.memory[player_y_mem]} Player X Block: {pyboy_instance.memory[player_x_block_mem]} Player Y Block: {pyboy_instance.memory[player_y_block_mem]} Player Map: {pyboy_instance.memory[player_map_mem]} Pokedex\n: {pokedex}")
        print(f"Items: {pyboy_instance.memory[item_start:item_end]} \nStored Items: {pyboy_instance.memory[stored_item_start:stored_item_end]}")
        for menu in menu:
            print("".join([f"{byte:3d}" for byte in menu]))
        # if input_batch == input_batches and  current_input > input_block_length - 3 or current_input % 10 == 0:
            

        if pokedex != last_pokedex:
            print(f"{pyboy_instance.frame_count:4d} {current_input} Player X: {pyboy_instance.memory[player_x_mem]} Player Y: {pyboy_instance.memory[player_y_mem]} Player X Block: {pyboy_instance.memory[player_x_block_mem]} Player Y Block: {pyboy_instance.memory[player_y_block_mem]} Player Map: {pyboy_instance.memory[player_map_mem]} Pokedex\n: {last_pokedex}")
            print(f": {pokedex} CHANGE DETECTED!")
            
        if render:
            renderer.load_image(pyboy_instance.screen.image)
            renderer.render(Ansi24HblockMethod)
