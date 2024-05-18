import sys
import os
from pyboy import PyBoy


def peek_memory(game_file, save_file):
    pyboy = PyBoy(game_file, window="null", cgb=False)
    pyboy.set_emulation_speed(0)
    pyboy.load_state(open(save_file, "rb"))
    print("Loaded save state")
    print("Peeking memory")
    item_start = 0xD31E
    item_end = 0xD345

    memory = pyboy.memory[item_start:item_end]
    print(f"Items={memory}")
    pyboy.stop()


if __name__ == "__main__":
    # accept command line parameters for game_file and save_file
    game_file = sys.argv[1]
    save_file = sys.argv[2]
    peek_memory(game_file, save_file)
