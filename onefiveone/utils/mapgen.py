import argparse
from pyboy import PyBoy
from tqdm import tqdm
# in theory, I should be able to start the game with a sequence of button presses to get into the world, and then loop through
# every X/Y location and map to record the entire map of the game.  What happens when I set the X/Y in memory? Does it work?
# If I set the player locatino to invalid coordinates, what happens?
# how many possibilities are there on the map with a range of 0,255 for X, Y, and map #?

# 255 * 255 * 255 = 16,581,375 

PRESS_FRAMES=10
RELEASE_FRAMES=20


def button_sequence(game : PyBoy, sequence):
    buttons = {"a":"a", "b":"b", "s":"start", "t":"select", "l":"left", "r":"right", "u":"up" ,"d":"down", "-":None}
    for button in sequence:
        if buttons[button]:
            print("Pressing button", button)
            game.button(buttons[button], 1)
        game.tick(PRESS_FRAMES + RELEASE_FRAMES)

def initialize_game_state(rom_path):
    # Initialize the game state
    game = PyBoy(rom_path, window="SDL2")# , window_type="headless")
    game.set_emulation_speed(0)
    game.button
    for i in tqdm(range(0,10)):
        button_sequence(game, "a-a-a-a-a-a-a-a-a-")

    for i in range(0,10):
        game.tick(PRESS_FRAMES + RELEASE_FRAMES)
    return game
    

if __name__ == "__main__":
    # Accept parameters for rom file and output directory
    parser = argparse.ArgumentParser(description="Generate a map of the game.")
    parser.add_argument("--rom", type=str, help="Path to the game ROM file")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    args = parser.parse_args()
    game = initialize_game_state(args.rom)
    print("Game initialized")

    # loop through all maps:
    for map in range(0,254):
        for x in range(0,254):
            for y in range(0,254):
                # set the game state to the current map, x, and y
                print("Setting position to", map, x, y)
                print(game.memory[0xD36E-1], game.memory[0xD361-1], game.memory[0xD362-1])
                game.memory[0xD361-1] = x
                game.button("b", 1)
                # run the game for a few frames
                game.tick()
                # take a screenshot of the game
                # save the screenshot to the output directory
                # save the map, x, and y to a csv file

    

