import os
import sys

from tqdm import tqdm

from PIL import Image
from timg import Renderer, Ansi24HblockMethod
import argparse


def parse_file(file_path, gif_file):
    renderer = Renderer()
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # open gif and turn frames into list of Images
    frames = []
    with Image.open(gif_file) as img:
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(img.copy())
        print (f"Total frames: {img.n_frames}")
    terminal_size = os.get_terminal_size()
    terminal_offset = 7
    w = 160
    h = 144
    
    previous_map = None
    moving_frames = []
    reward_countdown = 0
    for i, line in tqdm(enumerate(lines)):

        # refresh the screen to avoid weird output distortion from newlines
        print("\033[H\033[J")

        x, y, xblock, yblock, map_name, reward = line.strip()[1:-1].split(',')
        x, y, xblock, yblock, reward = int(x), int(y), int(xblock), int(yblock), float(reward)
        if previous_map is not None:
            dx = x - previous_x
            dy = y - previous_y
            dxblock = xblock - previous_xblock
            dyblock = yblock - previous_yblock
            dreward = reward - previous_reward

            if reward_countdown > 0:
                reward_countdown -= 1

            if dreward > 0.5:
                reward_countdown = 10
            
            if dx > 0 or dy > 0 or dxblock > 0 or dyblock > 0 or map_name != previous_map or reward_countdown > 0:
                image = frames[i]
                moving_frames.append(image)
                new_image = Image.new(
                   "RGB", (image.width, image.height + image.height // 2)
                )
                new_image.paste(image, (0, 0))
                renderer.load_image(image)
                renderer.resize(
                    terminal_size.columns, terminal_size.lines * 2 - terminal_offset
                )
                # term_image = renderer.to_string(method_class=Ansi24HblockMethod)
                # renderer.render(Ansi24HblockMethod)
                term_image=""
                print(f"{term_image}\nStep: {i} Delta x: {dx}, Delta y: {dy}, Delta xblock: {dxblock}, Delta yblock: {dyblock}, Map: {map_name}, Reward:{dreward}")



        previous_x, previous_y, previous_xblock, previous_yblock, previous_map, previous_reward = x, y, xblock, yblock, map_name, reward
    # save the moving_frames to a new gif with the same name + "_moving.gif"
    moving_frames[0].save(
        f"{".".join(gif_file.split('.')[:-1])}_moving.gif",
        save_all=True,
        format="GIF",
        append_images=moving_frames[1:],
        duration=1,
        loop=0,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse location data and GIF file.")
    parser.add_argument("--input_file", required=False, help="Path to the input file")
    parser.add_argument("--gif_file", required=False, help="Path to the GIF file")
    parser.add_argument("--gif_dir", required=False, help="Path to the GIF directory")

    args = parser.parse_args()

    # input_file = args.input_file
    # gif_file = args.gif_file

    # with Image.open(gif_file) as img:
    #     total_frames = img.n_frames

    # with open(input_file, 'r') as file:
    #     total_lines = sum(1 for line in file)

    # if total_frames != total_lines:
    #     print("Error: The number of frames in the GIF does not match the number of lines in the input file.")
    #     sys.exit(1)

    # parse_file(input_file, gif_file)
    if args.gif_dir:
        gif_dir = args.gif_dir
        for file_name in os.listdir(gif_dir):
            if file_name.endswith('.txt'):
                print(f"Processing {file_name}")
                txt_file_path = os.path.join(gif_dir, file_name)
                gif_file_path = os.path.join(gif_dir, f"{os.path.splitext(file_name)[0]}.gif")
                if os.path.exists(gif_file_path):
                    with Image.open(gif_file_path) as img:
                        total_frames = img.n_frames

                    with open(txt_file_path, 'r') as file:
                        total_lines = sum(1 for line in file)

                    if total_frames != total_lines:
                        print(f"Error: The number of frames in {gif_file_path} does not match the number of lines in {txt_file_path}.")
                        continue

                    parse_file(txt_file_path, gif_file_path)
                else:
                    print(f"Error: No corresponding GIF file for {txt_file_path}")
    else:
        parse_file(input_file, gif_file)