import os
import sys
from tqdm import tqdm
from PIL import Image
from timg import Renderer, Ansi24HblockMethod
import argparse
import subprocess


def parse_file(file_path, gif_files = []):
    renderer = Renderer()
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # open gif and turn frames into list of Images
    frames = []
    gif_files.sort(key=lambda x: int(x.split("/")[-1].split('_')[0]))
    for gif_file in gif_files:
        with Image.open(gif_file) as img:
            for i in range(img.n_frames):
                img.seek(i)
                frames.append(img.copy())
            print (f"Total frames: {len(frames)}")
            
    terminal_size = os.get_terminal_size()
    terminal_offset = 7
    w = 160
    h = 144
    
    previous_map = None
    moving_frames = []
    maps = [None] * 256
    reward_countdown = 0

    map_extents = {}


    term_image = ""
    for i, line in enumerate(tqdm(lines)):

        # refresh the screen to avoid weird output distortion from newlines
#        print("\033[H\033[J")

        y, x, xblock, yblock, map_name, reward = line.strip()[1:-1].split(',')
        y, x, xblock, yblock, map_name, reward = int(y), int(x), int(xblock), int(yblock), int(map_name), float(reward)
        
        if maps[map_name] is None:
            new_map = Image.new("RGB", (4096, 4096))
            new_map.paste(frames[i], (x * 16, y * 16))
            maps[map_name] = new_map
            renderer.load_image(new_map)
            renderer.resize(
                terminal_size.columns, terminal_size.lines * 2 - terminal_offset
            )
            term_image = renderer.to_string(method_class=Ansi24HblockMethod)
            
        
        if map_name not in map_extents:
            map_extents[map_name] = {"min_x": x, "max_x": x, "min_y": y, "max_y": y}
        else:
            if x < map_extents[map_name]["min_x"]:
                map_extents[map_name]["min_x"] = x
            if x > map_extents[map_name]["max_x"]:
                map_extents[map_name]["max_x"] = x
            if y < map_extents[map_name]["min_y"]:
                map_extents[map_name]["min_y"] = y
            if y > map_extents[map_name]["max_y"]:
                map_extents[map_name]["max_y"] = y
        
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
            
            if dx != 0 or dy != 0 or dxblock != 0 or dyblock != 0 or map_name != previous_map or reward_countdown > 0:
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
                # term_image=""
                

                maps[int(map_name)].paste(frames[i], (x * 16, y * 16))
                # renderer.load_image(maps[int(map_name)])
                # renderer.resize(
                #     terminal_size.columns, terminal_size.lines * 2 - terminal_offset
                # )
                # term_image = renderer.to_string(method_class=Ansi24HblockMethod)
                # print(f"{term_image}\nStep: {i} Delta x: {dx}, Delta y: {dy}, Delta xblock: {dxblock}, Delta yblock: {dyblock}, Map: {map_name}, Reward:{round(dreward,3)}")



        previous_x, previous_y, previous_xblock, previous_yblock, previous_map, previous_reward = x, y, xblock, yblock, map_name, reward
    # save the moving_frames to a new gif with the same name + "_moving.gif"
    print(f"{term_image}\nMap extents: {map_extents}")

    output_gif = f"{".".join(file_path.split('.')[:-1])}_moving.gif"
    moving_frames[0].save(
        output_gif,
        save_all=True,
        format="GIF",
        append_images=moving_frames[1:],
        duration=1,
        loop=0,
    )
    
    gif_to_movie(output_gif, f"{".".join(output_gif.split('.')[:-1])}.mp4")

    for map_name, map_image in enumerate(maps):
        if map_image is not None:

            map_image = map_image.crop((map_extents[map_name]["min_x"] * 16, map_extents[map_name]["min_y"] * 16, map_extents[map_name]["max_x"] * 16 + 160, map_extents[map_name]["max_y"] * 16 + 144))
            map_image.save(f"{".".join(output_gif.split('.')[:-1])}_map_{map_name}.png")


def gif_to_movie(gif_path, movie_path):
    command = [
        'ffmpeg', '-i', gif_path, '-movflags', 'faststart', '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', movie_path
    ]
    subprocess.run(command, check=True)


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
                gif_folder_path =  f"{"_".join(txt_file_path.split('_')[:-2])}"
                gifs = [os.path.join(gif_folder_path, f) for f in os.listdir(gif_folder_path) if f.endswith('.gif')]
                if len(gifs) == 0:
                    print(f"Error: No GIF files found in {gif_folder_path}")
                    continue
                # sort gifs by the number in the filename before the first _
                parse_file(txt_file_path, gifs)
                    
                
                # if os.path.exists(gif_file_path):
                #     with Image.open(gif_file_path) as img:
                #         total_frames = img.n_frames

                #     with open(txt_file_path, 'r') as file:
                #         total_lines = sum(1 for line in file)

                #     if total_frames != total_lines:
                #         print(f"Error: The number of frames in {gif_file_path} does not match the number of lines in {txt_file_path}.")
                #         continue

                #     parse_file(txt_file_path, gif_file_path)
                # else:
                #     print(f"Error: No corresponding GIF file for {txt_file_path}")
    # else:
    #     parse_file(input_file, gif_file)