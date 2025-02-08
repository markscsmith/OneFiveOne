# TODO: Memory usage per thread goes up to like 15gb while processing to gif. This is tooooo much!

import os
import sys
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import argparse
import glob
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
from ..emulator.pyboy_env import PyBoyEnv
from PIL import Image, ImageDraw, ImageFont

import random

CGB = False
PRESS_FRAMES = 10
RELEASE_FRAMES = 20
FRAME_BATCH_SIZE = 5000

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

def extract_tensorboard_data(log_dir: str) -> Dict[str, Dict[str, List[Tuple[float, int, any]]]]:
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    tags = event_acc.Tags()
    scalar_data, histogram_data, image_data, audio_data, tensor_data = {}, {}, {}, {}, {}

    for tag in tags["scalars"]:
        scalar_events = event_acc.Scalars(tag)
        scalar_data[tag] = [(e.wall_time, e.step, e.value) for e in scalar_events]

    for tag in tags["histograms"]:
        histogram_events = event_acc.Histograms(tag)
        histogram_data[tag] = [(e.wall_time, e.step, e.histogram_value) for e in histogram_events]

    for tag in tags["images"]:
        image_events = event_acc.Images(tag)
        image_data[tag] = [(e.wall_time, e.step, e.encoded_image_string) for e in image_events]

    for tag in tags["audio"]:
        audio_events = event_acc.Audio(tag)
        audio_data[tag] = [(e.wall_time, e.step, e.encoded_audio_string) for e in audio_events]

    for tag in tags["tensors"]:
        tensor_events = event_acc.Tensors(tag)
        tensor_data[tag] = [(e.wall_time, e.step, e.tensor_proto) for e in tensor_events]

    return {
        "scalars": scalar_data,
        "histograms": histogram_data,
        "images": image_data,
        "audio": audio_data,
        "tensors": tensor_data,
    }

def process_item(args, roundnum, position=0, tfevents_file="", total=0):
    print(f"Processing item {tfevents_file}...")
    
    frames = []
    buttons_to_action_map = {"-": 0, "U": 1, "D": 2, "L": 3, "R": 4, "A": 5, "B": 6, "S": 7}
    item, current_score, seen, caught = action_data_parser(tfevents_file, position)
    
    env = PyBoyEnv(args.rom, emunum=0, cgb=CGB, log_level="CRITICAL", num_steps = len(item))
    env.reset()
    seen = item[-1][4].split("=")[-1]
    caught = item[-1][5].split("=")[-1]
    max_frame = len(item)
    curr_frame = 0

    # Assign unique position to each tqdm instance for multi-line display
    locations = [None] * len(item)
    print("Processing", len(item), "frames.")
    pos = (position % (cpu_count()))
    tf_filename = "_".join(tfevents_file.split("/")[-2:])
    phase = 0
    last_score = float(-1)
    max_seen = 0
    max_caught = 0
    for action in item:
        # if curr_frame % 50 == 0 and random.randint(0, 100) < 20:
        #     print("\033[H\033[J")
        _, button, _, score, raw_seen, raw_caught, x, y, map_num = action
        reset = False
        if round(float(score), 3) < last_score:
            # reset the environment 
            env.reset()
            reset = True
            print("Resetting environment due to score decrease.", score, last_score, action)
        
        button = buttons_to_action_map[action[1]]
        
        # image = env.render_screen_image(target_index=0, frame=curr_frame, max_frame=max_frame, action=action[1], other_info=action[-2:])

        
        image = env.speed_step(button)
        seen = int(raw_seen.split("=")[-1])
        caught = int(raw_caught.split("=")[-1])

        if seen > max_seen:
            max_seen = seen
        if caught > max_caught:
            max_caught = caught


        
        tr = round(float(score), 3)
        last_score = tr
        location = [x,
                    y,
                    map_num,
                    tr,]
        locations[curr_frame] = location
        curr_frame += 1
        image = add_string_overlay(image, f"Step: {curr_frame}/{max_frame}", position=(20, 20))
        frames.append(image)
        if len(frames) > FRAME_BATCH_SIZE or curr_frame == max_frame or reset:
            output_dir = args.output_dir if args.output_dir else "gif"
            if not os.path.exists(f"{output_dir}/{tf_filename}_output_{roundnum}"):
                os.makedirs(f"{output_dir}/{tf_filename}_output_{roundnum}")
            filename = f"{output_dir}/{tf_filename}_output_{roundnum}/{phase}_S{max_seen}_C{max_caught}.gif"
            print("Saving", filename)
            frames[0].save(
                filename,
                save_all=True,
                format="GIF",
                append_images=frames[1:],
                duration = 1,
                loop=0,
                )
            # load image to verify all frames written:
            image = Image.open(filename)
            if image.n_frames != len(frames):
                print(f"Error: {filename} has {image.n_frames} frames, but {len(frames)} were written.")
                sys.exit(1)

            phase += 1
            frames = []
    

    

    # write locations out to a file next to the gif
    with open(f"{output_dir}/{tf_filename}_output_{roundnum}_S{seen}_C{caught}.txt", "w") as file:
        for location in locations:
            file.write(f"{location}\n")
    print("Done processing", tfevents_file)

def action_data_parser(filename, env_num):
    with open(filename, "r") as file:
        data = file.read()

    action_blocks_raw = data.split("|")
    action_blocks = [None] * len(action_blocks_raw)
    print("Processing", len(action_blocks_raw), "action blocks.")
    max_seen, max_caught, final_score = 0, 0, 0

    for i, block in enumerate(action_blocks_raw):
        button, step, score, caught, seen, x, y, map_num = block.split(":")
        x = int(x[1:])
        y = int(y[1:])
        map_num = int(map_num[1:])
        caught, seen = caught[0] + "=" + caught[1:], seen[0] + "=" + seen[1:]
        if int(caught.split("=")[-1]) > int(str(max_caught).split("=")[-1]):
            max_caught = caught
        if int(seen.split("=")[-1]) > int(str(max_seen).split("=")[-1]):
            max_seen = seen
        if float(score) > float(final_score):
            final_score = score
        action_blocks[i] = (env_num, button, step, score, caught, seen, x, y, map_num)

    return action_blocks, final_score, max_seen, max_caught, 

def main():
    parser = argparse.ArgumentParser(description="Extract and print TensorBoard data.")
    parser.add_argument("--log_dir", type=str, help="Path to the TensorBoard log directory")
    parser.add_argument("--rom", type=str, help="Path to the game ROM file")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory for gifs", default="gif")
    parser.add_argument("--multithread", action="store_true", help="Use multiple threads to process the data")
    args = parser.parse_args()
    to_emulate = []

    if args.log_dir and os.path.isdir(args.log_dir):
        print(f"Processing log directory: {args.log_dir}")
        tfevents_files = glob.glob(os.path.join(args.log_dir, "**/*actions-*.txt"), recursive=True)
        if args.multithread:
            for tfevents_file in tfevents_files:
                env_num = tfevents_file.split("/")[-1].split("-")[1].split(".")[0]
                # action_data, seen, caught, final_score = action_data_parser(tfevents_file, env_num)
                to_emulate.append(tfevents_file)
            try:
                with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                # with ProcessPoolExecutor(max_workers=1) as executor:
                    futures = []
                    for position, tfevents_file in enumerate(to_emulate):
                        futures.append(executor.submit(process_item, args, position, position, tfevents_file, len(to_emulate)))

                    for future in as_completed(futures):

                        future.result()  # Get result to raise exceptions if any
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt detected! Shutting down processes...")
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                executor.shutdown(wait=True, cancel_futures=True)  # Cancel any pending futures
        else:
            for position, tfevents_file in enumerate(tfevents_files):
                process_item(args, 0, position, tfevents_file, len(tfevents_files))

    else:
        print(f"The directory {args.log_dir} does not exist.")

if __name__ == "__main__":
    main()