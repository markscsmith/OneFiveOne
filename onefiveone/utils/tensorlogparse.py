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

import random

CGB = False
PRESS_FRAMES = 10
RELEASE_FRAMES = 20

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
    env = PyBoyEnv(args.rom, emunum=0, cgb=CGB, log_level="CRITICAL")
    env.reset()
    frames = []
    buttons_to_action_map = {"-": 0, "U": 1, "D": 2, "L": 3, "R": 4, "A": 5, "B": 6, "S": 7}
    item, final_score, seen, caught = action_data_parser(tfevents_file, position)
    max_frame = len(item)
    curr_frame = 0

    # Assign unique position to each tqdm instance for multi-line display
    locations = [None] * len(item)
    pos = (position % (cpu_count()))
    tf_filename = "_".join(tfevents_file.split("/")[-2:])
    phase = 0
    for action in tqdm(item, position=pos, desc=f"Processing {position:3d} of {total}"):
        # if curr_frame % 50 == 0 and random.randint(0, 100) < 20:
        #     print("\033[H\033[J")
        curr_frame += 1
        button = buttons_to_action_map[action[1]]
        env.step(button)
        image = env.render_screen_image(target_index=0, frame=curr_frame, max_frame=max_frame, action=action[1], other_info=action[-2:])
        tr = env.total_reward
        if tr > 0:
            tr = round(tr, 3)
        location = [env.last_player_x,
                    env.last_player_y,
                    env.last_player_x_block,
                    env.last_player_y_block,
                    env.last_player_map,
                    tr,]
        locations[curr_frame - 1] = location

        frames.append(image.copy())
        if len(frames) > 10000 or curr_frame == max_frame:
            seen = item[-1][-1].split("=")[-1]
            caught = item[-1][-2].split("=")[-1]
            if not os.path.exists(f"gif/{tf_filename}_output_{roundnum}"):
                os.makedirs(f"gif/{tf_filename}_output_{roundnum}")
            frames[0].save(
                f"gif/{tf_filename}_output_{roundnum}/{phase}_S{seen}_C{caught}.gif",
                save_all=True,
                format="GIF",
                append_images=frames[1:],
                duration=1,
                loop=0,
                )
            phase += 1
            frames = []

    

    print(locations)
    # write locations out to a file next to the gif
    with open(f"gif/{tf_filename}_output_{roundnum}_S{seen}_C{caught}.txt", "w") as file:
        for location in locations:
            file.write(f"{location}\n")
    print("Done processing", tfevents_file)

def action_data_parser(filename, env_num):
    with open(filename, "r") as file:
        data = file.read()

    action_blocks_raw = data.split("|")
    action_blocks = [None] * len(action_blocks_raw)
    max_seen, max_caught, final_score = 0, 0, 0

    for i, block in enumerate(action_blocks_raw):
        button, step, score, caught, seen = block.split(":")
        caught, seen = caught[0] + "=" + caught[1:], seen[0] + "=" + seen[1:]
        if int(caught.split("=")[-1]) > int(str(max_caught).split("=")[-1]):
            max_caught = caught
        if int(seen.split("=")[-1]) > int(str(max_seen).split("=")[-1]):
            max_seen = seen
        final_score = score
        action_blocks[i] = (env_num, button, step, score, caught, seen)

    return action_blocks, max_seen, max_caught, final_score

def main():
    parser = argparse.ArgumentParser(description="Extract and print TensorBoard data.")
    parser.add_argument("--log_dir", type=str, help="Path to the TensorBoard log directory")
    parser.add_argument("--rom", type=str, help="Path to the game ROM file")
    args = parser.parse_args()
    to_emulate = []

    if args.log_dir and os.path.isdir(args.log_dir):
        print(f"Processing log directory: {args.log_dir}")
        tfevents_files = glob.glob(os.path.join(args.log_dir, "**/*actions-*.txt"), recursive=True)
        for tfevents_file in tqdm(tfevents_files):
            env_num = tfevents_file.split("/")[-1].split("-")[1].split(".")[0]
            # action_data, seen, caught, final_score = action_data_parser(tfevents_file, env_num)
            to_emulate.append(tfevents_file)
        try:
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                futures = []
                for position, tfevents_file in enumerate(to_emulate):
                    futures.append(executor.submit(process_item, args, position, position, tfevents_file, len(to_emulate)))

                for future in as_completed(futures):
                    try:
                        future.result()  # Get result to raise exceptions if any
                    except Exception as e:
                        print(f"Error occurred: {e}")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected! Shutting down processes...")
            
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)  # Cancel any pending futures

    else:
        print(f"The directory {args.log_dir} does not exist.")

if __name__ == "__main__":
    main()