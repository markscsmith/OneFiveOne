import os
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import os
import argparse
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from typing import Dict, List, Tuple
import glob
from tqdm import tqdm
from pyboy import PyBoy
from multiprocessing import Process, cpu_count  # Add cpu_count import

from ..emulator.pyboy_env import PyBoyEnv

CGB = False
PRESS_FRAMES = 10
RELEASE_FRAMES = 20


def extract_tensorboard_data(
    log_dir: str,
) -> Dict[str, Dict[str, List[Tuple[float, int, any]]]]:
    # Initialize the event accumulator
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    # Available tags in the log file
    tags = event_acc.Tags()
    # Extract scalars
    scalar_data = {}
    for tag in tags["scalars"]:
        scalar_events = event_acc.Scalars(tag)
        scalar_data[tag] = [(e.wall_time, e.step, e.value) for e in scalar_events]
    # Extract histograms
    histogram_data = {}
    for tag in tags["histograms"]:
        histogram_events = event_acc.Histograms(tag)
        histogram_data[tag] = [
            (e.wall_time, e.step, e.histogram_value) for e in histogram_events
        ]
    # Extract images
    image_data = {}
    for tag in tags["images"]:
        image_events = event_acc.Images(tag)
        image_data[tag] = [
            (e.wall_time, e.step, e.encoded_image_string) for e in image_events
        ]
    # Extract audio
    audio_data = {}
    for tag in tags["audio"]:
        audio_events = event_acc.Audio(tag)
        audio_data[tag] = [
            (e.wall_time, e.step, e.encoded_audio_string) for e in audio_events
        ]
    # Extract tensors
    tensor_data = {}
    for tag in tags["tensors"]:
        tensor_events = event_acc.Tensors(tag)
        tensor_data[tag] = [
            (e.wall_time, e.step, e.tensor_proto) for e in tensor_events
        ]
    return {
        "scalars": scalar_data,
        "histograms": histogram_data,
        "images": image_data,
        "audio": audio_data,
        "tensors": tensor_data,
    }


def extract_action_data(
    data: Dict[str, Dict[str, List[Tuple[float, int, any]]]]
) -> None:
    actions = {}
    # Print scalar data
    # if data['scalars']:
    # print("\nScalars:")
    # for tag, values in data['scalars'].items():
    # print(f"\nTag: {tag}")
    # for wall_time, step, value in values:

    # print(f"  Step: {step}, Value: {value}")
    # # Print histogram data
    # if data['histograms']:
    #     print("\nHistograms:")
    #     for tag, values in data['histograms'].items():
    #         print(f"\nTag: {tag}")
    #         for wall_time, step, histogram_value in values:
    #             print(f"  Step: {step}, Histogram: {histogram_value}")
    # # Print image data
    # if data['images']:
    #     print("\nImages:")
    #     for tag, values in data['images'].items():
    #         #print(f"\nTag: {tag}")
    #         for wall_time, step, encoded_image_string in values:
    #             #print(f"  Step: {step}, Encoded Image String: [data]")
    # # Print audio data
    # if data['audio']:
    #     #print("\nAudio:")
    #     for tag, values in data['audio'].items():
    #         print(f"\nTag: {tag}")
    #         for wall_time, step, encoded_audio_string in values:
    #             #print(f"  Step: {step}, Encoded Audio String: [data]")
    # Print tensor data
    actions_event = []
    if data["tensors"]:
        # print("\nTensors:")
        for tag, values in data["tensors"].items():
            # print(f"\nTag: {tag}")
            for wall_time, step, tensor_proto in values:
                if "action" in tag:
                    if str(tag) not in actions:
                        actions[str(tag)] = []
                    actions[str(tag)].append((step, tensor_proto))
                # print(f"  Step: {step}, Tensor Proto: {tensor_proto}")
    for env_num, env_actions in actions.items():
        # print(f"\nEnvironment {env_num}")
        for step, tensor_proto in env_actions:
            # print(f"  Step: {step},")
            for proto in tensor_proto.string_val:
                bulk = proto.decode("utf-8").split(":")
                if len(bulk) == 5:
                    button_press, reward, _, caught, seen = proto.decode("utf-8").split(":")
                    action_blob = (env_num, button_press, step, reward, caught, seen)
                    actions_event.append(action_blob)
                # print(f"{env_num} {step} {button_press[-10:]} {reward} {caught} {seen}")
    return actions_event


def process_item(item, args, round, position=0, tfevents_file=""):
    frames = []
    # print(item)
    env = PyBoyEnv(args.rom, emunum=0, cgb=CGB)
    env.reset()
    buttons = {
        0: ("", "-"),
        1: ("up", "U"),
        2: ("down", "D"),
        3: ("left", "L"),
        4: ("right", "R"),
        5: ("a", "A"),
        6: ("b", "B"),
        7: ("start", "S"),
    }
    
    buttons_to_action_map = {
        "-": 0,
        "U": 1,
        "D": 2,
        "L": 3,
        "R": 4,
        "A": 5,
        "B": 6,
        "S": 7,
    }
    print("I:", item[-5:])
    max_frame = len(item)
    curr_frame = 0
    for action in tqdm(item): # tqdm(item, position=round):
        curr_frame += 1
        button = buttons_to_action_map[action[1]]
        env.step(button)
        # print("E:", env.actions.split('|')[-1], ">", ":".join(action).replace("=",""))
        image = env.render_screen_image(target_index=0, frame=curr_frame, max_frame=max_frame, action=action[2], other_info=action[-2:])
        frames.append(image.copy())
    # Save the list of frame PIL images as a gif
    # print("There are", len(frames), "frames.")
    seen = item[-1][-2].split("=")[-1]
    caught = item[-1][-1].split("=")[-1]
    tf_filename = "_".join(tfevents_file.split("/")[-2:])
    frames[0].save(
        f"gif/{tf_filename}_output_{round}_S{seen}_C{caught}.gif",
        save_all=True,
        format="GIF",
        append_images=frames[1:],
        duration=1,
        loop=0,
    )


def action_data_parser(filename, env_num):
    data = ""
    with open(filename, "r") as file:
        data = file.read()

    # U:1:0.00:C0:S0|S:2:0.00:C0:S0|
    # Button, Step, Reward, Caught, Seen
    action_blocks_raw = data.split("|")
    action_blocks = [None] * len(action_blocks_raw)
    max_seen = 0
    max_caught = 0
    final_score = 0
    for i, block in enumerate(action_blocks_raw):
        button, step, score, caught, seen = block.split(":")
        # match original format
        caught = caught[0] + "=" + caught[1:]
        seen = seen[0] + "=" + seen[1:]
        if int(caught.split("=")[-1]) > int(str(max_caught).split("=")[-1]):
            max_caught = caught
        if int(seen.split("=")[-1]) > int(str(max_seen).split("=")[-1]):
            max_seen = seen
        final_score = score
        action_blocks[i] = (env_num, button, step, score, caught, seen)
        
    # print(action_blocks)

    return action_blocks, max_seen, max_caught, final_score


if __name__ == "__main__":

    
    print("Usage: python3 -m onefiveone.utils.tensorlogparse --log_dir ofo/")
    parser = argparse.ArgumentParser(description="Extract and print TensorBoard data.")
    parser.add_argument(
        "--log_dir", type=str, help="Path to the TensorBoard log directory"
    )
    # allow rom to be passed in
    parser.add_argument("--rom", type=str, help="Path to the game ROM file")
    args = parser.parse_args()
    to_emulate = []

    if args.log_dir and os.path.isdir(args.log_dir):
        print(f"Processing log directory: {args.log_dir}")
        # Find all tfevents files in the subdirectories
        tfevents_files = glob.glob(
            os.path.join(args.log_dir, "**/*actions-*.txt"), recursive=True
        )
        for tfevents_file in tfevents_files:
            # data = extract_tensorboard_data(tfevents_file)
            # action_data = extract_action_data(data)
            env_num = tfevents_file.split("/")[-1].split("-")[1].split(".")[0]
            print(f"Processing environment {env_num}")
            action_data, seen, caught, final_score = action_data_parser(tfevents_file, env_num)
            if int(str(seen).split("=")[-1]) > 0:
                to_emulate.append(action_data)
            
            
            
            # print the action data from the item with the highest reward:
            # Sort the action data by the 'rew' field in descending order
            # sorted_action_data = sorted(
            #     action_data, key=lambda x: float(x[3].split("=")[-1]), reverse=True
            # )

            # Print the action data from the item with the highest reward
            # if sorted_action_data:
            #     highest_reward = sorted_action_data[0]
            #     to_emulate.append(highest_reward)
            #     # print("\nAction with the highest reward:")
            #     # print(highest_reward[0])
            #     # print(highest_reward[1])
            #     # print("Actions:", highest_reward[2][-10:])
            #     # print(highest_reward[3])
            #     # print(highest_reward[4])
            # else:
            #     print("No action data found.")
            # to_emulate = sorted(
            # to_emulate, key=lambda x: float(x[3].split("=")[-1]), reverse=True
            # )
        round = 0
        processes = []  # List to keep track of processes
        NUM_CPU = cpu_count()  # Get the number of CPU cores available
        for item in to_emulate:
            round += 1
            process_item(item, args, round, round, tfevents_file)
    
    else:
        print(f"The directory {args.log_dir} does not exist.")


