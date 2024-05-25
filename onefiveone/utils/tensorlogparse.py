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

CGB = False
PRESS_FRAMES = 5
RELEASE_FRAMES = 10

def emulate_game(action_chunk, game_path="PokemonRed.gb"):
    pb = PyBoy(game_path, window_type="headless")
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
    
    offset = -1 if game_path == "PokemonYellow.gb" else 0
    
    caught_pokemon_start = 0xD2F7 + offset
    caught_pokemon_end = 0xD309 + 1 + offset
    seen_pokemon_start = 0xD30A + offset
    seen_pokemon_end = 0xD31C + 1 + offset
    
    button_map = {v[1]: v[0] for k, v in buttons.items()}
    ext = ".state" if CGB else ".ogb_state"
    save_state_path = f"{game_path}{ext}"
    
    pb.load_state(open(save_state_path, "rb"))
    pb.set_emulation_speed(0)
    
    print("action_check", len(action_chunk))
    env, step, actions, rew, caught, seen = action_chunk
    
    def get_pokedex_status_string(data_seen, data_owned):
        def get_status(data, poke_num):
            byte_index = poke_num // 8
            bit_index = poke_num % 8
            return (data[byte_index] >> bit_index) & 1
        
        status_string = ""
        for poke_num in range(151):
            seen = get_status(data_seen, poke_num)
            owned = get_status(data_owned, poke_num)
            if owned and seen:
                status_string += 'O'
            elif seen:
                status_string += 'S'
            elif owned:
                status_string += '?'
            else:
                status_string += '-'
        return status_string
    
    for action in tqdm(actions):
        button = button_map[action]
        if action != "-":
            pb.button_press(button)
        for _ in range(PRESS_FRAMES):
            pb.tick()
        if action != "-":
            pb.button_release(button)
        for _ in range(RELEASE_FRAMES):
            pb.tick()
        
        caught_pokedex = list(pb.memory[caught_pokemon_start:caught_pokemon_end])
        seen_pokedex = list(pb.memory[seen_pokemon_start:seen_pokemon_end])
        
        pokedex_status_string = get_pokedex_status_string(seen_pokedex, caught_pokedex)
        
        print("Caught Pokédex:", caught_pokedex)
        print("Seen Pokédex:", seen_pokedex)
        print("Pokédex Status String:", pokedex_status_string)
        print(pb.frame_count, action, pokedex_status_string)
    
    pb.stop()
    del pb


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
                button_press, reward, _, caught, seen = proto.decode("utf-8").split(":")
                action_blob = (env_num, step, button_press, reward, caught, seen)
                actions_event.append(action_blob)
                # print(f"{env_num} {step} {button_press[-10:]} {reward} {caught} {seen}")
    return actions_event


if __name__ == "__main__":
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
            os.path.join(args.log_dir, "**/*.tfevents.*"), recursive=True
        )
        for tfevents_file in tfevents_files:
            data = extract_tensorboard_data(tfevents_file)
            action_data = extract_action_data(data)
            # print the action data from the item with the highest reward:
            # Sort the action data by the 'rew' field in descending order
            sorted_action_data = sorted(
                action_data, key=lambda x: float(x[3].split("=")[-1]), reverse=True
            )

            # Print the action data from the item with the highest reward
            if sorted_action_data:
                highest_reward = sorted_action_data[0]
                to_emulate.append(highest_reward)
                print("\nAction with the highest reward:")
                print(highest_reward[0])
                print(highest_reward[1])
                print("Actions:", highest_reward[2][-10:])
                print(highest_reward[3])
                print(highest_reward[4])
            else:
                print("No action data found.")
        to_emulate = sorted(
            to_emulate, key=lambda x: float(x[3].split("=")[-1]), reverse=True
        )
        for item in to_emulate:
            print(item)
            emulate_game(item, args.rom)
    else:
        print(f"The directory {args.log_dir} does not exist.")
