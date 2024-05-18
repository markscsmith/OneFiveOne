import os
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import os
import argparse
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from typing import Dict, List, Tuple
import glob


def extract_tensorboard_data(log_dir: str) -> Dict[str, Dict[str, List[Tuple[float, int, any]]]]:
    # Initialize the event accumulator
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    # Available tags in the log file
    tags = event_acc.Tags()
    # Extract scalars
    scalar_data = {}
    for tag in tags['scalars']:
        scalar_events = event_acc.Scalars(tag)
        scalar_data[tag] = [(e.wall_time, e.step, e.value) for e in scalar_events]
    # Extract histograms
    histogram_data = {}
    for tag in tags['histograms']:
        histogram_events = event_acc.Histograms(tag)
        histogram_data[tag] = [(e.wall_time, e.step, e.histogram_value) for e in histogram_events]
    # Extract images
    image_data = {}
    for tag in tags['images']:
        image_events = event_acc.Images(tag)
        image_data[tag] = [(e.wall_time, e.step, e.encoded_image_string) for e in image_events]
    # Extract audio
    audio_data = {}
    for tag in tags['audio']:
        audio_events = event_acc.Audio(tag)
        audio_data[tag] = [(e.wall_time, e.step, e.encoded_audio_string) for e in audio_events]
    # Extract tensors
    tensor_data = {}
    for tag in tags['tensors']:
        tensor_events = event_acc.Tensors(tag)
        tensor_data[tag] = [(e.wall_time, e.step, e.tensor_proto) for e in tensor_events]
    return {
        'scalars': scalar_data,
        'histograms': histogram_data,
        'images': image_data,
        'audio': audio_data,
        'tensors': tensor_data,
    }

def print_tensorboard_data(data: Dict[str, Dict[str, List[Tuple[float, int, any]]]]) -> None:
    actions = {}
    # Print scalar data
    if data['scalars']:
        print("\nScalars:")
        for tag, values in data['scalars'].items():
            print(f"\nTag: {tag}")
            for wall_time, step, value in values:
                print(f"  Step: {step}, Value: {value}")
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
    if data['tensors']:
        #print("\nTensors:")
        for tag, values in data['tensors'].items():
            #print(f"\nTag: {tag}")
            for wall_time, step, tensor_proto in values:
                if "action" in tag:
                    if str(tag) not in actions:
                        actions[str(tag)] = []
                    actions[str(tag)].append((step, tensor_proto))
                #print(f"  Step: {step}, Tensor Proto: {tensor_proto}")
    for env_num, env_actions in actions.items():
        print(f"\nEnvironment {env_num}")
        for step, tensor_proto in env_actions:
            print(f"  Step: {step},")
            for proto in tensor_proto.string_val:
                print(f"Tensor Proto: {proto.decode('utf-8')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and print TensorBoard data.')
    parser.add_argument('--log_dir', type=str, help='Path to the TensorBoard log directory')
    args = parser.parse_args()

    if args.log_dir and os.path.isdir(args.log_dir):
        log_dirs = glob.glob(os.path.join(args.log_dir.strip('/'), '/*/*'), recursive=True)
        for log_dir in log_dirs:
            if os.path.isdir(log_dir):
                data = extract_tensorboard_data(log_dir)
                print_tensorboard_data(data)
    else:
        print(f"The directory {args.log_dir} does not exist.")
