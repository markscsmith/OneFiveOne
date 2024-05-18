try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def gpu_test():
    torch_device = None
    if tf is not None:
        tf_device = tf.config.list_physical_devices("GPU")

    if torch is not None:
        torch_device = "cpu"
        torch_device = (
            "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else torch_device
        )
        torch_device = "cuda" if torch.cuda.is_available() else torch_device
    tf_device = None
    return (torch_device, tf_device)


if __name__ == "__main__":
    results = gpu_test()
    print(f"PyTorch Device: {results[0]} | TensorFlow Device: {results[1]}")
