import glob
import os

# Small utility to fix error in TensorLog data
# (Better to correct the extraneous np.float64() in the data than to waste the multi-hour run!)
# [2, 0, 0, 0, 37, np.float64(-30.100000000001973)]

def fix_tensorlog_data(data):
    data = data.replace("np.float64(", "").replace(")", "")
    return data


# Open each file specified, read the contents, and call fix_tensorlog_data on the contents
def fix_tensorlog_files(files):
    for file in files:
        with open(file, "r") as f:
            data = f.read()
        data = fix_tensorlog_data(data)
        with open(file, "w") as f:
            f.write(data)
        print("Corrected", file)

# accept a directory and use *.txt glob to get all of the files in the directory
def fix_tensorlog_dir(directory):
    files = glob.glob(os.path.join(directory, "*.txt"))
    fix_tensorlog_files(files)

if __name__ == "__main__":
    import argparse
    # accpet a directory via --dir parameter and call fix_tensorlog_dir on the directory
    parser = argparse.ArgumentParser(description="Fix TensorLog data.")
    parser.add_argument("--dir", type=str, help="Path to the directory containing the files")
    args = parser.parse_args()
    if args.dir:
        fix_tensorlog_dir(args.dir)
    else:
        print("Please provide a directory with the --dir parameter.")
