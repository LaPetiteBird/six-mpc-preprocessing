import argparse
import pathlib
import random

def get_args():
    parser = argparse.ArgumentParser(
        description="Makes inputs for SecureML",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mode", choices=["linear", "logistic", "neural"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dir", default="Player-Data")
    parser.add_argument("--filename", default="Input-P0-0")
    return parser

def rnd_float():
    return f"{random.random():.5f}"

def label_class():  # for classification
    return str(random.randint(0, 1))

def label_real():  # for regression
    return f"{random.random():.5f}"

def write_input_file(mode, outpath):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # use 40 for neural, 100 otherwise
    if mode == "neural":
        num_rows = 40
        label_gen = label_real
    elif mode == "logistic":
        num_rows = 100
        label_gen = label_class
    else:  # linear
        num_rows = 100
        label_gen = label_real

    num_cols = 10

    with outpath.open("w") as f:
        for _ in range(num_rows):
            row = [rnd_float() for _ in range(num_cols)]
            f.write(" ".join(row) + "\n")

        for _ in range(num_rows):
            f.write(label_gen() + "\n")

    print(f"Saved file to: {outpath} (shape: {num_rows}x{num_cols + 1})")

def main():
    args = get_args().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    file_path = pathlib.Path(args.dir) / args.filename
    write_input_file(args.mode, file_path)

    # silence MP-SPDZ warning 
    p1_path = pathlib.Path(args.dir) / "Input-P1-0"
    if not p1_path.exists():
        p1_path.touch()

if __name__ == "__main__":
    main()
