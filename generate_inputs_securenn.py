import argparse
import os
from pathlib import Path
import numpy as np 

BATCHES     = 10      # we only loop 10×, good enough
BATCH_SIZE  = 128     # mini-batch
INPUT_DIM   = 784     # 28*28 pixels
OUTPUT_DIM  = 10      # digits 0-9  

TOTAL_LINES = BATCHES * BATCH_SIZE * (INPUT_DIM + OUTPUT_DIM)


def dump_all_the_numbers(seed=42, out_dir="Player-Data"):
    rng = np.random.default_rng(seed)

    p0 = Path(out_dir) / "Input-P0-0"
    p0.parent.mkdir(parents=True, exist_ok=True)

    with p0.open("w") as f:

        for _ in range(BATCHES):


            xb = 2 * rng.random((BATCH_SIZE, INPUT_DIM)) - 1
            for val in xb.ravel():
                f.write(f"{val:.5f}\n")

            yb = np.zeros((BATCH_SIZE, OUTPUT_DIM))
            yb[np.arange(BATCH_SIZE),
               rng.integers(0, OUTPUT_DIM, BATCH_SIZE)] = 1
            for bit in yb.ravel():
                f.write(f"{bit:.0f}\n")   # 0 or 1

    print(f"saved {TOTAL_LINES:,} numbers to {p0}")

    # make empty input files 
    for pid in (1, 2):
        Path(out_dir, f"Input-P{pid}-0").touch(exist_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="super-basic SecureNN input generator")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed (default 42)")
    ap.add_argument("--dir", default="Player-Data",
                    help="where to drop Input-P0-0 (default Player-Data)")
    args = ap.parse_args()

    dump_all_the_numbers(args.seed, args.dir)


if __name__ == "__main__":
    main()
