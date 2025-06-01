import numpy as np
import sys, os

def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    d   = int(sys.argv[1])
    rng = np.random.default_rng(int(sys.argv[2]))

    X = rng.integers(0, 100, size=(d, d), dtype=np.int64)
    Y = rng.integers(0, 100, size=(d, d), dtype=np.int64)

    os.makedirs("Player-Data", exist_ok=True)
    with open("Player-Data/Input-P0-0", "w") as f:
        for val in np.concatenate((X.flatten(), Y.flatten())):
            f.write(f"{int(val)}\n")        

    open("Player-Data/Input-P1-0", "w").close()

    print("Inputs written for Party 0.")
    print("X shape:", X.shape, "Y shape:", Y.shape)

if __name__ == "__main__":
    main()
