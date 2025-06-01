import numpy as np
import os

# Match MPC program
ROWS = 64
INNER = 128
COLS = 32
SCALE = 2 ** 16
PRIME = 2**64 - 59  # Should match modulus used with -F 64 

def encode_fixed_point(mat):
    return np.round(mat * SCALE).astype(np.int64) % PRIME

def write_matrix(path, matrices):
    with open(path, "w") as f:
        for mat in matrices:
            for row in mat:
                f.write(" ".join(map(str, row)) + "\n")

def main():
    os.makedirs("Player-Data", exist_ok=True)

    A = np.random.uniform(-1, 1, size=(ROWS, INNER))
    B = np.random.uniform(-1, 1, size=(INNER, COLS))

    A0 = encode_fixed_point(A)
    B1 = encode_fixed_point(B)

    write_matrix("Player-Data/Input-P0-0", [A0])
    write_matrix("Player-Data/Input-P1-0", [B1])

    print("✓ Wrote input matrices for A and B")

if __name__ == "__main__":
    main()
