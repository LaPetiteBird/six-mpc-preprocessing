import numpy as np
import os

# Parameters (must match .mpc file)
BATCH = 2
IN_H, IN_W, IN_C = 4, 4, 1
K_H, K_W = 2, 2
OUT_C = 2

OUT_H = IN_H - K_H + 1
OUT_W = IN_W - K_W + 1
N_WINDOWS = BATCH * OUT_H * OUT_W
FILTER_SIZE = K_H * K_W * IN_C

def im2col(images, k_h, k_w):
    B, H, W, C = images.shape
    out_h = H - k_h + 1
    out_w = W - k_w + 1
    patches = []
    for b in range(B):
        for i in range(out_h):
            for j in range(out_w):
                patch = images[b, i:i+k_h, j:j+k_w, :].reshape(-1)
                patches.append(patch)
    return np.array(patches)

def write_input_text(path, matrix):
    flat = matrix.flatten()
    with open(path, "w") as f:
        f.write(" ".join(str(int(v)) for v in flat) + "\n")

def main():
    os.makedirs("Player-Data", exist_ok=True)

    # Generate input tensor (images)
    x = np.random.randint(0, 10, size=(BATCH, IN_H, IN_W, IN_C), dtype=np.int64)
    x_patches = im2col(x, K_H, K_W)  # shape: (N_WINDOWS, FILTER_SIZE)

    # Generate weights
    W = np.random.randint(0, 10, size=(FILTER_SIZE, OUT_C), dtype=np.int64)

    print(f"✅ X shape (im2col): {x_patches.shape}, W shape: {W.shape}")

    # Write as plain text
    write_input_text("Player-Data/Input-P0-0", x_patches)
    write_input_text("Player-Data/Input-P1-0", W)

    print("Input files written to Player-Data/")

if __name__ == "__main__":
    main()
