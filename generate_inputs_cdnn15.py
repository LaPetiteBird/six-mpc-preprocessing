import numpy as np
import os

# values 
n = 20
m = 5
f = 16
scale = 2**f  

# make up some random data
X = np.random.uniform(-1, 1, size=(n,m))
y = np.random.uniform(-1, 1, size=n)

# solve the thing to see if it works
beta = np.linalg.solve(X.T @ X, X.T @ y)
print("Expected coefficients (plaintext):")
for idx, val in enumerate(beta):
    print("beta[%d] = %.6f" % (idx, val))

# split stuff up randomly to share it
Xpart0 = np.random.uniform(-0.5, 0.5, size=(n,m))
Xpart1 = X - Xpart0
y0 = np.random.uniform(-0.5, 0.5, size=n)
y1 = y - y0

# turn into ints with fake decimals
def encode_fp(matrixThing):
    return np.round(matrixThing * scale).astype(np.int64)

X0_fixed = encode_fp(Xpart0)
X1_fixed = encode_fp(Xpart1)
y0_fixed = encode_fp(y0)
y1_fixed = encode_fp(y1)

# save files for the MPC to read 
os.makedirs("Player-Data", exist_ok=True)

def save_inputs(fileName, Xmatrix, yvec):
    with open(fileName, "w") as f:
        for row in Xmatrix:
            f.write(" ".join(map(str, row)) + "\n")
        for val in yvec:
            f.write(str(val) + "\n")

save_inputs("Player-Data/Input-P0-0", X0_fixed, y0_fixed)
save_inputs("Player-Data/Input-P1-0", X1_fixed, y1_fixed)

print("inputs written to Player-Data/")
