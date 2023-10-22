import numpy as np
import matplotlib.pyplot as plt

FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1
NUM_EPOCHS = 3000
NUM_DECAY_EPOCHS = 2500

def f(e):
    return FINAL_EPSILON + (max(NUM_EPOCHS/2 - e, 0) * (INITIAL_EPSILON - FINAL_EPSILON) / NUM_DECAY_EPOCHS)

x = np.arange(0, NUM_EPOCHS, 1)
y = [f(x) for x in range(NUM_EPOCHS)]

plt.plot(x, y)
plt.xlabel("Epoch")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay")
plt.show()