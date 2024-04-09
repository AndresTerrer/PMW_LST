import numpy as np
import matplotlib.pyplot as plt

# Generate random matrix
matriz = np.random.rand(30, 30)

# Save and display the matrix as an image
plt.imshow(matriz, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.savefig("random_matrix.png")
plt.show()
