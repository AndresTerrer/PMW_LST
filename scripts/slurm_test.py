import numpy as np
import matplotlib.pyplot as plt

# Generate random matrix
matrix = np.random.rand(30, 30)

# Save and display the matrix as an image
plt.imshow(matrix, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.savefig("random_matrix.png")
plt.show()
