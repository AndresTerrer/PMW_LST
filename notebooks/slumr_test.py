import numpy as np
import matplotlib.pyplot as plt

# Generar matriz aleatoria
matriz = np.random.rand(30, 30)

# Guardar matriz como imagen
plt.imshow(matriz, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.savefig("matriz_aleatoria.png")
plt.show()
