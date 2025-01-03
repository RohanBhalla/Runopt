import numpy as np
import pandas as pd
from noise import pnoise2
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# STEP 1: Define the Grid for Synthetic Data
# ----------------------------------------------------------------------------
num_points = 100  # Grid resolution
x = np.linspace(0, 10, num_points)
y = np.linspace(0, 10, num_points)
X, Y = np.meshgrid(x, y)

# Parameters for Perlin Noise
scale = 100.0       # Determines the level of detail
octaves = 6         # Number of layers of noise
persistence = 0.5   # Amplitude of each octave
lacunarity = 2.0    # Frequency of each octave

# Generate Perlin Noise-based Z values
Z = np.zeros((num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        Z[i][j] = pnoise2(X[i][j] / scale,
                          Y[i][j] / scale,
                          octaves=octaves,
                          persistence=persistence,
                          lacunarity=lacunarity,
                          repeatx=1024,
                          repeaty=1024,
                          base=42)  # 'base' can be changed for different terrains

# Normalize Z to a desired range (e.g., 0 to 1000 meters)
Z_min, Z_max = np.min(Z), np.max(Z)
Z_normalized = 1000 * (Z - Z_min) / (Z_max - Z_min)

# ----------------------------------------------------------------------------
# STEP 2: Create Synthetic DataFrame
# ----------------------------------------------------------------------------
df_synth = pd.DataFrame({
    'X': X.flatten(),
    'Y': Y.flatten(),
    'Z (Synthetic)': Z_normalized.flatten()
})

# Save to CSV
df_synth.to_csv('/Users/akshat/Documents/WORK/Github/Runopt/synthetic_topography_perlin.csv', index=False)

# ----------------------------------------------------------------------------
# STEP 3: Visualization
# ----------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z_normalized, cmap='terrain')
plt.colorbar(contour, label='Elevation (meters)')
plt.title('Synthetic Topography Generated Using Perlin Noise')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()