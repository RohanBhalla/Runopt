import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.fft import fft2, ifft2, fftshift
from scipy.stats import norm
import gstools as gs  # Add gstools import

# ----------------------------------------------------------------------------
# STEP 1: Read in your real dataset
# ----------------------------------------------------------------------------
# Assume your data is stored in 'real_data.csv' with columns: X, Y, Z
df_real = pd.read_excel('/teamspace/studios/this_studio/InputFile_NEW.xlsx')

# Extract NumPy arrays
x_real = df_real['X'].values
y_real = df_real['Y'].values
z_real = df_real['Z (Existing)'].values

# ----------------------------------------------------------------------------
# STEP 1.1: Sample the real dataset to reduce memory usage
# ----------------------------------------------------------------------------
sample_size = 10000  # Adjust this number based on your memory constraints
if len(x_real) > sample_size:
    indices = np.random.choice(len(x_real), sample_size, replace=False)
    x_real = x_real[indices]
    y_real = y_real[indices]
    z_real = z_real[indices]

# ----------------------------------------------------------------------------
# STEP 2: Fit an RBF interpolator
# ----------------------------------------------------------------------------
# - function='thin_plate' or 'multiquadric' are common choices
# - Adjust 'epsilon' or 'smooth' to tweak how tightly/loosely it fits.
rbf_func = Rbf(x_real, y_real, z_real, function='multiquadric', smooth=1)

# ----------------------------------------------------------------------------
# STEP 3: Define a new grid where you'll generate synthetic data
# ----------------------------------------------------------------------------
# First, calculate the min and max values from real data
x_min, x_max = np.min(x_real), np.max(x_real)
y_min, y_max = np.min(y_real), np.max(y_real)

# Then calculate the expanded boundaries
margin_factor = 0.1  # 10% extension on each side
x_range = x_max - x_min
y_range = y_max - y_min
x_min_ext = x_min - margin_factor * x_range
x_max_ext = x_max + margin_factor * x_range
y_min_ext = y_min - margin_factor * y_range
y_max_ext = y_max + margin_factor * y_range

num_points = 100  # Increased number of points per dimension for higher resolution
x_synth = np.linspace(x_min_ext, x_max_ext, num_points)
y_synth = np.linspace(y_min_ext, y_max_ext, num_points)

# Create a mesh for evaluation
X_synth, Y_synth = np.meshgrid(x_synth, y_synth)
X_flat = X_synth.flatten()
Y_flat = Y_synth.flatten()

# ----------------------------------------------------------------------------
# STEP 4: Evaluate the RBF to create synthetic Z
# ----------------------------------------------------------------------------
z_synth = rbf_func(X_flat, Y_flat)

# ----------------------------------------------------------------------------
# STEP 5: Synthetic Data Generation Methods
# ----------------------------------------------------------------------------
def generate_perlin_terrain(num_points, x_range=(0, 10), y_range=(0, 10)):
    """Generate terrain using Perlin noise"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    # Parameters for Perlin Noise
    scale = 100
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    
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
                             base=42)
    
    # Normalize Z
    Z_min, Z_max = np.min(Z), np.max(Z)
    Z_normalized = 1000 * (Z - Z_min) / (Z_max - Z_min)
    
    return X, Y, Z_normalized

def generate_grf_terrain(num_points, x_range=(0, 10), y_range=(0, 10)):
    """Generate terrain using Gaussian Random Fields"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    # GRF parameters
    model = gs.Spherical(dim=2, var=1.0, len_scale=1.0)
    srf = gs.SRF(model, seed=42)
    
    # Generate GRF values
    Z = srf.structured([x, y])
    
    # Normalize Z
    Z_min, Z_max = np.min(Z), np.max(Z)
    Z_normalized = 1000 * (Z - Z_min) / (Z_max - Z_min)
    
    return X, Y, Z_normalized

# Modify STEP 5 to use both methods
# Generate base terrain using RBF
z_synth_rbf = rbf_func(X_flat, Y_flat).reshape((num_points, num_points))

# Generate Perlin noise terrain
X_perlin, Y_perlin, z_perlin = generate_perlin_terrain(
    num_points, 
    x_range=(x_min_ext, x_max_ext),
    y_range=(y_min_ext, y_max_ext)
)

# Generate GRF terrain
X_grf, Y_grf, z_grf = generate_grf_terrain(
    num_points,
    x_range=(x_min_ext, x_max_ext),
    y_range=(y_min_ext, y_max_ext)
)

# Combine all terrain types with weights
terrain_weights = {
    'rbf': 0.5,
    'perlin': 0.3,
    'grf': 0.2
}

z_synth_combined = (
    terrain_weights['rbf'] * z_synth_rbf +
    terrain_weights['perlin'] * z_perlin +
    terrain_weights['grf'] * z_grf
)

# Add some random noise for additional variation
noise_strength = 0.05
z_synth_final = z_synth_combined + noise_strength * np.random.randn(*z_synth_combined.shape)

# Update the synthetic dataset creation
df_synth = pd.DataFrame({
    'X': X_flat,
    'Y': Y_flat,
    'Z (Synthetic)': z_synth_final.flatten()
})

# Save to CSV if desired
df_synth.to_csv('/teamspace/studios/this_studio/synthetic_data_new_2.csv', index=False)

# ----------------------------------------------------------------------------
# STEP 7: (Optional) Quick Visualization
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_flat, Y_flat, z_synth_final, c=z_synth_final, cmap='terrain', marker='o', s=10)
ax.set_title('Synthetic Topography')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Synthetic)')
plt.show()
