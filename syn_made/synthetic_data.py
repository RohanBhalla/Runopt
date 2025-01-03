import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from noise import pnoise2
import gstools as gs
import plotly.graph_objects as go

# -----------------------------------------
# Generate Perlin Noise terrain
# -----------------------------------------
def generate_perlin_terrain(num_points, params, x_range=(0, 10), y_range=(0, 10)):
    """
    Generates a 2D Perlin noise field over [x_range, y_range], normalized to [-1, 1].
    
    :param num_points: Number of points in each dimension (int)
    :param params: Dictionary containing noise parameters (scale, octaves, etc.)
    :param x_range: Tuple specifying the x-min and x-max
    :param y_range: Tuple specifying the y-min and y-max
    :return: (X, Y, Z) where X, Y are the meshgrid, Z is normalized in [-1, 1]
    """
    scale = params['scale']
    octaves = params['octaves']
    persistence = params['persistence']
    lacunarity = params['lacunarity']
    
    # Create a grid
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    # Generate Perlin noise
    Z = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = pnoise2(
                X[i, j] / scale,
                Y[i, j] / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=999999,
                repeaty=999999,
                base=42
            )
    
    # Normalize to [-1, 1]
    z_min, z_max = Z.min(), Z.max()
    if z_max - z_min < 1e-8:
        Z_norm = np.zeros_like(Z)
    else:
        Z_norm = 2.0 * (Z - z_min) / (z_max - z_min) - 1.0
    
    return X, Y, Z_norm

# -----------------------------------------
# Generate Gaussian Random Field terrain
# -----------------------------------------
def generate_grf_terrain(num_points, x_range=(0, 10), y_range=(0, 10), len_scale=1.0):
    """
    Generates a 2D Gaussian Random Field using gstools, normalized to [-1, 1].
    
    :param num_points: Number of points in each dimension
    :param x_range: Tuple specifying the x-min and x-max
    :param y_range: Tuple specifying the y-min and y-max
    :param len_scale: Correlation length scale for the GRF
    :return: (X, Y, Z) with Z in [-1, 1]
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    # Define a variogram model (Exponential as an example)
    model = gs.Exponential(dim=2, var=1.0, len_scale=len_scale)
    srf = gs.SRF(model, seed=42)
    
    # Generate the GRF on the structured grid
    Z = srf.structured([x, y])  # shape = (ny, nx)
    
    # Normalize Z to [-1, 1]
    z_min, z_max = Z.min(), Z.max()
    if z_max - z_min < 1e-8:
        Z_norm = np.zeros_like(Z)
    else:
        Z_norm = 2.0 * (Z - z_min) / (z_max - z_min) - 1.0
    
    return X, Y, Z_norm

# -------------------------------------------------------------------
# STEP 1: Choose Topography Type and Grid Size
# -------------------------------------------------------------------
print("\nDefine Grid Dimensions:")
grid_size = float(input("Enter the grid size in meters (e.g., 1000 for 1km x 1km): "))
resolution = int(input("Enter the grid resolution (number of points per side, e.g., 100): "))

topography_options = {
    '1': 'Hill',
    '2': 'Valley',
    '3': 'Plateau',
    '4': 'Mountain Range',
    '5': 'Depression',
    '6': 'Plains'
}

print("\nSelect the type of topography to generate:")
for key, value in topography_options.items():
    print(f"{key}. {value}")

choice = input("Enter the number corresponding to your choice: ")
selected_topography = topography_options.get(choice, 'Hill')  # Default to 'Hill' if invalid input

# -------------------------------------------------------------------
# STEP 2: Define Terrain Generation Parameters Based on Selection
# -------------------------------------------------------------------
def set_parameters(topography_type):
    """
    Returns a dictionary of parameters for Perlin noise (scale, octaves, etc.)
    and a length scale for the Gaussian Random Field (GRF).
    """
    if topography_type == 'Hill':
        return {
            'scale': 50.0,
            'octaves': 4,
            'persistence': 0.6,
            'lacunarity': 2.5,
            'grf_len_scale': 2.0
        }
    elif topography_type == 'Valley':
        return {
            'scale': 80.0,
            'octaves': 5,
            'persistence': 0.5,
            'lacunarity': 2.0,
            'grf_len_scale': 3.0
        }
    elif topography_type == 'Plateau':
        return {
            'scale': 100.0,
            'octaves': 3,
            'persistence': 0.4,
            'lacunarity': 2.2,
            'grf_len_scale': 1.5
        }
    elif topography_type == 'Mountain Range':
        return {
            'scale': 40.0,
            'octaves': 7,
            'persistence': 0.7,
            'lacunarity': 2.8,
            'grf_len_scale': 1.0
        }
    elif topography_type == 'Depression':
        return {
            'scale': 90.0,
            'octaves': 6,
            'persistence': 0.5,
            'lacunarity': 2.3,
            'grf_len_scale': 2.5
        }
    elif topography_type == 'Plains':
        return {
            'scale': 300.0,
            'octaves': 1,
            'persistence': 0.2,
            'lacunarity': 2.0,
            'grf_len_scale': 10.0
        }

params = set_parameters(selected_topography)

# -------------------------------------------------------------------
# STEP 3: Define the grid
# -------------------------------------------------------------------
num_points = resolution
x = np.linspace(0, grid_size, num_points)
y = np.linspace(0, grid_size, num_points)

# Create a mesh for evaluation
X_synth, Y_synth = np.meshgrid(x, y)
X_flat = X_synth.flatten()
Y_flat = Y_synth.flatten()

# -------------------------------------------------------------------
# STEP 4: Generate base terrain
# -------------------------------------------------------------------
# Perlin noise component
X_perlin, Y_perlin, z_perlin = generate_perlin_terrain(
    num_points,
    params,
    x_range=(0, grid_size),
    y_range=(0, grid_size)
)

# GRF component
z_grf = generate_grf_terrain(
    num_points,
    x_range=(0, grid_size),
    y_range=(0, grid_size),
    len_scale=params['grf_len_scale']
)[2]  # Only the Z values

# Blend Perlin and GRF by topography
if selected_topography in ['Mountain Range', 'Hill']:
    perlin_weight = 0.7
    grf_weight = 0.3
elif selected_topography in ['Valley', 'Depression']:
    perlin_weight = 0.4
    grf_weight = 0.6
elif selected_topography == 'Plains':
    perlin_weight = 0.2
    grf_weight = 0.8
else:  # Plateau
    perlin_weight = 0.5
    grf_weight = 0.5

z_base = perlin_weight * z_perlin + grf_weight * z_grf

# Add small random noise for detail
noise_strength = 0.02
z_synth_final = z_base + noise_strength * np.random.randn(*z_base.shape)

# Scale the heights
height_scales = {
    'Hill': 100,            # ~100m high hills
    'Valley': -150,         # ~150m deep valley
    'Plateau': 500,         # ~500m high plateau
    'Mountain Range': 2000, # ~2000m high mountains
    'Depression': -300,     # ~300m deep depression
    'Plains': 10
}

max_abs_val = np.max(np.abs(z_synth_final))
if max_abs_val < 1e-8:
    z_synth_final[:] = 0
else:
    z_synth_final *= (height_scales[selected_topography] / max_abs_val)

# -------------------------------------------------------------------
# STEP 5: Create and save the synthetic dataset
# -------------------------------------------------------------------
df_synth = pd.DataFrame({
    'X': X_flat,
    'Y': Y_flat,
    'Z (Synthetic)': z_synth_final.flatten()
})

output_filename = 'synthetic_topography.csv'
df_synth.to_csv(output_filename, index=False)
print(f"\nSynthetic {selected_topography} topography saved to '{output_filename}'")

# -------------------------------------------------------------------
# STEP 6: Visualization with Plotly
#      - Custom color scales / camera angles for each topography
# -------------------------------------------------------------------
# Define custom color scales (Plotly built-ins or named ones)
color_scales = {
    'Hill': 'Greens',
    'Valley': 'Blues',
    'Plateau': 'Earth',
    'Mountain Range': 'Picnic',
    'Depression': 'Viridis',
    'Plains': 'YlOrBr'
}
chosen_colorscale = color_scales.get(selected_topography, 'terrain')  # fallback

# Define custom camera positions (just examples)
camera_settings = {
    'Hill': dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    'Valley': dict(eye=dict(x=-1.2, y=1.2, z=1.4)),
    'Plateau': dict(eye=dict(x=2.0, y=2.0, z=2.0)),
    'Mountain Range': dict(eye=dict(x=1.8, y=1.8, z=1.8)),
    'Depression': dict(eye=dict(x=1.0, y=-1.4, z=1.2)),
    'Plains': dict(eye=dict(x=1.5, y=1.0, z=1.0))
}
chosen_camera = camera_settings.get(selected_topography, dict(x=1.5, y=1.5, z=1.2))

fig = go.Figure(data=[
    go.Surface(
        x=X_synth,
        y=Y_synth,
        z=z_synth_final.reshape(X_synth.shape),
        colorscale=chosen_colorscale,
        contours={
            "z": {
                "show": True, 
                "start": np.min(z_synth_final), 
                "end": np.max(z_synth_final), 
                "size": (np.max(z_synth_final) - np.min(z_synth_final)) / 8,
                "color":"black"
            }
        }
    )
])

fig.update_layout(
    title=f'Synthetic {selected_topography} Topography',
    scene=dict(
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        zaxis_title='Elevation (meters)',
        camera=chosen_camera
    ),
    width=1000,
    height=800
)

fig.show()
