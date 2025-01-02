'''
num_points: Controls the resolution of your survey grid. Higher values give more detail but increase computation time
site_width/length: Physical dimensions of your site in feet. Adjust these to match your actual site dimensions
variance: Controls the amplitude of the features (higher = more dramatic elevation changes)
length_scale: Controls the spatial extent of features (higher = more gradual changes, lower = more rapid changes)
0: Mean of the noise (usually keep at 0)
0.02: Standard deviation - controls how "rough" the surface is at very small scales
elevation_range: Maximum elevation difference across the site
base_elevation: The lowest elevation of your site
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gstools as gs
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# STEP 1: Define the Grid for Synthetic Data
# ----------------------------------------------------------------------------
num_points = 100  # Reduced for typical site survey resolution
site_width = 100.0  # Site width in feet (typical construction site)
site_length = 250.0  # Site length in feet
x = np.linspace(0, site_width, num_points)
y = np.linspace(0, site_length, num_points)
X, Y = np.meshgrid(x, y)
grid = np.column_stack([X.ravel(), Y.ravel()])

# ----------------------------------------------------------------------------
# STEP 2: Generate Site Topography
# ----------------------------------------------------------------------------
def generate_terrain_layer(variance, len_scale):
    model = gs.Gaussian(dim=2, var=variance, len_scale=len_scale)
    srf = gs.SRF(model, seed=np.random.randint(1000))
    return srf.structured([x, y])

# Generate base terrain with gentle slopes
Z = np.zeros((num_points, num_points))

# Base terrain with gradual elevation changes
Z += generate_terrain_layer(1.0, 100.0)  # Large-scale gradual changes

# Add local variations (small hills, depressions)
Z += 0.3 * generate_terrain_layer(0.5, 30.0)  # Medium-scale features

# Add micro-topography
Z += 0.1 * generate_terrain_layer(0.2, 3.0)  # Small-scale variations

# Add slight random noise for soil irregularities
noise = np.random.normal(0, 0.02, Z.shape)
Z += noise

# Scale elevation to realistic site elevations (typical construction site variation 10-30 feet)
Z_min, Z_max = np.min(Z), np.max(Z)
elevation_range = 10.0  # 10 feet of elevation change across site
Z_normalized = elevation_range * (Z - Z_min) / (Z_max - Z_min)

# Add base elevation (typical site elevation)
base_elevation = 0.0  # Starting elevation in feet
Z_normalized += base_elevation

# Add a gentle slope from one end to the other (typical drainage requirement)
# slope = np.linspace(0, 5, num_points)  # 5 foot drop across length
# Z_normalized += slope[:, np.newaxis]  # Add slope along length

# Add a random slope component using gstools
slope_variation = generate_terrain_layer(0.5, 50.0)  # Parameters for slope randomness
slope_scale = 5.0  # Maximum slope variation in feet
slope_pattern = slope_scale * (slope_variation - np.min(slope_variation)) / (np.max(slope_variation) - np.min(slope_variation))

# Apply the random slope instead of linear slope
Z_normalized += slope_pattern

# ----------------------------------------------------------------------------
# STEP 3: Create Site Data with Metadata
# ----------------------------------------------------------------------------
df_site = pd.DataFrame({
    'Station_X_ft': X.flatten(),
    'Station_Y_ft': Y.flatten(),
    'Ground_Elevation_ft': Z_normalized.flatten()
})

# Add metadata
metadata = {
    'Site_Dimensions': f"{site_width:.0f} ft x {site_length:.0f} ft",
    'Survey_Grid_Resolution': f"{num_points}x{num_points} points",
    'Elevation_Range': f"{base_elevation:.1f}-{np.max(Z_normalized):.1f} ft",
    'Average_Slope': f"{(np.max(Z_normalized)-np.min(Z_normalized))/site_length*100:.2f}%",
    'Base_Elevation': f"{base_elevation:.1f} ft",
    'Survey_Type': 'Synthetic Ground Survey Data',
    'Coordinate_System': 'Local Grid (feet)',
    'Vertical_Datum': 'Local Benchmark'
}

# Save to CSV with metadata
with open('site_survey_metadata.txt', 'w') as f:
    for key, value in metadata.items():
        f.write(f"{key}: {value}\n")

df_site.to_csv(f'site_survey_data_{site_width}_{np.max(Z_normalized):.0f}_feet.csv', index=False)

# ----------------------------------------------------------------------------
# STEP 4: Site Visualization
# ----------------------------------------------------------------------------
fig = go.Figure(data=[go.Scatter3d(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z_normalized.flatten(),
    mode='markers',
    marker=dict(
        size=2,
        color=Z_normalized.flatten(),
        colorscale='earth',
        opacity=0.8,
        colorbar=dict(
            title='Ground Elevation (ft)',
            tickformat='.1f'
        )
    )
)])

fig.update_layout(
    title={
        'text': 'Site Topography Survey',
        'y':0.95
    },
    scene=dict(
        xaxis_title='Station X (ft)',
        yaxis_title='Station Y (ft)',
        zaxis_title='Ground Elevation (ft)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        aspectratio=dict(x=1, y=1.5, z=0.3)  # Adjusted for site proportions
    ),
    width=1200,
    height=900
)

# Add contour lines on the surface
contour_intervals = 1.0  # 1 foot contour intervals
fig.add_trace(go.Surface(
    x=X,
    y=Y,
    z=Z_normalized,
    opacity=0.7,
    contours=dict(
        z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True,
            width=2
        )
    ),
    showscale=False
))

fig.show()