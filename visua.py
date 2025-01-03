import plotly.express as px
import pandas as pd

# Read the CSV file with the correct header
data = pd.read_csv('/Users/akshat/Documents/WORK/Github/Runopt/syn_made/synthetic_Flat_100x100.csv')

# Convert Z values to numeric
data['Z'] = pd.to_numeric(data['Z'])

# Create an interactive 3D scatter plot using Plotly
fig = px.scatter_3d(data, 
                    x='X', 
                    y='Y', 
                    z='Z',
                    color='Z',  # Color points based on elevation
                    color_continuous_scale='earth',  # Use terrain-like colorscale
                    title='Synthetic Topography')

# Update the layout for better visualization
fig.update_layout(
    scene = dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z (Elevation)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)  # Adjust the camera view
        )
    ),
    width=1000,
    height=800
)

# Show the interactive plot
fig.show()