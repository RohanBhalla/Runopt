import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging
from datetime import datetime
import os

from shapely.geometry import Point, Polygon, box
from shapely.affinity import rotate, translate, scale
from mpl_toolkits.mplot3d import Axes3D
from slope_stability import slope_stability_calculation
from multiprocessing import Pool, cpu_count


def setup_logging():
    """
    Set up logging configuration to write to both file and console.
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a timestamp for the log file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/runopt_debug_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    
    logging.info(f"Logging started. Log file: {log_file}")
    return log_file


def read_excel_to_dataframe(file_path):
    """
    Reads an Excel file with a single sheet and returns it as a DataFrame.
    :param file_path: Path to the Excel file.
    :return: DataFrame containing the data from the Excel file.
    """
    df = pd.read_excel(file_path)
    columns_to_keep = ['X', 'Y', 'Z (Existing)']

    df = df.loc[:, columns_to_keep]
    # Convert column names to lowercase immediately after reading
    df.columns = df.columns.str.lower()

    return df

# Example usage:
# df = read_excel_to_dataframe('Site Example.xlsx')


def create_building(length, width):
    """
    Creates a rectangle shape using the given length and width.
    :return: A shapely box object representing the rectangle.
    """
    rectangle_shape = box(0, 0, width, length)
    return rectangle_shape

def create_building_dataframe(buildings_json):
    """
    Creates a DataFrame from a list of JSON objects containing building information.
    
    :param buildings_json: List of JSON objects with 'building_name', 'length', and 'width'.
    :return: DataFrame with columns 'Building Name', 'Length', and 'Width'.
    """
    # Extract data from JSON objects
    data = [{'Building Name': building['building_name'], 
             'Length': building['length'], 
             'Width': building['width']} for building in buildings_json]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


#Limiting offset/boundary inside the main site itself
#Arbitrary Percentage coverage or Region selection from within
def create_confined_region(polygon, percentage):
    # Ensure the percentage is expressed as a decimal
    scale_factor = percentage / 100.0

    # Get the centroid of the polygon to scale around it
    centroid = polygon.centroid

    # Scale the polygon by the specified factor
    confined_region = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin=centroid)

    return confined_region


#Place building inside of the Confined Region defined above
def find_valid_placements(site_polygon, building_polygon, rotations=4, steps=10):
    valid_placements = []
    min_x, min_y, max_x, max_y = site_polygon.bounds

    # Get building dimensions from the building_polygon bounds
    building_min_x, building_min_y, building_max_x, building_max_y = building_polygon.bounds
    building_width = building_max_x - building_min_x
    building_length = building_max_y - building_min_y

    # Create rotation angles
    rotation_angles = np.linspace(0, 360, rotations, endpoint=False)

    # Create grid of translation steps
    x_steps = np.linspace(min_x, max_x - building_width, steps)
    y_steps = np.linspace(min_y, max_y - building_length, steps)

    # Iterate over each rotation
    for angle in rotation_angles:
        rotated_building = rotate(building_polygon, angle, origin=(0, 0))

        # Iterate over each translation step
        for x in x_steps:
            for y in y_steps:
                translated_building = translate(rotated_building, xoff=x, yoff=y)

                # Check if the translated and rotated building fits within the site
                if site_polygon.contains(translated_building):
                    valid_placements.append(translated_building)

    return valid_placements




def plot_3d_surface_grid(df, x_col='X', y_col='Y', z_col='Z (Existing)', title='Surface Plot of the Construction Site'):
    """
    Plots a 3D surface from a DataFrame using a grid.
    
    :param df: DataFrame containing the data.
    :param x_col: Column name for X coordinates.
    :param y_col: Column name for Y coordinates.
    :param z_col: Column name for Z coordinates.
    :param title: Title of the plot.
    """
    X = df[x_col].values
    Y = df[y_col].values
    Z = df[z_col].values

    # Create a grid for X and Y
    X_grid, Y_grid = np.meshgrid(np.unique(X), np.unique(Y))
    
    # Reshape Z to match the grid shape
    Z_grid = Z.reshape(X_grid.shape)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot
    ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none')

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(z_col)
    ax.set_title(title)

    # Show the plot
    plt.show()

# Example usage:
# plot_3d_surface_grid(RandomSurfaceDF)




# CUT AND FILL Work 
def extend_building_region(building_placement, extension_percentage):
    """
    Extend the perimeter of the building region by a given percentage.
    """
    extended_region = building_placement.buffer(extension_percentage)
    return extended_region

def find_points_in_extended_region(gdf, extended_region, spatial_index):
    """
    Optimized: Use GeoPandas with spatial indexing to find points within the extended region.
    """
    # Use spatial index to filter possible points within the bounding box
    possible_matches_index = list(spatial_index.intersection(extended_region.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    
    # Precise containment check using vectorized operations
    points_in_region = possible_matches[possible_matches.within(extended_region)]
    
    return points_in_region

def calculate_cut_fill_from_grid(relevant_points_df, proposed_elevation):
    """
    Calculate cut and fill volumes using grid-based method for a given set of points and proposed elevation.
    """
    if relevant_points_df.empty:
        print("No points found within the relevant area.")
        return None

    # Ensure column names are lowercase
    relevant_points_df.columns = relevant_points_df.columns.str.lower()

    # Calculate delta Z (difference between proposed and existing elevations)
    delta_z = proposed_elevation - relevant_points_df['z (existing)']

    # Calculate grid cell area using lowercase column names
    grid_cell_area = (relevant_points_df['x'].max() - relevant_points_df['x'].min()) * \
                     (relevant_points_df['y'].max() - relevant_points_df['y'].min()) / len(relevant_points_df)

    # Calculate cut and fill volumes
    cut_volume = np.sum(np.maximum(0, -delta_z)) * grid_cell_area  # Cut (existing elevation > proposed)
    fill_volume = np.sum(np.maximum(0, delta_z)) * grid_cell_area   # Fill (proposed elevation > existing)

    # Structure of the results being returned 
    return {
        'cut_volume': cut_volume,
        'fill_volume': fill_volume,
        'relevant_points': relevant_points_df
    }

def process_single_placement(args):
    idx, placement, gdf, spatial_index, extension_percentage, z_min, z_max, z_step = args
    logging.info(f"\nProcessing building {idx + 1}...")
    
    extended_region = extend_building_region(placement, extension_percentage)
    relevant_points_gdf = find_points_in_extended_region(gdf, extended_region, spatial_index)
    
    logging.info(f"Relevant Points Count: {len(relevant_points_gdf)}")
    
    if relevant_points_gdf.empty:
        logging.warning(f"Skipping building placement {idx + 1}: No points found in the extended region.")
        return None
    
    proposed_zs = np.arange(z_min, z_max + z_step, z_step)
    logging.info(f"Elevation range: {z_min:.2f} to {z_max:.2f}, step: {z_step}")
    
    # Vectorized cut-fill calculations
    cut_fill_results = calculate_cut_fill_vectorized(relevant_points_gdf, proposed_zs)
    
    min_cost_idx = np.argmin(cut_fill_results['total_cost'])
    min_cost_z = proposed_zs[min_cost_idx]
    min_cost = cut_fill_results['total_cost'][min_cost_idx]
    
    logging.info(f"Building {idx + 1} Results:")
    logging.info(f"Best Z = {min_cost_z:.2f}")
    logging.info(f"Cut Volume = {cut_fill_results['cut_volume'][min_cost_idx]:.2f}")
    logging.info(f"Fill Volume = {cut_fill_results['fill_volume'][min_cost_idx]:.2f}")
    logging.info(f"Total Cost = {min_cost:.2f}")
    
    return (placement, {
        'best_z': min_cost_z,
        'min_cost': min_cost,
        'cut_volume': cut_fill_results['cut_volume'][min_cost_idx],
        'fill_volume': cut_fill_results['fill_volume'][min_cost_idx],
        'relevant_points': relevant_points_gdf,
        'all_cut_fill_by_z': {
            z: {
                'cut_volume': cv,
                'fill_volume': fv,
                'cut_cost': cv * 143,
                'fill_cost': fv * 144,
                'total_cost': cv * 143 + fv * 144
            } for z, cv, fv in zip(
                proposed_zs,
                cut_fill_results['cut_volume'],
                cut_fill_results['fill_volume']
            )
        }
    })

def calculate_optimum_cut_fill(building_positions, surface_df, extension_percentage, z_min, z_max, z_step):
    # Ensure column names are lowercase at the start
    surface_df.columns = surface_df.columns.str.lower()
    # Verify required columns are present
    verify_required_columns(surface_df)
    
    # Convert surface_df to GeoDataFrame for spatial operations
    gdf = gpd.GeoDataFrame(surface_df, geometry=gpd.points_from_xy(surface_df['x'], surface_df['y']))
    spatial_index = gdf.sindex
    
    # Initialize lists to store results
    initial_results = {}
    unstable_points = []
    
    # Cost variables
    unclassified_excavation_cost = 143
    select_granular_fill = 144
    
    # Prepare arguments for parallel processing
    args = [
        (idx, placement, gdf, spatial_index, extension_percentage, z_min, z_max, z_step)
        for idx, placement in enumerate(building_positions)
    ]
    
    # Use multiprocessing Pool to process placements in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_placement, args)
    
    # Filter out None results and populate initial_results
    for result in results:
        if result:
            placement, data = result
            initial_results[placement] = data
    
    # Sort placements by initial cost and get top 10
    top_10_placements = sorted(initial_results.items(), key=lambda x: x[1]['min_cost'])[:10]
    
    # Final results dictionary
    optimum_results = {}
    
    # Second pass: Analyze slope stability only for top 10 placements
    print("\nAnalyzing slope stability for top 10 placements...")
    for rank, (placement, initial_data) in enumerate(top_10_placements, 1):
        print(f"\nAnalyzing slope stability for rank {rank} placement...")
        
        # Get the optimal Z level and relevant points from initial analysis
        min_cost_z = initial_data['best_z']
        relevant_points_df = initial_data['relevant_points']
        all_cut_fill_by_z = initial_data['all_cut_fill_by_z']
        
        # Vectorized slope stability analysis
        slope_data = prepare_slope_data(relevant_points_df, min_cost_z)
        stability_results, _ = slope_stability_calculation(slope_data)
        
        # Identify unstable points
        failed_stability = stability_results['Factor of Safety'] < 1.5
        if failed_stability.any():
            failed_points = relevant_points_df[failed_stability].copy()
            failed_points = failed_points.assign(
                Height_Difference=abs(min_cost_z - failed_points['z (existing)']),
                Factor_of_Safety=stability_results['Factor of Safety'][failed_stability],
                Building_Rank=rank,
                Proposed_Z=min_cost_z
            )
            unstable_points.append(failed_points[['x', 'y', 'z (existing)', 'Height_Difference', 'Factor_of_Safety', 'Building_Rank', 'Proposed_Z']])
        
        # Store final results without wall calculations
        optimum_results[placement] = {
            'best_z': min_cost_z,
            'cut_volume': all_cut_fill_by_z[min_cost_z]['cut_volume'],
            'fill_volume': all_cut_fill_by_z[min_cost_z]['fill_volume'],
            'min_cost': initial_data['min_cost'],
            'initial_rank': rank,
            'all_cut_fill_by_z': all_cut_fill_by_z
        }
    
    # Concatenate all unstable points at once
    if unstable_points:
        unstable_points_df = pd.concat(unstable_points, ignore_index=True)
    else:
        unstable_points_df = pd.DataFrame(columns=[
            'x', 'y', 'z (existing)', 'Height_Difference', 
            'Factor_of_Safety', 'Building_Rank', 
            'Proposed_Z'
        ])
    
    return optimum_results, unstable_points_df



# Sort results based on net_volume
def sort_results_by_net_volume(optimum_results):
    # Sort the dictionary by 'net_volume' in ascending order
    sorted_results = sorted(optimum_results.items(), key=lambda x: x[1]['net_volume'])
    return sorted_results




                        # # Example usage:
                        # z_min = NewSurfaceDF['Z (Existing)'].min()
                        # z_max = NewSurfaceDF['Z (Existing)'].max()
                        # z_step = 0.5  # Example Z step value
                        # extension_percentage = 0.10  # Extend building region by 10%

                        # optimum_cut_fill_results = calculate_optimum_cut_fill(
                        #     valid_building_positions,
                        #     NewSurfaceDF,
                        #     extension_percentage,
                        #     z_min,
                        #     z_max,
                        #     z_step
                        # )

def create_cut_fill_dataframe(optimum_results):
    """
    Create a DataFrame from the optimum cut and fill results.
    Include building position information.
    """
    data = []
    
    for idx, (placement, results) in enumerate(optimum_results.items(), start=1):
        # Get building position coordinates
        centroid = placement.centroid
        bounds = placement.bounds
        
        building_number = idx
        z_value = results['best_z']
        cut_volume = results['cut_volume']
        fill_volume = results['fill_volume']
        cut_cost = cut_volume * 143
        fill_cost = fill_volume * 144
        total_cost = cut_cost + fill_cost
        
        data.append({
            'Building Number': building_number,
            'Z Value': z_value,
            'Cut Volume': cut_volume,
            'Fill Volume': fill_volume,
            'Cut Cost': cut_cost,
            'Fill Cost': fill_cost,
            'Total Cost': total_cost,
            'Center X': centroid.x,
            'Center Y': centroid.y,
            'Min X': bounds[0],
            'Min Y': bounds[1],
            'Max X': bounds[2],
            'Max Y': bounds[3],
            'Area': placement.area,
            'Rotation': calculate_rotation(placement)  # New helper function
        })
    
    df = pd.DataFrame(data)
    return df

def calculate_rotation(polygon):
    """
    Calculate the approximate rotation angle of a building polygon.
    """
    coords = list(polygon.exterior.coords)
    if len(coords) >= 2:
        # Calculate angle between first two points
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 360
    return 0

# Example usage:
# optimum_cut_fill_results = calculate_optimum_cut_fill(...)
# cut_fill_df = create_cut_fill_dataframe(optimum_cut_fill_results)
# print(cut_fill_df)
def plot_3d_terrain_plotly(surface_df, best_placement, best_z, title='Site Terrain with Optimal Building Placement'):
    """
    Create an interactive 3D plot using Plotly instead of Matplotlib
    """
    # Create the terrain scatter plot
    terrain = go.Scatter3d(
        x=surface_df['x'],
        y=surface_df['y'],
        z=surface_df['z (existing)'],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.6
        ),
        name='Terrain'
    )

    # Create the building footprint
    x, y = best_placement.exterior.xy
    building = go.Scatter3d(
        x=list(x),
        y=list(y),
        z=[best_z] * len(x),
        mode='lines',
        line=dict(
            color='red',
            width=4
        ),
        name='Best Building Position'
    )

    # Create the figure
    fig = go.Figure(data=[terrain, building])

    # Update the layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def plot_stability_results_plotly(relevant_points_df, unstable_points_df, building_placement, rank):
    """
    Create an interactive 3D plot showing stability analysis using Plotly
    """
    # Create the stable points scatter
    stable_points = go.Scatter3d(
        x=relevant_points_df['x'],
        y=relevant_points_df['y'],
        z=relevant_points_df['z (existing)'],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.3
        ),
        name='Stable Points'
    )

    # Create the unstable points scatter with color gradient
    if not unstable_points_df.empty:
        unstable_points = go.Scatter3d(
            x=unstable_points_df['x'],
            y=unstable_points_df['y'],
            z=unstable_points_df['z (existing)'],
            mode='markers',
            marker=dict(
                size=5,
                color=unstable_points_df['Factor_of_Safety'],
                colorscale='RdYlGn',
                opacity=0.8,
                colorbar=dict(title='Factor of Safety'),
                cmin=0.5,
                cmax=1.5
            ),
            name='Unstable Points'
        )
    else:
        unstable_points = None

    # Create the building footprint
    x, y = building_placement.exterior.xy
    z = [unstable_points_df['Proposed_Z'].iloc[0] if not unstable_points_df.empty else 0] * len(x)
    building = go.Scatter3d(
        x=list(x),
        y=list(y),
        z=z,
        mode='lines',
        line=dict(
            color='black',
            width=4,
            dash='dash'
        ),
        name='Building Footprint'
    )

    # Create the figure
    fig = go.Figure(data=[stable_points, building])
    if unstable_points is not None:
        fig.add_trace(unstable_points)

    # Update the layout
    fig.update_layout(
        title=f'Slope Stability Analysis Results - Rank {rank} Placement',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def main():
    # Set up logging at the start
    log_file = setup_logging()
    logging.info("Starting RunOpt optimization process")
    
    # Example file path for testing
    excel_file_path = '/Users/ronballer/Desktop/RunOpt/RunoptCode/InputFile NEW.xlsx'
    
    # Read data from Excel file
    surface_df = read_excel_to_dataframe(excel_file_path)
    logging.info(f"Columns in surface_df: {surface_df.columns.tolist()}")
    logging.info("Data loaded successfully from Excel")
    
    # Inspect elevation data
    logging.info("Elevation Data Statistics:")
    logging.info(f"\n{surface_df['z (existing)'].describe()}")
    
    # Create example building dimensions
    building_length = 30
    building_width = 30
    building = create_building(length=building_length, width=building_width)
    print(f"\nCreated building with dimensions: {building_length}m x {building_width}m")

    # Create site polygon from surface data bounds
    min_x, max_x = surface_df['x'].min(), surface_df['x'].max()
    min_y, max_y = surface_df['y'].min(), surface_df['y'].max()
    site_polygon = box(min_x, min_y, max_x, max_y)
    print("\nSite boundary created")
    print(site_polygon)

    # Create confined region (50% of total site area)
    confined_region = create_confined_region(site_polygon, percentage=50)
    print("Confined region created for building placement")

    # Find valid placements (increase steps for more granular results)
    valid_placements = find_valid_placements(confined_region, building, rotations=8, steps=20)
    print(f"\nFound {len(valid_placements)} valid building positions")

    # Calculate optimum cut and fill for each valid placement
    z_min = surface_df['z (existing)'].min()
    z_max = surface_df['z (existing)'].max()
    z_step = 0.5  # 0.5m intervals for elevation analysis
    extension_percentage = 0.40  # Extend building region by 40%

    print("\nCalculating optimum cut and fill volumes...")
    optimum_results, unstable_points_df = calculate_optimum_cut_fill(
        valid_placements,
        surface_df,
        extension_percentage,
        z_min,
        z_max,
        z_step
    )

    # Create summary DataFrame
    cut_fill_df = create_cut_fill_dataframe(optimum_results)
    
    # Display results with position information
    print("\nCut and Fill Analysis Summary:")
    summary_df = cut_fill_df.groupby('Building Number').agg({
        'Total Cost': 'min',
        'Cut Volume': lambda x: x[cut_fill_df.groupby('Building Number')['Total Cost'].transform('min') == cut_fill_df['Total Cost']].iloc[0],
        'Fill Volume': lambda x: x[cut_fill_df.groupby('Building Number')['Total Cost'].transform('min') == cut_fill_df['Total Cost']].iloc[0],
        'Z Value': lambda x: x[cut_fill_df.groupby('Building Number')['Total Cost'].transform('min') == cut_fill_df['Total Cost']].iloc[0],
        'Center X': 'first',
        'Center Y': 'first',
        'Rotation': 'first'
    })
    
    # Format the output
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("\nDetailed Building Analysis:")
    print(summary_df)
    
    # Create a more readable summary
    print("\nBuilding Positions Summary:")
    for idx, row in summary_df.iterrows():
        print(f"\nBuilding {idx}:")
        print(f"  Position: ({row['Center X']:.2f}, {row['Center Y']:.2f})")
        print(f"  Rotation: {row['Rotation']:.1f}°")
        print(f"  Elevation: {row['Z Value']:.2f}m")
        print(f"  Cut Volume: {row['Cut Volume']:.2f}m³")
        print(f"  Fill Volume: {row['Fill Volume']:.2f}m³")
        print(f"  Total Cost: ${row['Total Cost']:.2f}")

    if not unstable_points_df.empty:
        print("\nUnstable Points Found at:")
        print(unstable_points_df)

    # Plot 3D surface with best building placement
    best_placement = min(optimum_results.items(), key=lambda x: x[1]['min_cost'])[0]
    best_z = optimum_results[best_placement]['best_z']
    
    # Replace matplotlib plotting with Plotly
    print("\nGenerating interactive 3D plots...")
    
    # Create terrain plot
    terrain_fig = plot_3d_terrain_plotly(
        surface_df, 
        best_placement, 
        best_z, 
        'Site Terrain with Optimal Building Placement'
    )
    
    # Save the plot as HTML file
    terrain_fig.write_html("terrain_plot.html")
    
    # Open in browser (optional)
    terrain_fig.show()

    # Create stability analysis plot if there are unstable points
    if not unstable_points_df.empty:
        stability_fig = plot_stability_results_plotly(
            surface_df,  # Using full surface as relevant points
            unstable_points_df,
            best_placement,
            1  # Rank of the best placement
        )
        
        # Save the plot as HTML file
        stability_fig.write_html("stability_plot.html")
        
        # Open in browser (optional)
        stability_fig.show()

    print("\nPlots have been saved as HTML files.")

    logging.info("\nOptimization process completed")
    logging.info(f"Debug log has been saved to: {log_file}")

def calculate_cut_fill_vectorized(relevant_points_gdf, proposed_zs):
    """
    Vectorized calculation of cut and fill volumes for multiple Z levels.
    """
    existing_z = relevant_points_gdf['z (existing)'].values
    logging.debug(f"Proposed Elevations: {proposed_zs}")
    logging.debug(f"Existing Elevations Sample: {existing_z[:5]}")
    
    # Calculate grid cell area
    grid_area = calculate_grid_cell_area(relevant_points_gdf)
    logging.debug(f"Grid Cell Area: {grid_area}")
    
    # Calculate elevation differences
    delta_z = proposed_zs[:, np.newaxis] - existing_z
    logging.debug(f"Delta Z Sample (first 5 points, first 3 elevations):\n{delta_z[:3, :5]}")
    
    # Calculate volumes and costs
    cut_volume = np.maximum(0, -delta_z).sum(axis=1) * grid_area
    fill_volume = np.maximum(0, delta_z).sum(axis=1) * grid_area
    
    logging.debug("\nVolumes and Costs:")
    for i, z in enumerate(proposed_zs):
        logging.debug(f"Z={z:.2f}: Cut={cut_volume[i]:.2f}, Fill={fill_volume[i]:.2f}, "
                     f"Total Cost={(cut_volume[i] * 143 + fill_volume[i] * 144):.2f}")
    
    return {
        'cut_volume': cut_volume,
        'fill_volume': fill_volume,
        'total_cost': cut_volume * 143 + fill_volume * 144
    }

def calculate_grid_cell_area(relevant_points_gdf):
    """
    Calculate the area of each grid cell based on the DataFrame.
    """
    if len(relevant_points_gdf) == 0:
        return 0
    x_range = relevant_points_gdf['x'].max() - relevant_points_gdf['x'].min()
    y_range = relevant_points_gdf['y'].max() - relevant_points_gdf['y'].min()
    return (x_range * y_range) / len(relevant_points_gdf)

def prepare_slope_data(relevant_points_df, min_cost_z):
    """
    Prepare slope data in a vectorized manner.
    """
    height_diff = abs(min_cost_z - relevant_points_df['z (existing)'])
    slope_data = pd.DataFrame({
        'X': relevant_points_df['x'],
        'Y': relevant_points_df['y'],
        'Z': relevant_points_df['z (existing)'],
        'Slope Angle': 26.57,  # 2:1 slope
        'Height of slope': height_diff,
        'Friction Angle': 30,  # Assumed soil properties
        'Cohesion': 25,
        'Unit Weight': 20
    })
    return slope_data

# Add error handling for required columns
def verify_required_columns(df):
    """
    Verify that all required columns are present in the DataFrame.
    Raises KeyError if any required column is missing.
    """
    required_columns = ['x', 'y', 'z (existing)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Required columns {missing_columns} not found in DataFrame")

if __name__ == "__main__":
    main()