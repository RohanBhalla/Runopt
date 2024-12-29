import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, box
from shapely.affinity import rotate, translate, scale
from mpl_toolkits.mplot3d import Axes3D
from slope_stability import slope_stability_calculation


def read_excel_to_dataframe(file_path):
    """
    Reads an Excel file with a single sheet and returns it as a DataFrame.
    :param file_path: Path to the Excel file.
    :return: DataFrame containing the data from the Excel file.
    """
    df = pd.read_excel(file_path)
    columns_to_keep = ['X', 'Y', 'Z (Existing)']

    df = df.loc[:, columns_to_keep]

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

def find_points_in_extended_region(extended_region, surface_df):
    """
    Find the points in surface_df that are within the extended region using spatial indexing.
    Uses R-tree for efficient spatial querying.
    """
    # Convert column names to lowercase for consistency
    surface_df.columns = surface_df.columns.str.lower()
    
    # Get the bounds of the extended region to pre-filter points
    minx, miny, maxx, maxy = extended_region.bounds
    
    # First filter: Quick bounding box check
    mask = (
        (surface_df['x'] >= minx) & 
        (surface_df['x'] <= maxx) & 
        (surface_df['y'] >= miny) & 
        (surface_df['y'] <= maxy)
    )
    potential_points = surface_df[mask]
    
    if potential_points.empty:
        return potential_points
    
    # Second filter: Precise containment check for remaining points
    # Create Point objects only for points that passed the bounding box check
    points_in_region = potential_points[
        potential_points.apply(lambda row: Point(row['x'], row['y']).within(extended_region), axis=1)
    ]
    
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

def calculate_optimum_cut_fill(building_positions, surface_df, extension_percentage, z_min, z_max, z_step):
    # Ensure column names are lowercase at the start
    surface_df.columns = surface_df.columns.str.lower()
    
    # Initialize DataFrame for storing all unstable points
    unstable_points_df = pd.DataFrame(columns=[
        'X', 'Y', 'Z', 'Height_Difference', 
        'Factor_of_Safety', 'Building_Rank', 
        'Proposed_Z'
    ])
    
    # Cost variables
    unclassified_excavation_cost = 143
    select_granular_fill = 144

    # Dictionary to store initial results without slope stability analysis
    initial_results = {}

    # First pass: Calculate cut-fill volumes and basic costs for all placements
    for idx, placement in enumerate(building_positions):
        print(f"Initial processing building {idx + 1}...")

        extended_region = extend_building_region(placement, extension_percentage)
        relevant_points_df = find_points_in_extended_region(extended_region, surface_df)

        if relevant_points_df.empty:
            print(f"Skipping building placement {idx + 1}: No points found in the extended region.")
            continue

        min_cost = float('inf')
        min_cost_z = None
        all_cut_fill_by_z = {}

        # Calculate cut-fill volumes for different Z levels
        for proposed_z in np.arange(z_min, z_max + z_step, z_step):
            cut_fill_result = calculate_cut_fill_from_grid(relevant_points_df, proposed_z)

            if cut_fill_result:
                cut_volume = cut_fill_result['cut_volume']
                fill_volume = cut_fill_result['fill_volume']
                
                # Calculate base costs without retaining wall
                cut_cost = cut_volume * unclassified_excavation_cost
                fill_cost = fill_volume * select_granular_fill
                total_cost = cut_cost + fill_cost

                all_cut_fill_by_z[proposed_z] = {
                    'cut_volume': cut_volume,
                    'fill_volume': fill_volume,
                    'cut_cost': cut_cost,
                    'fill_cost': fill_cost,
                    'total_cost': total_cost
                }

                if total_cost < min_cost:
                    min_cost = total_cost
                    min_cost_z = proposed_z

        if min_cost_z is not None:
            initial_results[placement] = {
                'best_z': min_cost_z,
                'min_cost': min_cost,
                'relevant_points': relevant_points_df,
                'all_cut_fill_by_z': all_cut_fill_by_z
            }

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
        
        # Analyze each point for slope stability
        for idx, point in relevant_points_df.iterrows():
            # Calculate height difference
            height_difference = abs(min_cost_z - point['z (existing)'])
            
            # Calculate slope stability for this point
            slope_data = {
                'X': [point['x']],
                'Y': [point['y']],
                'Z': [point['z (existing)']],
                'Slope Angle': [26.57],  # 2:1 slope
                'Height of slope': [height_difference],
                'Friction Angle': [30],  # Assumed soil properties
                'Cohesion': [25],
                'Unit Weight': [20]
            }
            
            slope_stability_df = pd.DataFrame(slope_data)
            stability_results, _ = slope_stability_calculation(slope_stability_df)
            
            # If point fails stability check, add to unstable points
            if stability_results['Factor of Safety'].iloc[0] < 1.5:
                unstable_point = {
                    'X': point['x'],
                    'Y': point['y'],
                    'Z': point['z (existing)'],
                    'Height_Difference': height_difference,
                    'Factor_of_Safety': stability_results['Factor of Safety'].iloc[0],
                    'Building_Rank': rank,
                    'Proposed_Z': min_cost_z
                }
                unstable_points_df = pd.concat([
                    unstable_points_df,
                    pd.DataFrame([unstable_point])
                ], ignore_index=True)

        # Store final results without wall calculations
        optimum_results[placement] = {
            'best_z': min_cost_z,
            'cut_volume': all_cut_fill_by_z[min_cost_z]['cut_volume'],
            'fill_volume': all_cut_fill_by_z[min_cost_z]['fill_volume'],
            'min_cost': initial_data['min_cost'],
            'initial_rank': rank,
            'all_cut_fill_by_z': all_cut_fill_by_z
        }

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
    
    :param optimum_results: Dictionary containing cut and fill results for each building placement.
    :return: DataFrame with columns 'Building Number', 'Z Value', 'Cut Cost', 'Fill Cost', 'Cut Volume', 'Fill Volume', 'Total Cost'.
    """
    data = []

    for idx, (placement, results) in enumerate(optimum_results.items(), start=1):
        building_number = idx
        all_cut_fill_by_z = results['all_cut_fill_by_z']

        for z_value, cut_fill_data in all_cut_fill_by_z.items():
            data.append({
                'Building Number': building_number,
                'Z Value': z_value,
                'Cut Cost': cut_fill_data['cut_cost'],
                'Fill Cost': cut_fill_data['fill_cost'],
                'Cut Volume': cut_fill_data['cut_volume'],
                'Fill Volume': cut_fill_data['fill_volume'],
                'Total Cost': cut_fill_data['total_cost']
            })

    df = pd.DataFrame(data)
    return df



# Example usage:
# optimum_cut_fill_results = calculate_optimum_cut_fill(...)
# cut_fill_df = create_cut_fill_dataframe(optimum_cut_fill_results)
# print(cut_fill_df)
def plot_stability_results(relevant_points_df, unstable_points_df, building_placement, rank):
    """
    Create an enhanced 3D plot showing stable and unstable points with color-coded Factor of Safety.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points in blue (stable)
    ax.scatter(relevant_points_df['x'], 
              relevant_points_df['y'], 
              relevant_points_df['z (existing)'],
              c='blue', 
              alpha=0.3,
              label='Stable Points')
    
    # Plot unstable points with color gradient based on Factor of Safety
    scatter = ax.scatter(unstable_points_df['X'],
                        unstable_points_df['Y'],
                        unstable_points_df['Z'],
                        c=unstable_points_df['Factor_of_Safety'],
                        cmap='RdYlGn',  # Red to Yellow to Green colormap
                        vmin=0.5,       # Minimum FoS
                        vmax=1.5,       # Maximum FoS (threshold)
                        s=100,
                        label='Unstable Points')
    
    # Add colorbar
    plt.colorbar(scatter, label='Factor of Safety')
    
    # Plot building footprint
    x, y = building_placement.exterior.xy
    z = [unstable_points_df['Proposed_Z'].iloc[0]] * len(x)  # Use proposed elevation
    ax.plot(x, y, z, 'k--', label='Building Footprint')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Slope Stability Analysis Results - Rank {rank} Placement')
    ax.legend()
    
    plt.show()

def main():
    # Example file path for testing
    excel_file_path = '/Users/ronballer/Desktop/RunOpt/RunoptCode/InputFile NEW.xlsx'
    
    # Read data from Excel file
    surface_df = read_excel_to_dataframe(excel_file_path)
    print("Data loaded successfully from Excel")

    # Create example building dimensions
    building_length = 30
    building_width = 30
    building = create_building(length=building_length, width=building_width)
    print(f"\nCreated building with dimensions: {building_length}m x {building_width}m")

    # Create site polygon from surface data bounds
    min_x, max_x = surface_df['X'].min(), surface_df['X'].max()
    min_y, max_y = surface_df['Y'].min(), surface_df['Y'].max()
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
    z_min = surface_df['Z (Existing)'].min()
    z_max = surface_df['Z (Existing)'].max()
    z_step = 2  # 0.5m intervals for elevation analysis
    extension_percentage = 0.40  # Extend building region by 10%

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
    
    # Display results
    print("\nCut and Fill Analysis Summary:")
    print(cut_fill_df.groupby('Building Number').agg({
        'Total Cost': 'min',
        'Cut Volume': 'min',
        'Fill Volume': 'min',
        'Z Value': lambda x: x[cut_fill_df.groupby('Building Number')['Total Cost'].transform('min') == cut_fill_df['Total Cost']].iloc[0]
    }))

    if not unstable_points_df.empty:
        print("\nUnstable Points Found at:")
        print(unstable_points_df)

    # Plot 3D surface with best building placement
    best_placement = min(optimum_results.items(), key=lambda x: x[1]['min_cost'])[0]
    best_z = optimum_results[best_placement]['best_z']
    
    # Plot 3D surface grid
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface points
    ax.scatter(surface_df['X'], surface_df['Y'], surface_df['Z (Existing)'], 
              c='blue', alpha=0.5, label='Terrain')
    
    # Plot best building placement
    x, y = best_placement.exterior.xy
    ax.plot(x, y, [best_z]*len(x), 'r-', linewidth=2, label='Best Building Position')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Site Terrain with Optimal Building Placement')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()