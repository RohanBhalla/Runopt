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
    Find the points in surface_df that are within the extended region.
    """
    # Convert column names to lowercase for consistency
    surface_df.columns = surface_df.columns.str.lower()
    
    points_in_region = surface_df[
        surface_df.apply(lambda row: Point(row['x'], row['y']).within(extended_region), axis=1)
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

def analyze_slope_stability_for_building(relevant_points_df, proposed_z, building_placement):
    """
    Analyze slope stability around a building placement and determine if retaining walls are needed.
    """
    # Create a slope stability dataframe for the points around the building
    slope_stability_df = pd.DataFrame()
    
    # Calculate average existing elevation
    avg_existing_z = relevant_points_df['z (existing)'].mean()
    
    # Calculate height difference
    height_difference = abs(proposed_z - avg_existing_z)
    
    # Calculate slope angle (assuming a basic 2:1 slope from building edge)
    slope_angle = np.degrees(np.arctan(1/2))  # 2:1 slope is approximately 26.57 degrees
    
    # Prepare slope stability input
    slope_data = {
        'X': [relevant_points_df['x'].mean()],
        'Y': [relevant_points_df['y'].mean()],
        'Z': [avg_existing_z],
        'Slope Angle': [slope_angle],
        'Height of slope': [height_difference],
        'Friction Angle': [30],  # Assumed soil friction angle
        'Cohesion': [25],       # Assumed soil cohesion (kPa)
        'Unit Weight': [20]      # Assumed soil unit weight (kN/m³)
    }
    
    slope_stability_df = pd.DataFrame(slope_data)
    
    # Calculate slope stability
    stability_results, _ = slope_stability_calculation(slope_stability_df)
    
    # Determine if retaining wall is needed
    needs_retaining_wall = stability_results['Factor of Safety'].iloc[0] < 1.5
    
    if needs_retaining_wall:
        # Calculate retaining wall properties
        wall_height = height_difference
        wall_length = np.sqrt(building_placement.area) # Approximate wall length based on building size
        
        retaining_wall_data = {
            'X': slope_data['X'][0],
            'Y': slope_data['Y'][0],
            'Z': avg_existing_z,
            'Wall Height': wall_height,
            'Wall Length': wall_length,
            'Factor of Safety': stability_results['Factor of Safety'].iloc[0]
        }
        
        return True, retaining_wall_data
    
    return False, None

def calculate_optimum_cut_fill(building_positions, surface_df, extension_percentage, z_min, z_max, z_step):
    # Ensure column names are lowercase at the start
    surface_df.columns = surface_df.columns.str.lower()
    
    # Initialize retaining walls DataFrame
    retaining_walls_df = pd.DataFrame(columns=['X', 'Y', 'Z', 'Wall Height', 'Wall Length', 'Factor of Safety'])
    
    # Cost variables
    unclassified_excavation_cost = 143
    select_granular_fill = 144
    retaining_wall_cost_per_sqm = 500  # Added cost for retaining walls

    optimum_results = {}

    for idx, placement in enumerate(building_positions):
        print(f"Processing building {idx + 1}...")

        extended_region = extend_building_region(placement, extension_percentage)
        relevant_points_df = find_points_in_extended_region(extended_region, surface_df)

        if relevant_points_df.empty:
            print(f"Skipping building placement {idx + 1}: No points found in the extended region.")
            continue

        best_z = None
        best_cut_fill = None
        min_net_volume = float('inf')
        min_cost = float('inf')
        min_cost_z = None
        all_cut_fill_by_z = {}

        # Calculate average Z value for the relevant points
        avg_z = relevant_points_df['z (existing)'].mean()

        for proposed_z in np.arange(z_min, z_max + z_step, z_step):
            cut_fill_result = calculate_cut_fill_from_grid(relevant_points_df, proposed_z)
            
            if cut_fill_result:
                # Analyze slope stability and check if retaining wall is needed
                needs_wall, wall_data = analyze_slope_stability_for_building(
                    relevant_points_df, 
                    proposed_z, 
                    placement
                )
                
                cut_volume = cut_fill_result['cut_volume']
                fill_volume = cut_fill_result['fill_volume']
                
                # Calculate base costs
                cut_cost = cut_volume * unclassified_excavation_cost
                fill_cost = fill_volume * select_granular_fill
                total_cost = cut_cost + fill_cost

                # Add retaining wall cost if needed
                if needs_wall:
                    wall_area = wall_data['Wall Height'] * wall_data['Wall Length']
                    total_cost += wall_area * retaining_wall_cost_per_sqm
                    # Add wall to retaining walls DataFrame
                    retaining_walls_df = pd.concat([
                        retaining_walls_df,
                        pd.DataFrame([wall_data])
                    ], ignore_index=True)

                # Store results
                all_cut_fill_by_z[proposed_z] = {
                    'cut_volume': cut_volume,
                    'fill_volume': fill_volume,
                    'cut_cost': cut_cost,
                    'fill_cost': fill_cost,
                    'total_cost': total_cost,
                    'needs_retaining_wall': needs_wall
                }

                # Update if this gives a smaller cost
                if total_cost < min_cost:
                    min_cost = total_cost
                    min_cost_z = proposed_z
                    best_cut_fill = cut_fill_result

        if best_cut_fill is not None:
            optimum_results[placement] = {
                'best_z': min_cost_z,
                'cut_volume': best_cut_fill['cut_volume'],
                'fill_volume': best_cut_fill['fill_volume'],
                'min_cost': min_cost,
                'all_cut_fill_by_z': all_cut_fill_by_z
            }

    return optimum_results, retaining_walls_df



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
def main():
    # Example file path for testing
    excel_file_path = '/Users/ronballer/Desktop/RunOpt/RunoptCode/InputFile.xlsx'
    
    # Read data from Excel file
    df = read_excel_to_dataframe(excel_file_path)
    print("DataFrame from Excel:")
    print(df.head())

    # Create a building
    building = create_building(length=10, width=5)
    print("\nCreated Building:")
    print(building)

    # Create a confined region
    # site_polygon = box(0, 0, 100, 100)  # Example site polygon
    min_x, max_x = df['X'].min(), df['X'].max()
    min_y, max_y = df['Y'].min(), df['Y'].max()
    # Create a site polygon using the bounds
    site_polygon = box(min_x, min_y, max_x, max_y)

    confined_region = create_confined_region(site_polygon, percentage=50)
    print("\nConfined Region:")
    print(confined_region)

    # Find valid placements
    valid_placements = find_valid_placements(confined_region, building)
    print("\nValid Placements:")
    for placement in valid_placements:
        print(placement)

    # Plot 3D surface grid
    plot_3d_surface_grid(df)

    

if __name__ == "__main__":
    main()