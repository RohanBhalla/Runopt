import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import plotly.graph_objects as go

taylor_chart = {
    0: { # Friction angle
        0: 0.0001,
        5: 0.05,  # Slope angle : Stability Number
        10: 0.08,
        15: 0.10,
        20: 0.114,
        25: 0.128,
        30: 0.138,
        35: 0.148,
        40: 0.158,
        45: 0.168,
        50: 0.178,
        55: 0.182,
        60: 0.19,
        65: 0.20,
        70: 0.21,
        75: 0.22,
        80: 0.23,
        85: 0.25,
        90: 0.27,
    },
    5: {
        0: 0.0001,
        5: 0.0001,
        10: 0.05,
        15: 0.068,
        20: 0.08,
        25: 0.092,
        30: 0.105,
        35: 0.115,
        40: 0.125,
        45: 0.134,
        50: 0.142,
        55: 0.15,
        60: 0.16,
        65: 0.17,
        70: 0.18,
        75: 0.195,
        80: 0.21,
        85: 0.225,
        90: 0.24,
    },
    10: {
        0: 0.0001,
        5: 0.0001,
        10: 0.0001,
        15: 0.04,
        20: 0.05,
        25: 0.068,
        30: 0.079,
        35: 0.088,
        40: 0.10,
        45: 0.11,
        50: 0.12,
        55: 0.13,
        60: 0.137,
        65: 0.15,
        70: 0.16,
        75: 0.171,
        80: 0.185,
        85: 0.20,
        90: 0.22,
    },
    15: {
        0: 0.0001,
        5: 0.0001,
        10: 0.0001,
        15: 0.0001,
        20: 0.02,
        25: 0.035,
        30: 0.05,
        35: 0.06,
        40: 0.07,
        45: 0.08,
        50: 0.091,
        55: 0.101,
        60: 0.115,
        65: 0.125,
        70: 0.139,
        75: 0.15,
        80: 0.164,
        85: 0.18,
        90: 0.20,
    },
    20: {
        0: 0.0001,
        5: 0.0001,
        10: 0.0001,
        15: 0.0001,
        20: 0.0001,
        25: 0.012,
        30: 0.03,
        35: 0.04,
        40: 0.05,
        45: 0.063,
        50: 0.078,
        55: 0.09,
        60: 0.10,
        65: 0.11,
        70: 0.125,
        75: 0.13,
        80: 0.141,
        85: 0.155,
        90: 0.17,
    },
    25: {
        0: 0.0001,
        5: 0.0001,
        10: 0.0001,
        15: 0.0001,
        20: 0.0001,
        25: 0.0001,
        30: 0.01,
        35: 0.03,
        40: 0.04,
        45: 0.042,
        50: 0.055,
        55: 0.07,
        60: 0.08,
        65: 0.09,
        70: 0.109,
        75: 0.12,
        80: 0.132,
        85: 0.15,
        90: 0.16,
    },
    30: {
        0: 0.0001,
        5: 0.0001,
        10: 0.0001,
        15: 0.0001,
        20: 0.0001,
        25: 0.0001,
        30: 0.0001,
        35: 0.01,
        40: 0.02,
        45: 0.025,
        50: 0.042,
        55: 0.05,
        60: 0.06,
        65: 0.075,
        70: 0.08,
        75: 0.10,
        80: 0.12,
        85: 0.13,
        90: 0.15,
    }
}
def interpolate_stability_number(x1, y1, x2, y2, x):
    """
    Perform linear interpolation to estimate a value.
    """
    return y1 + ( (x - x1) / (x2 - x1) ) * (y2 - y1)

def get_nearest_friction_angles(friction_angle, taylor_chart):
    #This function finds the two nearest friction angles available in the taylor_chart dictionary
    available_angles = sorted(taylor_chart.keys())
    lower_angle = max([angle for angle in available_angles if angle <= friction_angle], default=None)
    higher_angle = min([angle for angle in available_angles if angle >= friction_angle], default=None)
    return lower_angle, higher_angle

def get_nearest_slope_angles(friction_angle, slope_angle, taylor_chart):
   #This function finds the two nearest slope angles available in the taylor_chart dictionary for a given friction angle.

    if friction_angle not in taylor_chart:
        raise ValueError(f"Friction angle {friction_angle} not found in the dictionary.")

    # Get the available slope angles for the specified friction angle
    available_slope_angles = sorted(taylor_chart[friction_angle].keys())
    lower_angle = max([angle for angle in available_slope_angles if angle <= slope_angle], default=None)
    higher_angle = min([angle for angle in available_slope_angles if angle >= slope_angle], default=None)
    return lower_angle, higher_angle

def friction_angle_interpolation(friction_angle, slope_angle):
    # This function purpose is to handle cases where the friction_angle is not in the taylor_chart dictionary
    lower_friction_angle, higher_friction_angle = get_nearest_friction_angles(friction_angle, taylor_chart)

    stability_number_1 = taylor_chart[lower_friction_angle][slope_angle]

    stability_number_2 = taylor_chart[higher_friction_angle][slope_angle]

    # Calculate the interpolated stability number
    interpolated_stability_number = interpolate_stability_number(
        lower_friction_angle, stability_number_1,
        higher_friction_angle, stability_number_2,
        friction_angle
    )
    return interpolated_stability_number

def slope_angle_interpolation(friction_angle, slope_angle):
    # This function purpose is to handle cases where the slope_angle is not in the taylor_chart dictionary
    lower_slope_angle, higher_slope_angle = get_nearest_slope_angles(friction_angle, slope_angle, taylor_chart)

    stability_number_1 = taylor_chart[friction_angle][lower_slope_angle]

    stability_number_2 = taylor_chart[friction_angle][higher_slope_angle]


    interpolated_stability_number = interpolate_stability_number(
        lower_slope_angle, stability_number_1,
        higher_slope_angle, stability_number_2,
        slope_angle
    )
    return interpolated_stability_number

def double_interpolation(friction_angle, slope_angle):
    """
    Perform double interpolation for both friction angle and slope angle.
    Returns interpolated stability number.
    """
    # Get nearest friction angles
    lower_friction_angle, higher_friction_angle = get_nearest_friction_angles(friction_angle, taylor_chart)
    if lower_friction_angle is None or higher_friction_angle is None:
        raise ValueError(f"Cannot interpolate friction angle {friction_angle}: out of bounds")

    # Get nearest slope angles
    lower_slope_angle, higher_slope_angle = get_nearest_slope_angles(lower_friction_angle, slope_angle, taylor_chart)
    if lower_slope_angle is None or higher_slope_angle is None:
        raise ValueError(f"Cannot interpolate slope angle {slope_angle}: out of bounds")

    # First interpolation: along friction angle for lower slope angle
    stability_lower_slope = interpolate_stability_number(
        lower_friction_angle, taylor_chart[lower_friction_angle][lower_slope_angle],
        higher_friction_angle, taylor_chart[higher_friction_angle][lower_slope_angle],
        friction_angle
    )

    # Second interpolation: along friction angle for higher slope angle
    stability_higher_slope = interpolate_stability_number(
        lower_friction_angle, taylor_chart[lower_friction_angle][higher_slope_angle],
        higher_friction_angle, taylor_chart[higher_friction_angle][higher_slope_angle],
        friction_angle
    )

    # Final interpolation: along slope angle
    final_stability = interpolate_stability_number(
        lower_slope_angle, stability_lower_slope,
        higher_slope_angle, stability_higher_slope,
        slope_angle
    )
    
    return final_stability

def StabilityIteration(slopeStabilityDF):
    def get_stability_number(row):
        try:
            slope_angle, friction_angle = row['Slope Angle'], row['Friction Angle']
            
            # Input validation
            if not (0 <= slope_angle <= 90) or not (0 <= friction_angle <= 30):
                raise ValueError(f"Invalid input: slope_angle={slope_angle}, friction_angle={friction_angle}")
            
            # Check if exact values exist
            if friction_angle in taylor_chart and slope_angle in taylor_chart[friction_angle]:
                stability_number = taylor_chart[friction_angle][slope_angle]
                if stability_number <= 0.0001:  # Handle undefined values
                    raise ValueError("Stability number undefined for these parameters")
                return stability_number
                
            # Need interpolation
            try:
                if friction_angle in taylor_chart:
                    return slope_angle_interpolation(friction_angle, slope_angle)
                else:
                    return double_interpolation(friction_angle, slope_angle)
            except ValueError as e:
                print(f"Interpolation error for slope={slope_angle}, friction={friction_angle}: {str(e)}")
                return np.nan
                
        except Exception as e:
            print(f"Error calculating stability number: {str(e)}")
            return np.nan

    slopeStabilityDF['Stability Number'] = slopeStabilityDF.apply(get_stability_number, axis=1)
    
    # Handle any NaN values that resulted from errors
    nan_count = slopeStabilityDF['Stability Number'].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} points could not be calculated and will be excluded")
        slopeStabilityDF = slopeStabilityDF.dropna(subset=['Stability Number'])
    
    return slopeStabilityDF

def calculateCriticalHeight(slopeStabilityDF):
    # Vectorized calculation of Critical Height
    slopeStabilityDF['Critical Height'] = slopeStabilityDF['Cohesion'] / (
        slopeStabilityDF['Unit Weight'] * slopeStabilityDF['Stability Number']
    )
    return slopeStabilityDF
def calculateFactorofSafety(slopeStabilityDF):
    # Handle division by zero and negative values
    mask = (slopeStabilityDF['Height of slope'] > 0) & (slopeStabilityDF['Stability Number'] > 0)
    
    # Initialize Factor of Safety column with NaN
    slopeStabilityDF['Factor of Safety'] = np.nan
    
    # Calculate only for valid entries
    slopeStabilityDF.loc[mask, 'Factor of Safety'] = (
        slopeStabilityDF.loc[mask, 'Critical Height'] / 
        slopeStabilityDF.loc[mask, 'Height of slope']
    )
    
    # More detailed safety classification
    conditions = [
        slopeStabilityDF['Factor of Safety'].isna(),
        slopeStabilityDF['Factor of Safety'] < 1,
        (slopeStabilityDF['Factor of Safety'] >= 1) & (slopeStabilityDF['Factor of Safety'] < 1.25),
        (slopeStabilityDF['Factor of Safety'] >= 1.25) & (slopeStabilityDF['Factor of Safety'] < 1.5),
        slopeStabilityDF['Factor of Safety'] >= 1.5
    ]
    choices = ['Invalid', 'Not Safe', 'High Risk', 'Questionable', 'Safe']
    slopeStabilityDF['Danger Check'] = np.select(conditions, choices, default='Unknown')
    
    # Log any invalid calculations
    invalid_count = (slopeStabilityDF['Danger Check'] == 'Invalid').sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} points have invalid safety factors")
    
    return slopeStabilityDF

def slope_stability_calculation(slopeStabilityDF):
    slopeStabilityDF = StabilityIteration(slopeStabilityDF)
    slopeStabilityDF = calculateCriticalHeight(slopeStabilityDF)
    slopeStabilityDF = calculateFactorofSafety(slopeStabilityDF)
    
    # Debug print of stability results
    print(f"Stability calculation results:")
    print(f"Total points analyzed: {len(slopeStabilityDF)}")
    print(f"Points with FoS < 1.5: {(slopeStabilityDF['Factor of Safety'] < 1.5).sum()}")
    print(f"Factor of Safety range: {slopeStabilityDF['Factor of Safety'].min():.2f} - {slopeStabilityDF['Factor of Safety'].max():.2f}")
    
    color_map = {
        'Safe': 'green',
        'Questionable': 'yellow',
        'Not Safe': 'red'
    }

    # Assign colors based on the Danger Check column
    colors = slopeStabilityDF['Danger Check'].map(color_map)

    # Create the main 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=slopeStabilityDF['X'],
        y=slopeStabilityDF['Y'],
        z=slopeStabilityDF['Z'],
        mode='markers',
        marker=dict(
            size=5,
            color=colors,  # Use the color mapping
            opacity=0.8
        ),
        text=slopeStabilityDF['Danger Check']  # Add hover info
    ))

    # Add legend items as invisible traces
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Safe'
    ))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='yellow'),
        name='Questionable'
    ))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Not Safe'
    ))

    # Set plot layout
    fig.update_layout(
        title="3D Terrain Slope Stability Safety Mapping",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            title="Safety Categories",
            itemsizing="constant"
        )
    )

    # Show the plot in the backend
    #fig.show()

    # Serialize the figure to JSON
    fig_json = fig.to_json()
    return slopeStabilityDF, fig_json