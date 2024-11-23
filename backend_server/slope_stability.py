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
  # This function purpose is to handle cases where the friction_angle and slope_angle are not in the taylor_chart dictionary
    lower_friction_angle, higher_friction_angle = get_nearest_friction_angles(friction_angle, taylor_chart)

    lower_slope_angle, higher_slope_angle = get_nearest_slope_angles(lower_friction_angle, slope_angle, taylor_chart)

    #lower_slope_angle stability_number

    friction_angle_1 = lower_friction_angle
    stability_number_1 = taylor_chart[lower_friction_angle][lower_slope_angle]

    friction_angle_2 = higher_friction_angle
    stability_number_2 = taylor_chart[higher_friction_angle][lower_slope_angle]


    lower_stability_number = interpolate_stability_number(
        lower_friction_angle, stability_number_1,
        higher_friction_angle, stability_number_2,
        friction_angle
    )

    #higher_slope_angle stability_number
    stability_number_1 = taylor_chart[lower_friction_angle][higher_slope_angle]

    stability_number_2 = taylor_chart[higher_friction_angle][higher_slope_angle]


    higher_stability_number = interpolate_stability_number(
        lower_friction_angle, stability_number_1,
        higher_friction_angle, stability_number_2,
        friction_angle
    )

    #find interpolation

    interpolated_stability_number = interpolate_stability_number(
        lower_slope_angle, lower_stability_number,
        higher_slope_angle, higher_stability_number,
        slope_angle
    )
    return interpolated_stability_number

def StabilityIteration(slopeStabilityDF):
    res = []
    for index, row in slopeStabilityDF.iterrows():
        slope_angle, friction_angle = row['Slope Angle'], row['Friction Angle']

        if friction_angle in taylor_chart:
            if slope_angle in taylor_chart[friction_angle]:
                res.append(taylor_chart[friction_angle][slope_angle])
            else:
                res.append(slope_angle_interpolation(friction_angle, slope_angle))
        else:
            if slope_angle in taylor_chart[5]:
                res.append(friction_angle_interpolation(friction_angle,slope_angle))
            else:
                res.append(double_interpolation(friction_angle,slope_angle))
    slopeStabilityDF['Stability Number'] = res
    return slopeStabilityDF

def calculateCriticalHeight(slopeStabilityDF):
    criticalHeight = [] # Set up the result array

    #Fill the criticalHeight array with values
    for i in range(len(slopeStabilityDF)):
        criticalHeight.append(slopeStabilityDF['Cohesion'][i] / (slopeStabilityDF['Unit Weight'][i] * slopeStabilityDF['Stability Number'][i]))

    slopeStabilityDF['Critical Height'] = criticalHeight
    return slopeStabilityDF
def calculateFactorofSafety(slopeStabilityDF):
    # Calculate the factor of safety (FoS)
    FactorofSafetyValues = [] # Set up the result array
    DangerCheckValues = [] # Set up the result array

    # Fill the FactorofSafetyValues and DangerCheckValues arrays with values
    for i in range(len(slopeStabilityDF)):
        FoS = slopeStabilityDF['Critical Height'][i] / slopeStabilityDF['Height of slope'][i]
        if FoS < 1:
            DangerCheckValues.append('Not Safe')
        elif 1 <= FoS < 1.5:
            DangerCheckValues.append('Questionable')
        else:
            DangerCheckValues.append('Safe')
        FactorofSafetyValues.append(FoS)

    slopeStabilityDF['Factor of Safety'] = FactorofSafetyValues
    slopeStabilityDF['Danger Check'] = DangerCheckValues

    return slopeStabilityDF

def slope_stability_calculation(slopeStabilityDF):
    slopeStabilityDF = StabilityIteration(slopeStabilityDF)
    slopeStabilityDF = calculateCriticalHeight(slopeStabilityDF)
    slopeStabilityDF = calculateFactorofSafety(slopeStabilityDF)

#     color_map = {
#     'Safe': 'green',
#     'Questionable': 'yellow',
#     'Not Safe': 'red'
# }

#     # Assign colors based on the Danger Check column
#     colors = slopeStabilityDF['Danger Check'].map(color_map)

#     # Create a 3D plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Scatter plot of the terrain with color coding based on safety
#     scatter = ax.scatter(
#         slopeStabilityDF['X'],
#         slopeStabilityDF['Y'],
#         slopeStabilityDF['Z'],
#         c=colors,
#         marker='o'
#     )

#     # Set labels
#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     ax.set_zlabel('Z Axis')
#     ax.set_title('3D Terrain Slope Stability Safety Mapping')

#     # Create Labels
#     elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Safe'),
#                     Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Questionable'),
#                     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Not Safe')]

#     ax.legend(handles= elements, loc='upper left')

#     plt.show()
    # color_map = {
    #     'Safe': 'green',
    #     'Questionable': 'yellow',
    #     'Not Safe': 'red'
    # }

    # # Assign colors based on the Danger Check column
    # colors = slopeStabilityDF['Danger Check'].map(color_map)

    # # Create the 3D scatter plot using Plotly
    # fig = go.Figure(data=go.Scatter3d(
    #     x=slopeStabilityDF['X'],
    #     y=slopeStabilityDF['Y'],
    #     z=slopeStabilityDF['Z'],
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color=colors,  # Use the color mapping
    #         opacity=0.8
    #     ),
    #     text=slopeStabilityDF['Danger Check']  # Add hover info
    # ))

    # # Set plot layout
    # fig.update_layout(
    #     title="3D Terrain Slope Stability Safety Mapping",
    #     scene=dict(
    #         xaxis_title="X Axis",
    #         yaxis_title="Y Axis",
    #         zaxis_title="Z Axis"
    #     ),
    #     margin=dict(l=0, r=0, b=0, t=40)
    # )

    # # Serialize the figure to JSON
    # fig_json = fig.to_json()
    # color_map = {
    #     'Safe': 'green',
    #     'Questionable': 'yellow',
    #     'Not Safe': 'red'
    # }

    # # Assign colors based on the Danger Check column
    # colors = slopeStabilityDF['Danger Check'].map(color_map)

    # # Create the 3D scatter plot using Plotly
    # fig = go.Figure(data=go.Scatter3d(
    #     x=slopeStabilityDF['X'],
    #     y=slopeStabilityDF['Y'],
    #     z=slopeStabilityDF['Z'],
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color=colors,  # Use the color mapping
    #         opacity=0.8
    #     ),
    #     text=slopeStabilityDF['Danger Check']  # Add hover info
    # ))

    # # Set plot layout
    # fig.update_layout(
    #     title="3D Terrain Slope Stability Safety Mapping",
    #     scene=dict(
    #         xaxis_title="X Axis",
    #         yaxis_title="Y Axis",
    #         zaxis_title="Z Axis"
    #     ),
    #     margin=dict(l=0, r=0, b=0, t=40)
    # )

    # # Show the plot in the backend
    # fig.show()

    # # Serialize the figure to JSON
    # fig_json = fig.to_json()
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