import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

hazen_williams_constants = {
    'Cast Iron': 100,
    'Wrought Iron': 100,
    'Copper': 130,
    'Glass': 130,
    'Brass': 130,
    'Cement lined Steel': 140,
    'Cement lined Iron': 140,
    'PVC': 150,
    'ABS': 150,
    'Steel': 100
} # includes all constants relative to pipe materials

minor_loss_coefficients = {
    '90': 0.9,  # 90-degree elbow
    '45': 0.4,  # 45-degree elbow
    '22.5': 0.2,  # 22.5-degree elbow
}

g = 9.8  # gravitational constant (m/s²)
density = 1000  # density of water (kg/m³)

min_pressure = 60 * 6894.76 # psi to Pa
max_pressure = 80 * 6894.76 # psi to Pa


def useNodeFlowRateValues(pipeWaterSupplyDF, use_node_flow_rate):
    print(use_node_flow_rate)
    for i in range(len(pipeWaterSupplyDF)):
        
        if pipeWaterSupplyDF.at[i, 'Node_End'] in use_node_flow_rate:
            pipeWaterSupplyDF.at[i, 'Flow_Rate_End'] = use_node_flow_rate[pipeWaterSupplyDF.at[i, 'Node_End']]
            pipeWaterSupplyDF.at[i, 'Pressure_End'] = 101325 #  1 atm to Pa (0 gauge pressure at use nodes)
    return pipeWaterSupplyDF

def junctionNodeFlowRateValues(node, val, pipeWaterSupplyDF):
    for i in range(len(pipeWaterSupplyDF)):
        if pipeWaterSupplyDF.at[i, 'Node_End'] == node:
            pipeWaterSupplyDF.at[i, 'Flow_Rate_End'] += val
        if pipeWaterSupplyDF.at[i, 'Node_Start'] == node:
            pipeWaterSupplyDF.at[i, 'Flow_Rate_Start'] += val
    return pipeWaterSupplyDF
def setFlowRateValues(pipeWaterSupplyDF):


    # Reverse iterate through the dataframe
    for i in range(len(pipeWaterSupplyDF) - 1, -1, -1):

        # Add value to Flow_Rate_Start from Flow_Rate_Start + Flow_Rate_End
        val = pipeWaterSupplyDF.at[i, 'Flow_Rate_End']

        # Insert Flow_Rate_Start into Flow_Rate_Start and Flow_Rate_End where start_node is an end_node or even an start_node in another row.
        pipeWaterSupplyDF = junctionNodeFlowRateValues(pipeWaterSupplyDF.at[i, 'Node_Start'], val, pipeWaterSupplyDF)
    
    return pipeWaterSupplyDF

def set_initial_pressure(pipeWaterSupplyDF):
    # Set the initial pressure for the supply node to a random value between 60 and 80 psi
    #initial_pressure = np.random.uniform(min_pressure, max_pressure)
    initial_pressure = max_pressure
    pipeWaterSupplyDF.at[0, 'Pressure_Start'] = initial_pressure
    return pipeWaterSupplyDF

def calculateMinorHeadLoss(v, fittings):
    K = minor_loss_coefficients[str(fittings)]
    return K * (v ** 2) / (2 * g)

def caclulateMajorHeadLoss(flow_rate, diameter, length, material):
    # use hazen williams equation to calculate head loss
    constant = hazen_williams_constants[material]
    major_head_loss = (10.583 * length * (flow_rate**1.85)) / ((constant ** 1.85) * (diameter ** 4.87))
    return major_head_loss

def bernoulli_equation(pipeWaterSupplyDF):
    
    for i in range(len(pipeWaterSupplyDF)):

        p1 = pipeWaterSupplyDF.at[i, 'Pressure_Start']
        material = pipeWaterSupplyDF.at[i, 'Pipe Material']
        flow_rate_start = pipeWaterSupplyDF.at[i, 'Flow_Rate_Start']
        flow_rate_end = pipeWaterSupplyDF.at[i, 'Flow_Rate_End']
        elevation_start = pipeWaterSupplyDF.at[i, 'Elevation_Start']
        elevation_end = pipeWaterSupplyDF.at[i, 'Elevation_End']
        diameter = pipeWaterSupplyDF.at[i, 'Pipe_Diameter']
        length = pipeWaterSupplyDF.at[i, 'Pipe_Length']
        fittings = pipeWaterSupplyDF.at[i, 'Fittings']
        minor_head_loss = pipeWaterSupplyDF.at[i, 'Minor Head Loss']
       
        v1 = flow_rate_start / (np.pi * (diameter / 2) ** 2)
        v2 = flow_rate_end / (np.pi * (diameter / 2) ** 2)

        if pipeWaterSupplyDF.at[i, 'Pressure_End'] != 0:
            pipeWaterSupplyDF.at[i, 'Velocity_Start'] = v1
            pipeWaterSupplyDF.at[i, 'Velocity_End'] = v2
            continue

        major_head_loss = caclulateMajorHeadLoss(flow_rate_end, diameter, length, material)
        minor_head_loss += calculateMinorHeadLoss(v1, fittings)

        head_loss = major_head_loss + minor_head_loss

        p2 = density * g * ( # Bernoulli function
                (elevation_start - elevation_end) + 
                (p1 / (density * g)) + 
                ((v1 ** 2 - v2 ** 2) / (2 * g)) - head_loss
            )

        pipeWaterSupplyDF.at[i, 'Velocity_Start'] = v1
        pipeWaterSupplyDF.at[i, 'Velocity_End'] = v2
        pipeWaterSupplyDF.at[i, 'Pressure_End'] = p2

        current_end_node = pipeWaterSupplyDF.at[i, 'Node_End']

        for j in range(len(pipeWaterSupplyDF)):
            if pipeWaterSupplyDF.at[j, 'Node_Start'] == current_end_node:
                pipeWaterSupplyDF.at[j, 'Pressure_Start'] = p2
                pipeWaterSupplyDF.at[j, 'Minor Head Loss'] = minor_head_loss
    return pipeWaterSupplyDF


pipeWaterSupplyDF = None

def create_water_supply_df(df):
    global pipeWaterSupplyDF
    pipeWaterSupplyDF = df
    pipeWaterSupplyDF['Flow_Rate_End'] = pipeWaterSupplyDF['Flow_Rate_End'].astype(float)
    pipeWaterSupplyDF['Flow_Rate_Start'] = pipeWaterSupplyDF['Flow_Rate_Start'].astype(float)
    pipeWaterSupplyDF['Pressure_Start'] = pipeWaterSupplyDF['Pressure_Start'].astype(float)
    pipeWaterSupplyDF['Pressure_End'] = pipeWaterSupplyDF['Pressure_End'].astype(float)
    pipeWaterSupplyDF['Velocity_Start'] = pipeWaterSupplyDF['Velocity_Start'].astype(float)
    pipeWaterSupplyDF['Velocity_End'] = pipeWaterSupplyDF['Velocity_End'].astype(float)
    pipeWaterSupplyDF['Pressure_End'] = pipeWaterSupplyDF['Pressure_End'].astype(float)
    pipeWaterSupplyDF['Minor Head Loss'] = pipeWaterSupplyDF['Minor Head Loss'].astype(float)

    
def call_water_function(flow_rate):
    global pipeWaterSupplyDF
    optimal_path_1 = [(0, 0, 0), (2, 2, 2), (4, 4, 4), (6, 6, 6)]
    optimal_path_2 = [(0, 0, 0), (2, 2, 2), (4, 4, 4), (6, 6, 5)]
    use_node_flow_rate = {}
    #use_node_flow_rate[str(optimal_path_1[-1])] = 0.000702 #select flow rate demand at last node in m^3/s
    #use_node_flow_rate[str(optimal_path_2[-1])] = 0.000468 #select flow rate demand at last node in m^3/s
    use_node_flow_rate[optimal_path_1[-1]] = flow_rate[0] #select flow rate demand at last node in m^3/s
    use_node_flow_rate[optimal_path_2[-1]] = flow_rate[-1] #select flow rate demand at last node in m^3/s
   
    pipeWaterSupplyDF = useNodeFlowRateValues(pipeWaterSupplyDF, use_node_flow_rate)
    pipeWaterSupplyDF = setFlowRateValues(pipeWaterSupplyDF)
    pipeWaterSupplyDF = set_initial_pressure(pipeWaterSupplyDF)
    pipeWaterSupplyDF = bernoulli_equation(pipeWaterSupplyDF)
    return pipeWaterSupplyDF