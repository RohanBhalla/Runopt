import pandas as pd
import numpy as np
#Retaining walls DF
RetainingWallsDF = pd.DataFrame(columns=['Wall_Type','Height','Start_Pos','End_Pos','Depth','Est_Thickness','Est_Footing_Span'])
wallTypes = ['L', 'cantilevered', 'gravity']
RetainingWallsDF['Wall_ID'] = range(1, len(RetainingWallsDF) + 1)
RetainingWallsDF

#Can turn ID into ordering field/ set_index



#Ranking Equation
#Ranking Equation (work for a single wall)

#Inputs: Height, 
#Output: Calculation of Ka 

# Define default constants for unit weight and internal friction angle unless provided in df
gamma = 18.5 
phi = 30
# Convert internal friction angle phi to radians for trigonometric functions
phi_rad = np.radians(phi)

# Function to calculate Rankine coefficients considering sloped backfill with angle beta
def calculate_rankine_coefficients_with_slope(phi_rad, beta_rad):
    cos_phi = np.cos(phi_rad)
    cos_beta = np.cos(beta_rad)
    
    # Ensure stability condition, where cos(beta) >= cos(phi)
    if cos_beta < cos_phi:
        raise ValueError("Unstable slope condition: beta exceeds phi!")
    # Calculate active pressure coefficient (K_a) using the sloped surface formula
    K_a = (cos_beta - np.sqrt(cos_beta**2 - cos_phi**2)) / (cos_beta + np.sqrt(cos_beta**2 - cos_phi**2))
    # Calculate passive pressure coefficient (K_p) using the sloped surface formula
    K_p = (cos_beta + np.sqrt(cos_beta**2 - cos_phi**2)) / (cos_beta - np.sqrt(cos_beta**2 - cos_phi**2))
    return K_a, K_p


def rankine_active_ka(beta, phi):
    # Convert degrees to radians
    beta_rad = np.radians(beta)
    phi_rad = np.radians(phi)
    
    # Calculate cosines
    cos_beta = np.cos(beta_rad)
    cos_phi = np.cos(phi_rad)
    
    # Rankine equation for Ka
    term_sqrt = np.sqrt(cos_beta**2 - cos_phi**2)
    Ka = cos_beta * (((cos_beta - term_sqrt)) / (cos_beta + term_sqrt))
    return Ka

def coulumb_active_ka(alpha, beta, phi, delta):
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    phi_rad = np.radians(phi)
    delta_rad = np.radians(delta)
    
    # Numerator: sin²(α + ϕ)
    numerator = np.sin(alpha_rad + phi_rad) ** 2
    
    # Denominator: sin²(α - δ) * sin(α - δ)
    sin_alpha_minus_delta = np.sin(alpha_rad - delta_rad)
    denominator = (sin_alpha_minus_delta ** 2) * np.sin(alpha_rad - delta_rad)
    
    # Complex term in the denominator
    complex_term_numerator = np.sin(phi_rad + delta_rad) * np.sin(phi_rad - beta_rad)
    complex_term_denominator = np.sin(alpha_rad - delta_rad) * np.sin(alpha_rad + beta_rad)
    complex_term = (1 + (complex_term_numerator / np.sqrt(complex_term_denominator))) ** 2
    
    # Full denominator
    full_denominator = denominator * complex_term
    
    # Calculate Ka
    Ka = numerator / full_denominator
    
    return Ka



# Function to calculate Sliding Resistance (SR)
def calculate_sliding_resistance(y_concrete, y_soil, H, Stem_thickness, Base_height, Base_width, Heel_width, Heel_surcharge_weight, friction_coefficient):
    # Calculate weights of different components
    Wstem = y_concrete * H * Stem_thickness
    Wbase = y_concrete * Base_height * Base_width
    Wsoil = y_soil * H * Heel_width
    Wsurcharge = Heel_surcharge_weight

    # Total weight
    Wtotal = Wstem + Wbase + Wsoil + Wsurcharge
    
    # Sliding Resistance (SR)
    SR = Wtotal * friction_coefficient
    
    return SR

# Function to calculate Overturning Resistance (OR)
def calculate_overturning_resistance(y_concrete, y_soil, H, Stem_thickness, Base_height, Base_width, Heel_width, Toe_length, Toe_width, Heel_surcharge_weight):
    # Calculate weights of different components
    Wstem = y_concrete * H * Stem_thickness
    Wbase = y_concrete * Base_height * Base_width
    Wsoil = y_soil * H * Heel_width
    Wsurcharge = Heel_surcharge_weight

    # Calculate moments for overturning resistance
    Mstem = Wstem * (Toe_length + 0.5 * Stem_thickness)
    Mbase = Wbase * 0.5 * Base_width
    Msoil = Wsoil * (Toe_width + Stem_thickness + 0.5 * Heel_width)
    Msurcharge = Wsurcharge * Heel_width

    # Overturning Resistance (OR)
    OR = Mstem + Mbase + Msoil + Msurcharge
    
    return OR




# Sample Wall DataFrame with a new Slope Column 'Beta' added to define the angle of the backfill
RetainingWallsDF = pd.DataFrame({
    'Wall_Type': ['L', 'cantilevered', 'gravity'],
    'Height': [5, 7, 4],        # Wall heights in meters
    'Start_Pos': [0, 10, 20],    # Arbitrary starting position
    'End_Pos': [9, 19, 29],      # Arbitrary ending position
    'Depth': [1, 2, 1],          # Depth of the wall
    'Est_Thickness': [0.3, 0.5, 0.4], # Estimated thickness
    'Est_Footing_Span': [3, 4, 2],     # Estimated footing span
    'Beta': [5, 10, 0]           # Angle of the backfill slope in degrees (Beta angle)
})

# Add Wall ID for indexing/order (Optional)
RetainingWallsDF['Wall_ID'] = range(1, len(RetainingWallsDF) + 1)

# Function that applies the earth pressures calculation to each wall
def apply_calculations(df, gamma, phi_rad):
    # Adding initial columns for storing results
    df['Active_Earth_Pressure'] = np.nan  # To store P_a
    df['Passive_Earth_Pressure'] = np.nan  # To store P_p
    
    # Iterate through each row of the df
    for idx, row in df.iterrows():
        # Apply the modified Rankine pressure calculation with the slope
        P_a, P_p = rankine_active_ka(row, gamma, phi_rad)
        # Store the results into new columns
        df.at[idx, 'Active_Earth_Pressure'] = P_a
        df.at[idx, 'Passive_Earth_Pressure'] = P_p
    
    return df

# Apply the calculations to the dataframe
RetainingWallsDF = apply_calculations(RetainingWallsDF, gamma, phi_rad)

# View the resulting dataframe with calculated pressures
print(RetainingWallsDF)




#Coloumb Equation Step

# Define unit weight of soil (gamma) and internal friction angle of the soil (phi)
gamma = 18.5  # Unit weight of the soil in kN/m³
phi = 30      # Internal friction angle in degrees

# Convert phi to radians for trigonometric calculations
phi_rad = np.radians(phi)

# Function to calculate Coulomb coefficients (Active and Passive) using sin^2(alpha + phi) version
def calculate_coulomb_sinsq_alpha_phi_coefficients(phi_rad, beta_rad, delta_rad, alpha_rad):
    # Active pressure coefficient K_a using sin^2(alpha + phi) formula
    K_a = (np.sin(alpha_rad + phi_rad)**2) / (np.sin(beta_rad + delta_rad)**2 * np.sin(beta_rad + phi_rad) * np.sin(phi_rad))
    
    # Passive pressure coefficient K_p, also involving (alpha + phi) in calculations
    K_p = (np.sin(alpha_rad + phi_rad + delta_rad)**2) / (np.sin(beta_rad - delta_rad)**2 * np.sin(beta_rad - phi_rad))
    
    return K_a, K_p

# Function to calculate the actual earth pressures for a given wall, using the updated Coulomb sin^2(alpha + phi) formula
def coulomb_earth_pressures_with_alpha(row, gamma, phi_rad):
    H = row['Height']         # Wall height
    beta = row.get('Beta', 0)  # Backfill slope angle in degrees (default to 0)
    delta = row.get('Delta', 15)  # Wall friction angle, default to 15 degrees
    alpha = row.get('Alpha', 0)   # Wall inclination angle, defaults to 0 (vertical wall)

    # Convert Beta, Delta, and Alpha to radians for trigonometric functions
    beta_rad = np.radians(beta)
    delta_rad = np.radians(delta)
    alpha_rad = np.radians(alpha)
    
    # Calculate the earth pressure coefficients (K_a and K_p)
    K_a, K_p = calculate_coulomb_sinsq_alpha_phi_coefficients(phi_rad, beta_rad, delta_rad, alpha_rad)
    
    # Calculate active and passive earth pressures
    P_a = 0.5 * gamma * H**2 * K_a
    P_p = 0.5 * gamma * H**2 * K_p
    
    return P_a, P_p

# Sample Wall DataFrame with new 'Alpha' column (wall inclination angle), as well as existing Beta and Delta
RetainingWallsDF = pd.DataFrame({
    'Wall_Type': ['L', 'cantilevered', 'gravity'],
    'Height': [5, 7, 4],        # Wall heights in meters
    'Start_Pos': [0, 10, 20],    # Arbitrary start positions
    'End_Pos': [9, 19, 29],      # Arbitrary end positions
    'Depth': [1, 2, 1],          # Depth of the wall
    'Est_Thickness': [0.3, 0.5, 0.4], # Estimated thickness
    'Est_Footing_Span': [3, 4, 2],   # Estimated footing span
    'Beta': [5, 10, 0],         # Backfill slope angles (Beta) in degrees
    'Delta': [10, 15, 5],       # Wall friction angles (Delta) in degrees
    'Alpha': [12, 8, 0]         # Inclination angles of the retaining walls (Alpha) in degrees
})

# Add Wall ID for tracking (optional)
RetainingWallsDF['Wall_ID'] = range(1, len(RetainingWallsDF) + 1)

# Function to apply the calculations and update the DataFrame
def apply_coulomb_sinsq_alpha_phi_calculations(df, gamma, phi_rad):
    # Prepare DataFrame columns for pressure results
    df['Active_Earth_Pressure'] = np.nan  # Active Earth Pressure
    df['Passive_Earth_Pressure'] = np.nan  # Passive Earth Pressure
    
    # Loop through each row in the DataFrame
    for idx, row in df.iterrows():
        # Calculate pressures for each wall
        P_a, P_p = coulomb_earth_pressures_with_alpha(row, gamma, phi_rad)
        
        # Store the results in the DataFrame
        df.at[idx, 'Active_Earth_Pressure'] = P_a
        df.at[idx, 'Passive_Earth_Pressure'] = P_p
    
    return df

# Apply the calculations to the DataFrame
RetainingWallsDF = apply_coulomb_sinsq_alpha_phi_calculations(RetainingWallsDF, gamma, phi_rad)

# Output the DataFrame with results
print(RetainingWallsDF)


#Compute Horizontal Stress
def compute_horizontal_stress(ysoil, H, Ka):
    # Compute horizontal stress (σh) using the equation: σh = γ · H · K_a
    Oh = ysoil * H * Ka
    return Oh


#Compute Lateral Pressure
def compute_lateral_pressure(ysoil, H, Ka):
    # Calculate horizontal stress (Oh = γ * H * Ka)
    Oh = ysoil * H * Ka
    # Calculate lateral pressure (Psoil = 1/2 * H * Oh)
    Psoil = 0.5 * H * Oh
    
    return Psoil


#Surcharge
def compute_rectangular_surcharge(Ka, W, H):
    # Calculate horizontal stress (Oh = γ * H * Ka)
    return Ka*W*H

#Line Surcharge
def compute_line_surcharge(q, H, a, b):
    """
    Calculate the surcharge pressure (σ) based on the given line load and conditions.
    Parameters:
    q: Line load per unit length (kN/m)
    H: Height of the retaining wall (m)
    a: Distance ratio from the wall (a = horizontal distance from wall / height of wall)
    b: Depth ratio (b = depth / height of wall)
    Returns:
    Surcharge pressure (σ) in kPa
    """
    if a > 0.4:
        # σ = (4q / πH) * (a^2 * b) / (a^2 + b^2)^2
        numerator = 4 * q * a**2 * b
        denominator = np.pi * H * (a**2 + b**2)**2
        sigma = numerator / denominator
    else:
        # σ = (q / H) * (0.203b) / (0.16 + b^2)^2
        numerator = 0.203 * b * q
        denominator = H * (0.16 + b**2)**2
        sigma = numerator / denominator
    return sigma


# Z value calculation for line surcharge
def calculate_z_bar(H, a):
    """
    Calculate the value of z_bar based on the given height (H) and distance ratio (a).
    
    Parameters:
    H: Height of the retaining wall (m)
    a: Distance ratio (a = horizontal distance from wall / height of wall)
    
    Returns:
    z_bar: The calculated value of z_bar (m)
    """
    if a > 0.4:
        z_bar = H * (1 - a * np.sqrt(1 / (2 * a**2 + 1)))
    else:
        z_bar = H * (1 - 0.348155311911396)
    
    return z_bar


# Overturning Moment for both Surcharge Types
def calculate_om_sp(Psoil, Psurcharge, H, Z):
    """
    Calculate the Overturning Moment (OM) and Sliding Pressure (SP).

    Parameters:
    Psoil (float): The lateral pressure from the soil.
    Psurcharge (float): The lateral pressure from the surcharge.
    H (float): The height of the wall.
    Z (float): The distance factor for the surcharge.

    Returns:
    tuple: A tuple containing the Overturning Moment (OM) and Sliding Pressure (SP).
    """
    # Calculate Overturning Moment
    OM = Psoil * (1/3) * H + Psurcharge * Z * H

    # Calculate Sliding Pressure
    SP = Psoil + Psurcharge

    return OM, SP


# Safety check for overturning and sliding
def safety_check(OM, SP, OR, SR):
    # Safety factor for overturning
    FS_overturning = OR / OM
    # Safety factor for sliding
    FS_sliding = SR / SP
    return FS_overturning, FS_sliding

# Main handler function to process retaining wall calculations
def retaining_wall_handler(row, gamma, phi):
    """
    Parameters:
    row: DataFrame row containing wall properties
    gamma: Unit weight of soil
    phi: Internal friction angle
    
    Returns:
    dict: Dictionary containing all calculated values and safety factors
    """
    # Step 1: Calculate Ka based on wall conditions
    if row['Wall_Type'] == 'vertical':
        Ka = rankine_active_ka(row['Beta'], phi)
    else:
        Ka = coulumb_active_ka(row['Alpha'], row['Beta'], phi, row['Delta'])
    
    # Step 2: Calculate basic soil pressures
    H = row['Height']
    horizontal_stress = compute_horizontal_stress(gamma, H, Ka)
    lateral_pressure = compute_lateral_pressure(gamma, H, Ka)
    
    # Step 3: Process surcharge based on type
    surcharge_pressure = 0
    z_value = H/2  # Default z value for rectangular surcharge
    
    if 'Surcharge_Type' in row:
        if row['Surcharge_Type'] == 'rectangular':
            surcharge_pressure = compute_rectangular_surcharge(
                Ka, 
                row['Surcharge_Load'], 
                H
            )
        elif row['Surcharge_Type'] == 'line':
            a = row['Distance_Ratio']  # horizontal distance / wall height
            b = row['Depth_Ratio']     # depth / wall height
            surcharge_pressure = compute_line_surcharge(
                row['Line_Load'], 
                H, 
                a, 
                b
            )
            z_value = calculate_z_bar(H, a)
    
    # Step 4: Calculate moments and pressures
    OM, SP = calculate_om_sp(lateral_pressure, surcharge_pressure, H, z_value)
    
    # Step 5: Calculate resistances
    # Note: You'll need to provide these values in the row or as additional parameters
    SR = calculate_sliding_resistance(
        row.get('y_concrete', 24),    # Default concrete unit weight
        gamma,                        # Soil unit weight
        H,                           # Wall height
        row['Est_Thickness'],        # Stem thickness
        row.get('Base_Height', H*0.1),  # Base height (typically 0.1*H)
        row.get('Base_Width', H*0.5),   # Base width (typically 0.5*H)
        row.get('Heel_Width', H*0.3),   # Heel width (typically 0.3*H)
        row.get('Surcharge_Load', 0),    # Surcharge weight
        row.get('Friction_Coefficient', np.tan(np.radians(phi)))  # Friction coefficient
    )
    
    OR = calculate_overturning_resistance(
        row.get('y_concrete', 24),    # Default concrete unit weight
        gamma,                        # Soil unit weight
        H,                           # Wall height
        row['Est_Thickness'],        # Stem thickness
        row.get('Base_Height', H*0.1),  # Base height
        row.get('Base_Width', H*0.5),   # Base width
        row.get('Heel_Width', H*0.3),   # Heel width
        row.get('Toe_Length', H*0.2),   # Toe length
        row.get('Toe_Width', H*0.2),    # Toe width
        row.get('Surcharge_Load', 0)     # Surcharge weight
    )
    
    # Step 6: Check safety factors
    FS_overturning, FS_sliding = safety_check(OM, SP, OR, SR)
    
    # Return all calculated values
    return {
        'Ka': Ka,
        'Horizontal_Stress': horizontal_stress,
        'Lateral_Pressure': lateral_pressure,
        'Surcharge_Pressure': surcharge_pressure,
        'Overturning_Moment': OM,
        'Sliding_Pressure': SP,
        'Sliding_Resistance': SR,
        'Overturning_Resistance': OR,
        'Safety_Factor_Overturning': FS_overturning,
        'Safety_Factor_Sliding': FS_sliding,
        'Is_Safe_Overturning': FS_overturning >= 1.5,
        'Is_Safe_Sliding': FS_sliding >= 1.5
    }
    
    
