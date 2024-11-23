from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import io
import json
import uuid
import asyncio
#uvicorn file_server:app --reload
import sys
import os

# Append the directory to sys.path
sys.path.append('/backend_server')


from .building_placement import (
    create_building_dataframe,
    calculate_optimum_cut_fill,
    create_cut_fill_dataframe,
    create_building,
)

from .slope_stability import (
    slope_stability_calculation,
)

from .pipe_design import (
    set_nodes,
    find_path,
)

from .water_supply import (
    create_water_supply_df,
    call_water_function,
)

app = FastAPI()

# Configure CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory storage for site surface data with asyncio.Lock for thread safety
site_surface_storage = {}
storage_lock = asyncio.Lock()

class NodeSelection(BaseModel):
    supply_node: tuple[int, int, int]  
    use_nodes: list[tuple[int, int, int]] 

class FlowRates(BaseModel):
    flow_rates: list[float]

# Define Pydantic model for building data
class Building(BaseModel):
    building_name: str
    length: float
    width: float

def dataframe_to_csv_response(df: pd.DataFrame, filename: str):
    """
    Converts a DataFrame to a StreamingResponse with CSV format.
    """
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
    )
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response

@app.post("/upload/site-surface")
async def upload_site_surface(file: UploadFile = File(...)):
    """
    Endpoint to upload site surface Excel or CSV file.
    Stores the file data in temporary in-memory storage.
    Returns a unique session ID for further processing.
    """
    try:
        # Read the uploaded file into memory
        contents = await file.read()
        file_like = io.BytesIO(contents)

        # Read the uploaded file into a DataFrame
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_like)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # Debug: Print DataFrame columns
        print("Site Surface DataFrame Columns:", df.columns.tolist())

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # Debug: Print DataFrame columns after lowercasing
        print("Site Surface DataFrame Columns (Lowercased):", df.columns.tolist())

        # Generate a unique session ID using UUID4
        session_id = str(uuid.uuid4())

        # Store the DataFrame in the in-memory storage with thread safety
        async with storage_lock:
            site_surface_storage[session_id] = df

        return {
            "message": "Site surface uploaded successfully.",
            "session_id": session_id
        }
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/building-info")
async def upload_building_info(
    session_id: str = Form(...),
    buildings_json: str = Form(...)
):
    """
    Endpoint to upload building information.
    Performs processing using the site surface data associated with the session ID.
    Returns processed CSV.
    """
    try:
        # Retrieve the site surface DataFrame from storage with thread safety
        async with storage_lock:
            df = site_surface_storage.get(session_id)

        if df is None:
            raise HTTPException(status_code=400, detail="Invalid or expired session ID.")

        # Debug: Print session ID being used
        print(f"Processing session ID: {session_id}")

        # Parse buildings_json string into list of dicts
        try:
            buildings_data = json.loads(buildings_json)
            if not isinstance(buildings_data, list):
                raise ValueError("buildings_json must be a list of building objects.")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON for buildings_json: {e}")

        # Validate buildings_data using Pydantic
        try:
            buildings = [Building(**b) for b in buildings_data]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid building data: {e}")

        # Convert to list of dicts
        buildings_data = [b.dict() for b in buildings]

        # Debug: Print buildings data
        print("Buildings Data:", buildings_data)

        # Create a DataFrame from the JSON objects
        buildings_df = create_building_dataframe(buildings_data)

        # Debug: Print Building DataFrame columns
        print("Building DataFrame Columns:", buildings_df.columns.tolist())

        # Ensure column names are lowercase
        buildings_df.columns = [col.lower() for col in buildings_df.columns]

        # Debug: Print Building DataFrame columns after lowercasing
        print("Building DataFrame Columns (Lowercased):", buildings_df.columns.tolist())

        # Add a Shapely object for each building
        buildings_df['building_shape'] = buildings_df.apply(
            lambda row: create_building(row['length'], row['width']), axis=1
        )

        # Perform cut and fill calculations
        building_positions = buildings_df['building_shape'].tolist()
        extension_percentage = 10.0
        z_min = df['z (existing)'].min()
        z_max = df['z (existing)'].max()
        z_step = 0.5
        print("Z MIN:", z_min)
        print("Z MAX:", z_max)

        optimum_results = calculate_optimum_cut_fill(
            building_positions=building_positions,
            surface_df=df,
            extension_percentage=extension_percentage,
            z_min=z_min,
            z_max=z_max,
            z_step=z_step
        )

        # Create a DataFrame from the results
        cut_fill_df = create_cut_fill_dataframe(optimum_results)

        # Optionally, remove the session data after processing with thread safety
        async with storage_lock:
            del site_surface_storage[session_id]

        # Convert the DataFrame to CSV response
        return dataframe_to_csv_response(cut_fill_df, "site_surface_processed.csv")
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # Log the exception for debugging
        print(f"Error in /upload/building-info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/slope-stability")
async def upload_slope_stability(file: UploadFile = File(...)):
    """
    Endpoint to upload slope stability surface information.
    Processes the data and returns processed CSV.
    """
    try:
        # Read the uploaded file into memory
        contents = await file.read()
        file_like = io.BytesIO(contents)

        # Read the uploaded file into a DataFrame
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_like)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
            print(df.head())
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        processed_df, plot_json = slope_stability_calculation(df)  # Replace with actual processing logic
        processed_df = processed_df.to_dict(orient="records")
        # Convert the DataFrame to CSV response
        #return dataframe_to_csv_response(processed_df, "slope_stability_processed.csv")
        return {
            "message": "Slope stability data processed successfully.",
            "processed_data": processed_df,  # Optional
            "plot": plot_json  # Plotly JSON
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/water-supply")
async def upload_water_supply(file: UploadFile = File(...)):
    """
    Endpoint to upload water supply information.
    Processes the data and returns processed CSV.
    """
    try:
        # Read the uploaded file into memory
        contents = await file.read()
        file_like = io.BytesIO(contents)

        # Read the uploaded file into a DataFrame
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_like)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
            #print(df)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # TODO: Implement water supply processing logic
        # Placeholder: returning the original DataFrame
        #pipeWaterSupplyDF = df  # Replace with actual processing logic
        unique_path_count = df["Path_ID"].nunique()
        #print(unique_path_count)
        create_water_supply_df(df)
        # Convert the DataFrame to CSV response
        return {
            "message": "Water supply information uploaded successfully.",
            "unique_path_count": unique_path_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/upload/flow-rates")
async def submit_flow_rates(selection: FlowRates):
    """
    Endpoint to process flow rates submitted from the second page.
    """
    try:
        # Log the received flow rates
        print("Received flow rates:", selection.flow_rates)

        # Example: Perform backend logic (e.g., save to database, process)
        if any(rate <= 0 for rate in selection.flow_rates):
            raise HTTPException(status_code=400, detail="Flow rates must all be positive numbers.")
        processed_df = call_water_function(selection.flow_rates)
        
        # Return a confirmation response
        return dataframe_to_csv_response(processed_df, "water_supply_processed.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/underground-3d-coordinates")
async def upload_underground_3d_coordinates(file: UploadFile = File(...)):
    """
    Endpoint to upload underground 3D coordinate system for pipe design.
    Processes the data and returns processed CSV.
    """
    try:
        # Read the uploaded file into memory
        contents = await file.read()
        file_like = io.BytesIO(contents)
        df = None
        # Read the uploaded file into a DataFrame
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_like)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
            print(df.head())
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        processed_df = df  # Replace with actual processing logic

        node_tuples = df[["X", "Y", "Z"]].apply(tuple, axis=1).tolist()
        
        # Convert the DataFrame to CSV response
        #return dataframe_to_csv_response(processed_df, "underground_3d_coordinates_processed.csv")
        # return {
        #     "message": "File uploaded successfully",
        #     "nodes": node_tuples
        # }
        return {
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/select-nodes")
async def select_nodes(selection: NodeSelection):
    """
    Endpoint to process user-selected supply node and use nodes.
    Validates the input and returns the selections.
    """
    # Perform basic validation
    try:
        if not selection.supply_node:
            raise HTTPException(status_code=400, detail="Supply node is missing or invalid.")
        if not selection.use_nodes:
            raise HTTPException(status_code=400, detail="Use nodes are missing or invalid.")
        paths, price, fig_json = find_path(selection.supply_node, selection.use_nodes)

        # if not os.path.exists(plot_file):
        #     raise HTTPException(status_code=500, detail="Plot file not found.")

        # # Return both the price and the plot file as separate objects
        # return {
        #     "message": "Nodes selected successfully",
        #     "price": price,
        #     "plot_file": f"/get-plot/{os.path.basename(plot_file)}"
        # }
        return {
                "message": "Pipe design processed successfully.",
                "total_price": price,
                "plot": fig_json
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @app.get("/get-plot/{filename}")
# def get_plot(filename: str):
#     """
#     Endpoint to serve the 3D plot image.
#     """
#     file_path = f"./{filename}"
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found.")
#     return FileResponse(file_path, media_type="image/png", filename=filename)