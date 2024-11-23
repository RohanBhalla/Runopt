import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go

# Function to generate nodes in a 3D grid
def generate_nodes(grid_size, step):
    nodes = []
    for x in range(0, grid_size[0], step):
        for y in range(0, grid_size[1], step):
            for z in range(0, grid_size[2], step):
                nodes.append((x, y, z))
    return nodes

def calculatePrice(paths):
    totalVolume = 0
    visited = set()
    max_height = 100
    angle = 45
    for path in paths:
        height_sum = 0
        for node in range(1, len(path)):
            if path[node] not in visited:
                width = math.sqrt((path[node][1]- path[node - 1][1])**2 + (path[node][0] - path[node-1][0])**2)
                height = max_height - path[node][2]
                height_sum += (1/math.tan(angle * (math.pi/180))) * (height**2) * width
                visited.add(path[node])
        totalVolume += (height_sum)
    return totalVolume * 100


# Calculate Euclidean distance between two nodes
def euclidean_distance(node1, node2):
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2 + (node1[2] - node2[2])**2)

# Dijkstra's algorithm for finding the shortest path
def dijkstra(graph, start, end):
    queue = [(0, start)]  # (cost, node)
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    path = {node: None for node in graph}

    while queue:
        (cost, node) = heapq.heappop(queue)
        
        if node == end:
            break
        
        for neighbor, weight in graph[node].items():
            new_cost = cost + weight
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
                path[neighbor] = node
                
    return distances, path

# Function to update the graph with zero-cost paths for shared routes
def update_graph_with_path(graph, path):
    """
    Update the graph to set the cost of the edges along the given path to zero.
    """
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        
        # Set the cost of the path to zero for future calculations
        graph[node1][node2] = 0
        graph[node2][node1] = 0  # Assuming undirected graph

# Function to construct the graph based on nodes
def construct_graph(nodes, max_distance=15):
    graph = {}
    for i, node1 in enumerate(nodes):
        # Ensure the node is initialized in the graph dictionary
        if node1 not in graph:
            graph[node1] = {}
        
        for j, node2 in enumerate(nodes):
            if i != j:
                distance = euclidean_distance(node1, node2)
                if distance <= max_distance:  # Only connect nodes within the max distance
                    if node2 not in graph:
                        graph[node2] = {}
                    
                    graph[node1][node2] = distance
                    graph[node2][node1] = distance
    return graph

# Generate nodes
#nodes = generate_nodes((100, 100, 100), 10)
# df_nodes = pd.DataFrame(nodes, columns=["X", "Y", "Z"])

# # Save to CSV
# df_nodes.to_csv("pipe_design_nodes.csv", index=False)
nodes = []
def set_nodes(tuple_nodes):
    nodes.append(tuple_nodes)
    return "hello"

def find_path(supply_node, use_nodes):
    #supply_node = (0, 0, 0)
    #use_nodes = [nodes[-1], nodes[-2]]
    # use_nodes = [
    #     nodes[-1],       
    #     nodes[-2],       
    #     nodes[50],       
    #     (90, 0, 0),      
    #     (0, 90, 10),     
    #     (40, 40, 40),    
    #     (10, 90, 90),    
    #     (80, 20, 40),    
    #     nodes[100],      
    #     (0, 0, 50)      
    # ]
    use_nodes = sorted(use_nodes, key=lambda node: euclidean_distance(node, supply_node))

    print("Ordered use nodes by distance to supply node:", use_nodes)

    # Construct the graph
    graph = construct_graph(nodes[0])

    # List to store paths for visualization
    paths = []

    # Iterate over use nodes and calculate the paths
    for i, use_node in enumerate(use_nodes):
        print(f"Calculating shortest path to use node: {use_node}")
        
        # Calculate the shortest path using Dijkstra
        distances, path = dijkstra(graph, supply_node, use_node)
        
        # Reconstruct the optimal path
        current_node = use_node
        optimal_path = []
        while current_node is not None:
            optimal_path.append(current_node)
            current_node = path[current_node]
        
        optimal_path = optimal_path[::-1]  # Reverse the path to start from the supply node
        paths.append(optimal_path)
        
        #print(f"Optimal Path for Use Node {i+1}: {optimal_path}")
        #print(f"Total Cost (Distance) for Use Node {i+1}: {distances[use_node]}")
        
        # Update the graph to favor shared paths
        update_graph_with_path(graph, optimal_path)

    totalPrice = calculatePrice(paths)
    #print(f"The total price of the system is ${totalPrice}")
    #Visualization of paths in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Scatter plot for the generated nodes
    # x_coords = [node[0] for node in nodes]
    # y_coords = [node[1] for node in nodes]
    # z_coords = [node[2] for node in nodes]
    # ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o', label='Nodes')

    # # Plot the supply node and use nodes
    # ax.scatter(*supply_node, c='green', marker='o', s=100, label='Supply Node')
    # for use_node in use_nodes:
    #     ax.scatter(*use_node, c='yellow', marker='o', s=100, label='Use Node')

    # # Draw the paths with different colors
    # colors = ['red', 'orange', 'green', 'yellow']
    # for i, path in enumerate(paths):
    #     optimal_x = [node[0] for node in path]
    #     optimal_y = [node[1] for node in path]
    #     optimal_z = [node[2] for node in path]
    #     ax.plot(optimal_x, optimal_y, optimal_z, c='green', marker='o', label=f'Path to Use Node {i+1}')

    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Z Coordinate')
    # plt.legend()
    # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x_coords = [node[0] for node in nodes]
    # y_coords = [node[1] for node in nodes]
    # z_coords = [node[2] for node in nodes]
    # ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o', label='Nodes')
    # ax.scatter(*supply_node, c='green', marker='o', s=100, label='Supply Node')
    # for use_node in use_nodes:
    #     ax.scatter(*use_node, c='yellow', marker='o', s=100, label='Use Node')
    # colors = ['red', 'orange', 'green', 'yellow']
    # for i, path in enumerate(paths):
    #     optimal_x = [node[0] for node in path]
    #     optimal_y = [node[1] for node in path]
    #     optimal_z = [node[2] for node in path]
    #     ax.plot(optimal_x, optimal_y, optimal_z, c='green', label=f'Path to Use Node {i+1}')
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Z Coordinate')
    # plt.legend()

    # # Save the plot to a file
    # output_file = "3d_plot.png"
    # plt.savefig(output_file)
    # plt.close(fig)
    # return paths, totalPrice, output_file
    # Visualization of paths in 3D using Plotly
    # fig = go.Figure()

    # # Scatter plot for the generated nodes
    # x_coords = [node[0] for node in nodes]
    # y_coords = [node[1] for node in nodes]
    # z_coords = [node[2] for node in nodes]
    # fig.add_trace(go.Scatter3d(
    #     x=x_coords,
    #     y=y_coords,
    #     z=z_coords,
    #     mode='markers',
    #     marker=dict(size=5, color='blue'),
    #     name='Nodes'
    # ))

    # # Plot the supply node
    # fig.add_trace(go.Scatter3d(
    #     x=[supply_node[0]],
    #     y=[supply_node[1]],
    #     z=[supply_node[2]],
    #     mode='markers',
    #     marker=dict(size=10, color='green'),
    #     name='Supply Node'
    # ))

    # # Plot the use nodes
    # for use_node in use_nodes:
    #     fig.add_trace(go.Scatter3d(
    #         x=[use_node[0]],
    #         y=[use_node[1]],
    #         z=[use_node[2]],
    #         mode='markers',
    #         marker=dict(size=8, color='yellow'),
    #         name='Use Node'
    #     ))

    # # Draw the paths
    # for i, path in enumerate(paths):
    #     optimal_x = [node[0] for node in path]
    #     optimal_y = [node[1] for node in path]
    #     optimal_z = [node[2] for node in path]
    #     fig.add_trace(go.Scatter3d(
    #         x=optimal_x,
    #         y=optimal_y,
    #         z=optimal_z,
    #         mode='lines+markers',
    #         line=dict(color='green', width=2),
    #         name=f'Path {i+1}'
    #     ))

    # # Set axis labels
    # fig.update_layout(
    #     title="3D Pipe Design Visualization",
    #     scene=dict(
    #         xaxis=dict(title="X Coordinate"),
    #         yaxis=dict(title="Y Coordinate"),
    #         zaxis=dict(title="Z Coordinate")
    #     ),
    #     margin=dict(l=0, r=0, b=0, t=40)
    # )

    # # Serialize the Plotly figure to JSON
    # fig.show()
    # fig_json = fig.to_json()

    # return paths, totalPrice, fig_json
    # Separate nodes into categories
    other_nodes = [node for node in nodes if node not in use_nodes and node != supply_node]

    # Visualization of paths in 3D using Plotly
    fig = go.Figure()

    # Plot other nodes (blue)
    other_x = [node[0] for node in other_nodes]
    other_y = [node[1] for node in other_nodes]
    other_z = [node[2] for node in other_nodes]
    fig.add_trace(go.Scatter3d(
        x=other_x,
        y=other_y,
        z=other_z,
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Other Nodes'
    ))

    # Plot the supply node (green)
    fig.add_trace(go.Scatter3d(
        x=[supply_node[0]],
        y=[supply_node[1]],
        z=[supply_node[2]],
        mode='markers',
        marker=dict(size=15, color='green'),
        name='Supply Node'
    ))

    # Plot the use nodes (yellow)
    use_x = [node[0] for node in use_nodes]
    use_y = [node[1] for node in use_nodes]
    use_z = [node[2] for node in use_nodes]
    fig.add_trace(go.Scatter3d(
        x=use_x,
        y=use_y,
        z=use_z,
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Use Nodes'
    ))

    # Draw the paths
    for i, path in enumerate(paths):
        optimal_x = [node[0] for node in path]
        optimal_y = [node[1] for node in path]
        optimal_z = [node[2] for node in path]
        fig.add_trace(go.Scatter3d(
            x=optimal_x,
            y=optimal_y,
            z=optimal_z,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            name=f'Path {i+1}'
        ))

    # Set axis labels
    fig.update_layout(
        title="3D Pipe Design Visualization",
        scene=dict(
            xaxis=dict(title="X Coordinate"),
            yaxis=dict(title="Y Coordinate"),
            zaxis=dict(title="Z Coordinate")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    #fig.show()
    # Serialize the Plotly figure to JSON
    fig_json = fig.to_json()

    return paths, totalPrice, fig_json

