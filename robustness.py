import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rtamt
import sys

# Define the environment (10x10 grid with obstacles)
maze_size = (10, 10)
goal_pos = (9, 9)

# Create the maze (1 for obstacles, 0 for free space)
maze = np.zeros(maze_size)
obstacles = [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (4, 4), (5, 5), (6, 6), (7, 7)]


# Create STL specification to monitor distance to the goal
spec = rtamt.STLSpecification()
spec.declare_var('d_goal', 'float')  # Declare the distance to the goal

# STL formula to check if the distance to the goal is less than 0.1 within 50 steps
spec.spec = 'eventually[0, 1] (d_goal == 0)'

try:
    spec.parse()
    spec.pastify()
except rtamt.RTAMTException as err:
    print('RTAMT Exception: {}'.format(err))
    sys.exit()

# Function to calculate distance from a point (i, j) to the goal
def calculate_distance(x, y, goal_pos):
    goal_x, goal_y = goal_pos
    return np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)

# Function to calculate robustness based on STL for each cell
def calculate_stl_robustness(maze, goal_pos, spec):
    robustness = np.zeros(maze.shape)
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):

                # Calculate the Euclidean distance to the goal from position (i, j)
                distance_to_goal = calculate_distance(i, j, goal_pos)
                stl_time = 5  # We use time step 0 since we evaluate a single state
                
                # Evaluate robustness based on distance to the goal
                robustness[i, j] = spec.update(0, [('d_goal', distance_to_goal)])
    return robustness

# Calculate the robustness for each cell
robustness_values = calculate_stl_robustness(maze, goal_pos, spec)

# Visualize robustness heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(robustness_values, annot=True, cmap="coolwarm", cbar=True)
plt.title('STL-Based Robustness (Distance to Goal < 0.1)')
plt.show()
