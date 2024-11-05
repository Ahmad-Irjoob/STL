import numpy as np
import matplotlib.pyplot as plt
import random
import rtamt  # Library for STL specifications
import sys

# Define the 10x10 maze environment
maze_size = (10, 10)
goal1 = (3, 3)  # Define goal1 position
goal2 = (4, 4)  # Define goal2 position
goal3 = (6, 6)  # Define goal3 position
start_pos = (0, 0)  # Define the starting position
Robustness1=0
# Create the maze (0 for free space, 1 for obstacles)
maze = np.zeros(maze_size)
obstacles = [(1, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 5), (8, 8)]  # Example obstacles
for obs in obstacles:
    maze[obs] = 1

# Define the STL specification for visiting goal1 and (goal2 or goal3)
spec = rtamt.STLSpecification()
spec.declare_var('x', 'float')
spec.declare_var('y', 'float')

# STL formula for visiting goal1 and (goal2 or goal3)
spec.spec = f'eventually[0,10] (x == {goal1[0]} and y == {goal1[1]}) and (eventually[0,10] (x == {goal2[0]} and y == {goal2[1]}) or eventually[0,10] (x == {goal3[0]} and y == {goal3[1]}))'

# Parse and compile the STL specification
try:
    spec.parse()
    spec.pastify()
except rtamt.RTAMTException as err:
    print('RTAMT Exception: {}'.format(err))
    sys.exit()

# Function to check if a position is within maze bounds and not an obstacle
def is_valid_position(position, maze):
    x, y = position
    return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] != 1

# Function to calculate the Manhattan distance from a position to the goal
def manhattan_distance(position, goal_pos):
    return abs(position[0] - goal_pos[0]) + abs(position[1] - goal_pos[1])

# Function to move the robot with goal prioritization and minimize the robustness
def move_robot_with_goal_prioritization(start_pos, maze, spec, goals, max_steps=20):
    position = start_pos
    path = [position]  # Keep track of the robot's path
    steps = 0
    stl_satisfaction = False
    accumulative_robustness = 0  # Initialize accumulative robustness
    robustness_values = {}  # Dictionary to store robustness values for each cell
    completed_goals = set()  # Track goals that have been completed
    current_goal = goals.pop(0)  # Start with the first goal in the list

    # Calculate initial robustness value
    current_robustness = spec.update(steps, [('x', float(position[0])), ('y', float(position[1]))])
    robustness_values[position] = current_robustness
    accumulative_robustness += current_robustness
    print(f"Step {steps} - Position {position} - Robustness: {current_robustness:.4f} - Accumulative Robustness: {accumulative_robustness:.4f}")

    while steps < max_steps:
        # Track best move, accumulative robustness, and heuristic
        best_move = None
        best_next_position = position
        best_accumulative_robustness = float('inf')  # Initialize with a very high value
        best_robustness = current_robustness
        best_heuristic_value = float('inf')  # Initialize with a very high value

        # Check all possible directions
        directions = {'up': (position[0] - 1, position[1]),
                      'down': (position[0] + 1, position[1]),
                      'left': (position[0], position[1] - 1),
                      'right': (position[0], position[1] + 1)}

        for move, next_position in directions.items():
            # Ensure the next position is valid
            if is_valid_position(next_position, maze):
                # Evaluate robustness for this move
                robustness = spec.update(steps + 1, [('x', float(next_position[0])), ('y', float(next_position[1]))])
                
                # Calculate the potential new accumulative robustness if this move is taken
                potential_accumulative_robustness = accumulative_robustness + robustness

                # Calculate the heuristic value based on robustness and distance to the current goal
                distance_to_current_goal = manhattan_distance(next_position, current_goal)
                heuristic_value = potential_accumulative_robustness + distance_to_current_goal  # Combine robustness with distance

                # Print heuristic value for each potential move
                print(f"Step {steps + 1} - Checking move {move} to {next_position} - Robustness: {robustness:.4f} - Distance to Current Goal: {distance_to_current_goal} - Heuristic Value: {heuristic_value:.4f}")

                # Choose the move with the smallest heuristic value (to minimize robustness)
                if heuristic_value < best_heuristic_value:
                    best_heuristic_value = heuristic_value
                    best_move = move
                    best_next_position = next_position
                    best_accumulative_robustness = potential_accumulative_robustness
                    best_robustness = robustness

        # If no valid move found, break the loop
        if best_move is None:
            print(f"No more valid moves at step {steps}. Ending simulation.")
            break

        # Update to the best move
        position = best_next_position
        path.append(position)
        steps += 1

        # Check if the current goal has been reached and update goals
        if position == current_goal:
            completed_goals.add(current_goal)  # Mark the current goal as completed
            if goals:  # Move to the next goal in the sequence if available
                current_goal = goals.pop(0)
                print(f"Goal {current_goal} reached! Moving to the next goal: {current_goal}")

        # Update accumulative robustness with the best move's robustness value
        accumulative_robustness = best_accumulative_robustness

        # Store robustness value for each cell visited
        robustness_values[position] = best_robustness
        print(f"Step {steps} - Position {position} - Robustness: {best_robustness:.4f} - Accumulative Robustness: {accumulative_robustness:.4f}")

        # Check if the robot satisfies the STL specification
        if best_robustness >= 0:
            stl_satisfaction = True
            break

    return path, stl_satisfaction, robustness_values, accumulative_robustness

# List of goals to visit in sequence
goals_sequence = [goal1, goal2, goal3]

# Run the robot in the maze with goal prioritization and capture the robustness values and accumulative robustness
path, satisfied, robustness_values, accumulative_robustness = move_robot_with_goal_prioritization(start_pos, maze, spec, goals_sequence)

print("\nFinal Path taken by the robot: {}".format(path))
print("STL Specification Satisfied: {}".format(satisfied))
print(f"Final Accumulative Robustness: {accumulative_robustness:.4f}")

# Print the robustness values for each cell visited
print("\nRobustness Values for each cell visited:")
for cell, robustness in robustness_values.items():
    Robustness1=Robustness1+robustness
    print(f"Position: {cell} - Robustness: {robustness:.4f} - Robustness: {Robustness1:.4f}")

# Visualize the maze and the path taken by the robot
plt.figure(figsize=(6, 6))
plt.imshow(maze, cmap='gray_r', origin='upper')  # Plot the maze (obstacles in black)
plt.scatter([start_pos[1]], [start_pos[0]], color='blue', s=100, label='Start Position')
plt.scatter([goal1[1]], [goal1[0]], color='green', s=100, label='Goal 1')
plt.scatter([goal2[1]], [goal2[0]], color='orange', s=100, label='Goal 2')
plt.scatter([goal3[1]], [goal3[0]], color='purple', s=100, label='Goal 3')
path = np.array(path)
plt.plot(path[:, 1], path[:, 0], marker='o', color='red', label='Robot Path')

# Annotate each visited cell with its robustness value and accumulative robustness
for position, robustness in robustness_values.items():
    plt.text(position[1], position[0], f'{robustness:.2f}', fontsize=12, ha='center', va='center', color='white')

plt.title(f'Robot Movement in the Maze\nFinal Accumulative Robustness: {accumulative_robustness:.4f}')
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis to match the maze coordinate system
plt.grid(True)
plt.show()
