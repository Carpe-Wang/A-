import streamlit as st
from typing import List, Tuple, Dict, Callable

# Cost Dictionary
COSTS = {'🌾': 1, '🌲': 3, '⛰': 5, '🐊': 7}

# Possible moves
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# This list will hold the output messages to be displayed after the search
search_output = []


# Purpose:The heuristic function is used in the A* search algorithm to provide an estimated cost from the current
# location to the goal. This estimation aids the algorithm in prioritizing paths that are more likely to be closer to
# the goal.
def heuristic(location: Tuple[int, int], goal: Tuple[int, int]):
    # Use the Manhattan distance for the heuristic.
    x_distance = abs(goal[0] - location[0])
    y_distance = abs(goal[1] - location[1])
    return x_distance + y_distance

# Purpose: Implements the A* search algorithm to find the shortest path from a start to a goal point.
def a_star_search(world: List[List[str]], start: Tuple[int, int],
                  goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]],
                  heuristic: Callable) -> List[Tuple[int, int]]:
    """
    Perform A * search to find the shortest path from start to goal on a grid.

    This function implements the A* search algorithm, which is a pathfinding and graph traversal algorithm.
    It finds the shortest path from a start node to a goal node on a grid, taking into account different
    movement costs for various terrain types. It uses a heuristic to guide the search towards the goal.

    Parameters:
    - world (List[List[str]]): A grid represented by a list of lists of strings, where each string
      represents the terrain at that grid location.
    - start (Tuple[int, int]): The starting position on the grid as a tuple of (x, y) coordinates.
    - goal (Tuple[int, int]): The goal position on the grid as a tuple of (x, y) coordinates.
    - costs (Dict[str, int]): A dictionary mapping terrain symbols to their movement cost.
    - moves (List[Tuple[int, int]]): A list of tuples representing the possible moves from a
      position on the grid. Typically, this would include moves to adjacent squares.
    - heuristic (Callable): A heuristic function that estimates the cost to reach the goal from a node.
      The heuristic function takes two tuples of coordinates, the current position and the goal position.

    Returns:
    - List[Tuple[int, int]]: A list of tuples where each tuple represents a move along the shortest
      path found from start to goal. If no path is found, an empty list is returned.

    The A * search algorithm maintains an open set of nodes to explore, and uses the heuristic combined
    with the known cost so far to determine the most promising path to take at each step. Once the goal
    is reached, the path is reconstructed from the goal to the start using the recorded information
    about which node came from where.
    """
    open_set, came_from, g_cost, move_from = initialize_search(start, heuristic, goal)
    found, path = main_search_loop(world, open_set, came_from, g_cost, move_from, goal, costs, moves, heuristic)
    if found:
        return path
    return []

# Purpose: The `initialize_search` function is used to initialize key data structures for the A* search algorithm.
# This is the first step in the A* algorithm, responsible for setting up the initial state of the search.
def initialize_search(start: Tuple[int, int], heuristic: Callable,
                      goal: Tuple[int, int]) -> Tuple[List, Dict, Dict, Dict]:
    """
    Initializes the necessary data structures for the A* search algorithm.

    This function sets up the initial state required by the A* algorithm to start the search process.
    It creates and initializes the open set with the starting node, along with dictionaries for tracking
    the origin of each node (came_from), the cost of getting to each node (g_cost), and the move made to
    reach each node (move_from).

    Parameters:
    - start (Tuple[int, int]): The starting position on the grid as a tuple of (x, y) coordinates.
    - heuristic (Callable): The heuristic function used to estimate the cost from any node to the goal.
      It takes two arguments: the current node and the goal node, both represented as (x, y) coordinates.
    - goal (Tuple[int, int]): The goal position on the grid as a tuple of (x, y) coordinates.

    Returns:
    - Tuple[List, Dict, Dict, Dict]: A tuple containing four elements in the following order:
        - open_set (List): A list of tuples, where each tuple contains the estimated total cost (f_cost)
          from start to goal through this node, and the node's coordinates.
        - came_from (Dict): A dictionary mapping each node to the node it directly came from.
        - g_cost (Dict): A dictionary that stores the cost of the cheapest path from start to each node.
        - move_from (Dict): A dictionary that records the move (as a tuple of changes in x and y) made
          to reach each node.
    """
    open_set = [(heuristic(start, goal), start)]
    came_from = {start: None}
    g_cost = {start: 0}
    move_from = {start: None}
    return open_set, came_from, g_cost, move_from

def main_search_loop(world, open_set, came_from, g_cost, move_from, goal, costs, moves, heuristic):
    """
    Executes the main search loop of the A* algorithm to find the shortest path.

    This function iterates over an open set of nodes, examining each node's neighbors to determine the next best step.
    The path is determined by calculating the cost `g(n)` of the path from the start node to the current node `n`,
    adding the heuristic cost `h(n)` from `n` to the goal, and selecting the node with the lowest total cost `f(n)`.

    Parameters:
    - world (list of list of str): The 2D grid representing the world map.
    - open_set (list of tuples): The list of nodes to be evaluated, sorted by their f_cost.
    - came_from (dict): A dictionary that records where each node was reached from.
    - g_cost (dict): A dictionary that stores the cost of the path from the start node to each node.
    - move_from (dict): A dictionary that stores the move made to reach each node.
    - goal (tuple of int): The target node coordinates.
    - costs (dict): A dictionary mapping terrain types to their traversal cost.
    - moves (list of tuples): The list of possible moves from each node.
    - heuristic (function): The heuristic function used to estimate the cost from `n` to the goal.

    Returns:
    - A tuple (bool, list of tuples), where the boolean indicates if the goal was reached,
      and the list is the path of moves to get to the goal if the goal was reached.

    The function appends messages to a global `search_output` list, intended to store log messages
    for the search progress.
    """

    while open_set:
        open_set.sort(key=lambda x: x[0])
        current_cost, current_node = open_set.pop(0)  # Get the node with the lowest f_cost
        search_output.append(f"\nCurrent node: {current_node}, Cost: {current_cost}")
        if current_node == goal:
            return True, reconstruct_path(move_from, current_node)
        neighbors_info = []
        for move in moves:
            neighbor = (current_node[0] + move[0], current_node[1] + move[1])
            if 0 <= neighbor[1] < len(world) and 0 <= neighbor[0] < len(world[0]) and world[neighbor[1]][neighbor[0]] != '🌋':
                tentative_g = g_cost[current_node] + costs[world[neighbor[1]][neighbor[0]]]
                h_cost = heuristic(neighbor, goal)
                f_cost = tentative_g + h_cost
                neighbors_info.append((neighbor, tentative_g, h_cost, f_cost))
                search_output.append(
                    f"Checking neighbor: {neighbor}, g_cost: {tentative_g}, h_cost: {h_cost}, f_cost: {f_cost}")
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    open_set.append((f_cost, neighbor))
                    came_from[neighbor] = current_node
                    move_from[neighbor] = move
        # Find the neighbor with the smallest f_cost
        if neighbors_info:
            best_neighbor = min(neighbors_info, key=lambda x: x[3])
            search_output.append(f"Moving to node with lowest f_cost: {best_neighbor[0]}, f_cost: {best_neighbor[3]}")
            # Move to the best neighbor
            current_node = best_neighbor[0]
        if not open_set:
            return False, []
    return False, []


def reconstruct_path(move_from: Dict, current_node: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from the start node to the goal node.

    Given a dictionary mapping each node to the node it came from and the goal node,
    this function traces back the path from the goal to the start. The path is
    reconstructed in reverse, starting from the goal node and following the 'came from'
    links until the start node is reached.

    Parameters:
    - move_from (Dict[Tuple[int, int], Tuple[int, int]]): A dictionary that records
      the node from which each node was reached.
    - current_node (Tuple[int, int]): The goal node from which to start tracing back
      the path.

    Returns:
    - List[Tuple[int, int]]: A list of moves (as coordinate deltas) that represents
      the path from the start node to the goal node.
    """
    path = []
    while current_node in move_from and move_from[current_node]:
        move = move_from[current_node]
        path.append(move)
        current_node = (current_node[0] - move[0], current_node[1] - move[1])
    return path[::-1]


def pretty_print_path(world: List[List[str]], path: List[Tuple[int, int]],
                      start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    """
     Visualize the path on the world map and calculate its total cost.

     This function takes a 2D world map and a path defined by a list of moves,
     and prints out the world map with the path visualized using directional symbols.
     It also calculates the total cost of traversing the path based on the provided
     costs for each terrain type. The starting point is marked by the starting coordinates,
     and the goal is indicated with a '🎁' symbol.

     Parameters:
     - world (List[List[str]]): A 2D list representing the world map where each element
       is a string representing the terrain type.
     - path (List[Tuple[int, int]]): A list of tuples where each tuple represents the
       directional move taken from the current position.
     - start (Tuple[int, int]): A tuple representing the starting coordinates on the map.
     - goal (Tuple[int, int]): A tuple representing the goal coordinates on the map.
     - costs (Dict[str, int]): A dictionary mapping terrain types to their traversal cost.

     Returns:
     - Tuple[int, List[List[str]]]: A tuple containing the total path cost as an integer
       and the modified world map as a 2D list with the path visualized.
     """
    path_cost = 0
    current = start
    world_copy = [row[:] for row in world]
    for move in path:
        direction_symbol = get_direction_symbol(move)
        next_step = (current[0] + move[0], current[1] + move[1])
        path_cost += costs.get(world[next_step[1]][next_step[0]], float('inf'))
        world_copy[current[1]][current[0]] = direction_symbol
        current = next_step
    gx, gy = goal
    world_copy[gy][gx] = '🎁'
    for row in world_copy:
        print(' '.join(row))
    return path_cost, world_copy

def get_direction_symbol(move):
    """
    Convert a move tuple into a directional arrow symbol.

    This function takes a tuple representing a move on a grid and returns a
    corresponding arrow symbol that indicates the direction of the move. The grid
    is assumed to be oriented such that:
        - (1, 0) represents a move to the right
        - (-1, 0) represents a move to the left
        - (0, 1) represents a move downwards
        - (0, -1) represents a move upwards

    Parameters:
    - move (tuple): A 2-element tuple where the first element is the horizontal
      change (positive for right, negative for left), and the second element is
      the vertical change (positive for down, negative for up).

    Returns:
    - str: A string containing a single character: an arrow symbol pointing in
      the direction of the move. If the move is not one of the recognized
      directions, a question mark '?' is returned.
    """
    # Determine the direction symbol based on the move coordinates.
    if move == (1, 0): return '⏩'  # Right
    if move == (-1, 0): return '⏪'  # Left
    if move == (0, 1): return '⏬'  # Down
    if move == (0, -1): return '⏫'  # Up
    # If the move doesn't match any known direction, return a default symbol.
    return '?'


# #In line with full_world from mod1, convenient for testing.
small_world_map = """
🌾🌲🌲🌲🌲🌲🌲
🌾🌲🌲🌲🌲🌲🌲
🌾🌲🌲🌲🌲🌲🌲
🌾🌾🌾🌾🌾🌾🌾
🌲🌲🌲🌲🌲🌲🌾
🌲🌲🌲🌲🌲🌲🌾
🌲🌲🌲🌲🌲🌲🌾
"""

#In line with full_world from mod1, convenient for testing.
full_world_map = """
🌾🌾🌾🌾🌾🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌾🌾🌾🌾🌾🌾🌾🌾🌾🌾🌾🌾
🌾🌾🌾🌾🌾🌾🌾🌲🌲🌲🌲🌲🌲🌲🌲🌲🌾🌾🌋🌋🌋🌋🌋🌋🌋🌾🌾
🌾🌾🌾🌾🌋🌋🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌋🌋🌋⛰⛰⛰🌋🌋⛰⛰
🌾🌾🌾🌾⛰🌋🌋🌋🌲🌲🌲🌲🐊🐊🌲🌲🌲🌲🌲🌾🌾⛰⛰🌋🌋⛰🌾
🌾🌾🌾⛰⛰🌋🌋🌲🌲🌾🌾🐊🐊🐊🐊🌲🌲🌲🌾🌾🌾⛰🌋🌋🌋⛰🌾
🌾⛰⛰⛰🌋🌋⛰⛰🌾🌾🌾🌾🐊🐊🐊🐊🐊🌾🌾🌾🌾🌾⛰🌋⛰🌾🌾
🌾⛰⛰🌋🌋⛰⛰🌾🌾🌾🌾⛰🌋🌋🌋🐊🐊🐊🌾🌾🌾🌾🌾⛰🌾🌾🌾
🌾🌾⛰⛰⛰⛰⛰🌾🌾🌾🌾🌾🌾⛰🌋🌋🌋🐊🐊🐊🌾🌾⛰⛰⛰🌾🌾
🌾🌾🌾⛰⛰⛰🌾🌾🌾🌾🌾🌾⛰⛰🌋🌋🌾🐊🐊🌾🌾⛰⛰⛰🌾🌾🌾
🌾🌾🌾🐊🐊🐊🌾🌾⛰⛰⛰🌋🌋🌋🌋🌾🌾🌾🐊🌾⛰⛰⛰🌾🌾🌾🌾
🌾🌾🐊🐊🐊🐊🐊🌾⛰⛰🌋🌋🌋⛰🌾🌾🌾🌾🌾⛰🌋🌋🌋⛰🌾🌾🌾
🌾🐊🐊🐊🐊🐊🌾🌾⛰🌋🌋⛰🌾🌾🌾🌾🐊🐊🌾🌾⛰🌋🌋⛰🌾🌾🌾
🐊🐊🐊🐊🐊🌾🌾⛰⛰🌋🌋⛰🌾🐊🐊🐊🐊🌾🌾🌾⛰🌋⛰🌾🌾🌾🌾
🌾🐊🐊🐊🐊🌾🌾⛰🌲🌲⛰🌾🌾🌾🌾🐊🐊🐊🐊🌾🌾⛰🌾🌾🌾🌾🌾
🌾🌾🌾🌾🌋🌾🌾🌲🌲🌲🌲⛰⛰⛰⛰🌾🐊🐊🐊🌾🌾⛰🌋⛰🌾🌾🌾
🌾🌾🌾🌋🌋🌋🌲🌲🌲🌲🌲🌲🌋🌋🌋⛰⛰🌾🐊🌾⛰🌋🌋⛰🌾🌾🌾
🌾🌾🌋🌋🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌋🌋🌋🌾🌾🌋🌋🌋🌾🌾🌾🌾🌾
🌾🌾🌾🌋🌋🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌋🌋🌋🌋🌾🌾🌾🌾🌾🌾🌾
🌾🌾🌾🌋🌋🌋🌲🌲🌲🌲🌲🌲🌲🌲🌾🌾🌾⛰⛰🌾🌾🌾🌾🌾🌾🌾🌾
🌾🌾🌾🌾🌋🌋🌋🌲🌲🌲🌲🌲🌲🌾🌾🌾🌾🌾🌾🌾🌾🌾🌾🐊🐊🐊🐊
🌾🌾⛰⛰⛰⛰🌋🌋🌲🌲🌲🌲🌲🌾🌋🌾🌾🌾🌾🌾🐊🐊🐊🐊🐊🐊🐊
🌾🌾🌾🌾⛰⛰⛰🌋🌋🌋🌲🌲🌋🌋🌾🌾🌾🌾🌾🌾🐊🐊🐊🐊🐊🐊🐊
🌾🌾🌾🌾🌾🌾⛰⛰⛰🌋🌋🌋🌋🌾🌾🌾🌾⛰⛰🌾🌾🐊🐊🐊🐊🐊🐊
🌾⛰⛰🌾🌾⛰⛰⛰⛰⛰🌾🌾🌾🌾🌾⛰⛰🌋🌋⛰⛰🌾🐊🐊🐊🐊🐊
⛰🌋⛰⛰⛰⛰🌾🌾🌾🌾🌾🌋🌋🌋⛰⛰🌋🌋🌾🌋🌋⛰⛰🐊🐊🐊🐊
⛰🌋🌋🌋⛰🌾🌾🌾🌾🌾⛰⛰🌋🌋🌋🌋⛰⛰⛰⛰🌋🌋🌋🐊🐊🐊🐊
⛰⛰🌾🌾🌾🌾🌾🌾🌾🌾🌾🌾⛰⛰⛰⛰⛰🌾🌾🌾🌾⛰⛰⛰🌾🌾🌾
"""

def main():
    st.title("A* Search Demo")

    st.write("""
    The A* search algorithm is a smart way to find the shortest path from one point to another. It works somewhat like planning a route on a map:
    
    - **Starting Point & Destination**: First, the algorithm needs to know where you're starting from and where you want to go.
    
    - **Calculating Steps**: Then, it looks at every step you could take and calculates two costs:
        1. `g(n)`: The cost of the path from the start node to `n`, counting each step you've taken.
        2. `h(n)`: An estimated cost from `n` to the goal. It's like a guess of how far you are from the finish line.
        
    - **Choosing the Best Path**: The algorithm adds these two costs to get `f(n)`, the total estimated cost of a path going through `n`.
    
    - **Loop Until the End**: It repeats this process, always choosing the step with the lowest `f(n)`, which is like picking the path that looks cheapest at that moment, until it reaches the end.
    
    It's like solving a maze, trying to find a path that doesn't stray from the goal and isn't too hard to walk.
    """)

    map_choice = st.radio("Select Map", ('Custom Map', 'small_world', 'full_world'))
    if map_choice == 'small_world':
        user_map = small_world_map
    elif map_choice == 'full_world':
        user_map = full_world_map
    else:
        user_map = st.text_area("Please enter a map, using the symbol '🌾' for plains cost 1, '🌲' for forest cost 3, '⛰' for hills cost 5, '🐊' for swamp cost 7, and '🌋' for mountains is impassible.",
                            "🌾🌲🌲🌲🌲🌲🌲\n🌾🌲🌲🌲🌲🌲🌲\n🌾🌲🌲🌲🌲🌲🌲\n🌾🌾🌾🌾🌾🌾🌾\n🌲🌲🌲🌲🌲🌲🌾\n🌲🌲🌲🌲🌲🌲🌾\n🌲🌲🌲🌲🌲🌲🌾",height=300)
    # Converting user input map to a 2D list
    if map_choice in ['small_world', 'full_world']:
        st.text_area("Map Preview", user_map, height=300)
    world = [list(row) for row in user_map.split('\n') if row]

    # Getting start and goal points
    start_x = st.number_input("Start X Coordinate", value=0, min_value=0, max_value=len(world[0]) - 1)
    start_y = st.number_input("Start Y Coordinate", value=0, min_value=0, max_value=len(world) - 1)
    goal_x = st.number_input("Goal X Coordinate", value=len(world[0]) - 1, min_value=0, max_value=len(world[0]) - 1)
    goal_y = st.number_input("Goal Y Coordinate", value=len(world) - 1, min_value=0, max_value=len(world) - 1)
    start = (start_x, start_y)
    goal = (goal_x, goal_y)

    # Performing A* Search Algorithm
    if st.button('Execute A* Search'):
        path = a_star_search(world, start, goal, COSTS, MOVES, heuristic)
        path_cost, path_world = pretty_print_path(world, path, start, goal, COSTS)

        # Displaying results
        st.write("Searched Result Map and Path:")
        st.text("\n".join(" ".join(row) for row in path_world))

        st.write(f"Total Path Cost: {path_cost}")
        st.write(f"Path: {path}")

        st.write("Search Process:")
        for output in search_output:
            st.text(output)


# Streamlit application entry point
if __name__ == "__main__":
    main()