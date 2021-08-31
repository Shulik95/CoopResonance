# implements Prim's random maze generation algorithm as follows:
# 1. Start with a grid of walls
# 2. Pick a cell, mark it as part of the maze, Add the walls of the cell to the
# walls of the list
# 3. while there are walls in the list:
#     1) Pick a random wall from the list. if only one of the two cells the wall
#     divides is visited then:
#         a)Make the wall a passage and mark the unvisited cell as part of the maze
#         b)Add the neighboring walls of the cell to the wall list
#     2)Remove the wall from the list


# -------- imports --------#
from colorama import init, Fore

# -- const definitions -- #
cell = 'c'
wall = 'w'
unvisited = 'u'


# ------ functions ------ #

def init_maze(width, height):
    """
    inits a two dim grid of unvisited cells.
    :param width: int - number of columns in grid
    :param height: int - number of rows in grid
    :return: list of lists of size height X width
    """
    maze = []
    for i in range(height):
        row = ["u" for j in range(width)] # creates row
        maze.append(row)
    return maze