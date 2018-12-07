import random

def generate_random_maze(w,h):
    maze = []
    for y in range(h):
        row = []
        # populate the rows with either a 0 or 1
        for x in range(w):
            row.append(random.randint(0,1))
        maze.append(row)
    return maze

def count_walls(maze):
    wall_count = 0
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            wall_count += maze[y][x]
    return wall_count

def get_neighbors(maze, position):
    neighbors = []
    x = position[0]
    y = position[1]
    for xi,yi in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
        if (xi >= 0 and xi < len(maze[0])) and (yi >= 0 and yi < len(maze)):
            if maze[yi][xi] == 0 and not (xi == x and yi == y):
                neighbors.append((xi,yi))
    return neighbors

def bfs_floors(maze):
    start = None
    # start with the first 0 found scanning left to right, top to bottom
    while start == None:
        for y in range(len(maze)):
            for x in range(len(maze[0])):
                if maze[y][x] == 0:
                    start = (x,y)
    open_list = [start]
    visited = []
    while open_list:
        tile = open_list.pop(0)
        if tile not in visited:
            visited.append(tile)
            neighbors = get_neighbors(maze,tile)
            for n in neighbors:
                open_list.append(n)
    # compares the number of visited tiles to the number of floors (the complement of the number of walls) in the maze
    return(len(visited) == (len(maze)*len(maze[0]) - count_walls(maze)))

def print_maze(maze):
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            print(maze[y][x], end = "")
        print()

maze = generate_random_maze(5,5)

while(not bfs_floors(maze)):
    maze = generate_random_maze(5,5)

print_maze(maze)
print("Floors:", len(maze)*len(maze[0]) - count_walls(maze))
print("Walls:", count_walls(maze))
print("Connected:", bfs_floors(maze))
