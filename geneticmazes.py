import math
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

def breed_maze(m1,m2):
    maze = []
    for y in range(len(m1)):
        row = []
        for x in range (len(m1[0])):
            if random.random() <= 0.5:
                row.append(m1[y][x])
            else:
                row.append(m2[y][x])
        maze.append(row)
    return maze

# default 20% chance for cells to flip within the maze
def mutate_maze(maze):
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if random.random() <= 0.2:
                maze[y][x] = abs(maze[y][x] - 1)
    return maze

# find the sum of the fitness scores of all the mazes for usage in selection
def total_fitness(mazes):
    total_fitness = 0
    for m in mazes:
        total_fitness+=m[1]
    return total_fitness

# use a weighted probability distribution to select n/2 of the mazes for breeding
def roulette_selection(mazes,size,truncate,unique_parents):
    # sort the mazes by their selection probability
    mazes = sorted(mazes,key=lambda x: x[1])

    offspring = []

    # continuously shrink the pool of mazes as parents are pulled out

    # if truncate is set to true, we immediately prune the bottom quarter of the population
    if truncate:
        maze_pool = mazes[:int(3*len(mazes)/4)]
    else:
        maze_pool = mazes.copy()

    sum_fitness = total_fitness(maze_pool)

    # overwrite the fitness value of each maze with that value divided by the total fitness
    for m in maze_pool:
        m[1] = m[1]/sum_fitness

    # creates pairs of parents to create two children each
    for i in range(int(size/2)):
        # initialize an (eventual) tuple of selected parents
        parents = []
        while len(parents) < 2:
            # initialize increment and threshold
            increment = 0
            threshold = random.random()
            for m in maze_pool:
                increment += m[1]
                # checks whether the roulette ball has fallen into a slot and
                if increment >= threshold:
                    parents.append(m[0])
                    # in the case of unique_parents = True and there are enough candidates left given the size
                    # removes the parent from the pool
                    if(unique_parents and len(maze_pool) > int((size-len(offspring))/2)):
                        maze_pool.remove(m)
                        # need to update the fitness values to ensure all of the values can be selected
                        sum_fitness = total_fitness(maze_pool)
                        for m in maze_pool:
                            m[1] = m[1]/sum_fitness
                    break
        # create two children by breeding the selected parents and then mutating the child
        offspring.append(mutate_maze(breed_maze(parents[0],parents[1])))
        offspring.append(mutate_maze(breed_maze(parents[0],parents[1])))
    return offspring

def count_floors(maze):
    floor_count = 0
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            floor_count += (maze[y][x] + 1) % 2
    return floor_count

def count_horiz(maze):
    horiz_count = 0

    for y in range(len(maze)):
        # we want to make sure the block isn't 1x1, so we use a boolean and look-ahead check
        counting = False
        for x in range(len(maze[0])):
            if counting:
                if maze[y][x] == 1:
                    horiz_count += 1
                else:
                    counting = False
            else:
                if maze[y][x] == 1 and (x + 1 < len(maze[0]) and maze[y][x+1] == 1):
                    horiz_count += 1
                    counting = True
    return horiz_count

def count_vert(maze):
    vert_count = 0

    for x in range(len(maze[0])):
        # we want to make sure the block isn't 1x1, so we use a boolean and look-ahead check
        counting = False
        for y in range(len(maze)):
            if counting:
                if maze[y][x] == 1:
                    vert_count += 1
                else:
                    counting = False
            else:
                if maze[y][x] == 1 and (y + 1 < len(maze[0]) and maze[y+1][x] == 1):
                    vert_count += 1
                    counting = True
    return vert_count

# NOTE: could be improved with a helper function for the linear traversals
# and could stop once it found the maximum distance for a region,
# so for example if the grid was 5x5, it could halt once it found
# sqrt(5^2+5^2) ~ 7.01 at the outer level
def get_scatter(maze):
    row_start = 0
    col_start = 0
    row_end=len(maze);
    col_end=len(maze[0]);

    measuring = False
    start = (0,0)
    max_distance = 0

    # we want to iterate through the matrix in a spiral starting from the border to find the furthest-separated pair of floor tiles
    while(row_start <= row_end and col_start <= col_end):
        # go across the top from left to right
        for i in range(col_start,col_end):
            if maze[row_start][i] == 0:
                if not measuring:
                    measuring = True
                    start = (i,row_start)
                # we update the max distance using the pythagorean theorem on the first-found floor and every subsequent floor
                else:
                    max_distance = max(max_distance, math.sqrt((start[0] - i - 1)**2 + (start[1] - row_start - 1)**2))

        # go along the right from top to bottom
        for j in range(row_start+1,row_end):
            if maze[j][col_end-1] == 0:
                if not measuring:
                    measuring = True
                    start = (col_end-1,j)
                # we update the max distance using the pythagorean theorem on the first-found floor and every subsequent floor
                else:
                    max_distance = max(max_distance, math.sqrt((start[0] - col_end)**2 + (start[1] - j - 1)**2))

        # go across the bottom from right to left
        for i in range(col_end-2,col_start-1,-1):
            if maze[row_end-1][i] == 0:
                if not measuring:
                    measuring = True
                    start = (i,row_end-1)
                # we update the max distance using the pythagorean theorem on the first-found floor and every subsequent floor
                else:
                    max_distance = max(max_distance, math.sqrt((start[0] - i - 1)**2 + (start[1] - row_end)**2))

        # go along the left from bottom to top
        for j in range(row_end-2,row_start,-1):
            if maze[j][row_start] == 0:
                if not measuring:
                    measuring = True
                    start = (row_start,j)
                # we update the max distance using the pythagorean theorem on the first-found floor and every subsequent floor
                else:
                    max_distance = max(max_distance, math.sqrt((start[0] - row_start - 1)**2 + (start[1] - j - 1)**2))

        row_start+=1
        col_start+=1
        row_end+=-1
        col_end+=-1

    return max_distance

# returns a list of the 4-way neighbors that are floor tiles
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
    # compares the number of visited tiles to the number of floors in the maze
    return(len(visited) == (count_floors(maze)))

def evaluate_fitness(maze):
    score = 0
    # this parameter is very crucial to ensuring the final generation is connected
    if bfs_floors(maze):
        score += 10
    # maximize the ratio of distance over total walls
    score += (get_scatter(maze)/(len(maze)**2-count_floors(maze)))*0.1
    # balance the ratio of horizontal to vertical Walls
    score += max(0,(1-count_horiz(maze))/(count_vert(maze)+1)*0.1)

    # bound scores to 0 to avoid negative scores
    score = max(0,score)
    return score

def score_population(mazes):
    pop = []
    for m in mazes:
        pop.append([m,evaluate_fitness(m)])
    return pop

def print_maze(maze):
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            print(maze[y][x], end = "")
        print()
    print()

def output_mazes(mazes):
    scored_mazes = score_population(mazes)
    scored_mazes = sorted(scored_mazes,key=lambda x: x[1])

    for i in range(len(scored_mazes)):
        print_maze(scored_mazes[i][0])
        print("Fitness value:",scored_mazes[i][1])
        #print("Connected:",bfs_floors(scored_mazes[i][0]))


def evolve():
    # start with a random generation of 10 5x5 mazes
    current_gen = [generate_random_maze(5,5) for i in range(10)]
    print("First generation")
    output_mazes(current_gen)
    # over 100 generations, evolve the maze population
    for i in range(100):
        scored_pop = score_population(current_gen)
        # the second argument is the number of returned offspring,
        # the third argument enables bottom elimination,
        # the fourth prevents reusing parents when set to true
        current_gen = roulette_selection(scored_pop,10,False,False)

    print("Final generation")
    output_mazes(current_gen)

evolve()
