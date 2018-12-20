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

def generate_random_rule():
    rule = [random.randint(0,1) for i in range(257)]

    # generate a random number of iterations between 1 and 5
    rule[256] = random.randint(1,5)
    return rule

def breed_rule(r1,r2):
    rule = []
    for i in range (0,len(r1)-1):
        if random.random() <= 0.5:
            rule.append(r1[i])
        else:
            rule.append(r2[i])
    # make the offspring iteration count a random number in the range of the parents'
    if r1[256] != r2[256]:
        rule.append(random.randint(min(r1[256],r2[256]),max(r1[256],r2[256])))
    else:
        rule.append(r1[256])
    return rule

# default 20% chance for rules to flip within the maze
def mutate_rule(rule):
    # 20% chance to increment the number of iterations up or down
    if random.random() <= 0.2:
        if random.random() <= 0.5:
            rule[len(rule)-1] += 1
        else:
            rule[len(rule)-1] += -1

    # to avoid 0 or negative iterations, set the minimum count to 1
    rule[len(rule)-1] = max(1,rule[len(rule)-1])

    for i in range (0,len(rule)-1):
        if random.random() <= 0.2:
            rule[i] = abs(rule[i] - 1)
    return rule

# use a weighted probability distribution to select n/2 of the mazes for breeding
def roulette_selection(mazes):
    sum_fitness = total_fitness(mazes)

    # overwrite the fitness value of each maze with that value divided by the total fitness,
    # and append a third boolean value to determine if this candidate has already been selected
    for m in mazes:
        m[1] = m[1]/sum_fitness

    # sort the mazes by their selection probability
    mazes = sorted(mazes,key=lambda x: x[1])

    offspring = []

    # continuously shrink the pool of mazes as parents are pulled out
    maze_pool = mazes.copy()
    for i in range(int(len(mazes)/2)):

        # initialize an (eventual) tuple of parents
        parents = []
        for j in range(2):
            # initialize increment and threshold
            increment = 0
            threshold = random.random()
            for m in maze_pool:
                increment += m[1]
                #print("Increment:",increment)
                #print("Threshold:",threshold)
                if increment >= threshold:# and not m[2]:
                    parents.append(m[2])
                    break

        # create two children by breeding the selected parents and then mutating the child
        offspring.append(mutate_rule(breed_rule(parents[0],parents[1])))
        offspring.append(mutate_rule(breed_rule(parents[0],parents[1])))
    return offspring

# scan the 8 neighbors around a central cell
# assumes we are only looking at the interior cells (no borders)
def scan_byte(maze,cell):
    byte = []
    # iterate through the 8 neighbors left to right, top to bottom
    for y in range(cell[1]-1,cell[1]+2):
        for x in range(cell[0]-1,cell[0]+2):
            #exclude the middle cell
            if not (x == cell[0] and y == cell[1]):
                byte.append(maze[y][x])
    return byte

# a byte (generated from a left-to-right 3x3 neighborhood parsing above)
# is used to look up and return the corresponding cell state in a given CA ruleset
def ca_lookup(rules,byte):
    index = 0
    byte.reverse()
    # iterate through the 8 bits, incrementing the power of 2 to create a decimal number
    for i in range(len(byte)):
        index += byte[i]*(2**i)

    return rules[index]

# iterates across the maze, updating cells according to the paired rule
def iterate_ca(pair):
    maze = pair[0]
    rule = pair[1]
    for i in range(rule[256]):
        for y in range(1, len(maze)-1):
            for x in range(1, len(maze[0])-1):
                maze[y][x] = ca_lookup(rule,scan_byte(maze,(x,y)))
    return [maze,rule]

# find the sum of the fitness scores of all the mazes for usage in selection
def total_fitness(mazes):
    total_fitness = 0
    for m in mazes:
        total_fitness+=m[1]
    return total_fitness

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

def score_population(pairs):
    pop = []
    for p in pairs:
        pop.append([p[0],evaluate_fitness(p[0]),p[1]])
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
    # start with a random generation of 10 rulesets on 10 mazes
    mazes = [generate_random_maze(10,10) for i in range(30)]
    rules = [generate_random_rule() for i in range(30)]

    current_gen = [[mazes[i],rules[i]] for i in range(len(mazes))]
    print("First generation")
    output_mazes(current_gen)

    # over 100 generations, evolve the maze population
    for i in range(100):
        # run the iteration process on the pairs of rule and maze
        for j in range(len(current_gen)):
            current_gen[j] = iterate_ca(current_gen[j])
        scored_pop = score_population(current_gen)
        new_rules = roulette_selection(scored_pop)
        current_gen = [[mazes[i],new_rules[i]] for i in range(len(mazes))]

    print("Final generation")
    output_mazes(current_gen)

evolve()
