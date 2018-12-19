# GeneticMazes
Currently the cellular automata does not work, but to experiment with the standard GA, simply run geneticmazes.py and it will print out the final generation of the evolutionary cycle. You can tweak the number of samples, maze size, and the fitness parameters to get different results. The most important parameter seems to be the weight assigned to BFS. A high value for this coupled with many generations tends to result in about half of the final population being connected (and this might really be a higher proportion before the mutation process is applied to the final generation, which might corrupt pathways with walls 20% of the time).
