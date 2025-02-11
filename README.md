# Project work (Symbolic regression with GP)
The genetic programming algorithm used to solve the symbolic regression problem has the following main characteristics:

## Individuals structure
Each individual is characterized by a tree structure where each node is an instance of the class `Node`, that specifies the value of the node, its parent node and the left and right children (if present).
The whole tree is embedded by an instance of the class `Tree`, that contains the root node and the fitness value of the tree.<br>
Each node can either be:
- a function node, expressing one of the math functions in the set `["add", "sub", "mul", "div", "log", "sin", "tan", "exp", "sqrt", "square"]`
- a terminal (leaf) node, expressing a variable `x1, x2,...` or a random float constant with a value in the range `[-5,5]`

Almost all the math functions implement a "safe" version that allows handling overflows and/or other unexpected behaviors.

## Fitness function
The chosen fitness is based on the negative MSE (Mean Square Error). The negative sign is useful only to keep the fitness higher for better solutions and lower for worse solutions.<br>
This fitness was not extended taking into account, for example, penalties for excessively large trees since the trees have a maximum depth (which is 6) and the starting population generated using only grow method does not produce an impacting bloating effect.

## Algorithm structure
The starting population is composed by 300 individuals generated using the **grow method**, that is, the trees are generated with a variable depth with a certain probability (in this case there is a 40% probability that the tree is expanded for each depth level). The algorithm's main iteration, which represents a single generation, is repeated for 20.000 generations.<br>
The individuals population is managed using a **steady-state** approach. In particular, in each generation:
- we select the first parent using tournament selection, where the starting tournament size is `TOURNAMENT_SIZE = 2`
- we select the second parent using tournament selection with the same tournament size of the first parent
- considering a crossover probability that initially is `Pc = 0.95`, we create two children as follows:
  - with probability `Pc` we make a tree-swapping crossover between the two parents, producing two childen
  - with probability `1 - Pc` we mutate both the first and the second parent, where, in both cases, there is `Pm = 0.5` probability that a simple swap mutation is performed, otherwise a collapse mutation is performed
- with probability `Pi = 0.9` we sort the population in ascending order of fitness and we substitute the two lowest fitness individuals with the newly generated offsprings, otherwise, with probability `1 - Pi`, we insert the two offsprings in a random position in the population (excluding the last index, where there is the highest fitness individual)
- we extract the tree corresponding to the best generational fitness and we save it

## Stagnation avoidance and parameter optimization
The main iteration keeps track of the convergence speed of the current solution by using a `stagnation` variable, that counts the number of generations in which the best solution has been the same. In order to avoid excessively long periods of stagnation, in each generation, we introduce the following approach:
- if the stagnation level is greater or equal to 500 and less than 1000, we reduce the crossover probability `Pc` by 0.05 and we increase the tournament size by 4 every 50 generations, in order to explore new solutions and to modify the population genetic material
- if the stagnation level is greater or equal than 1000, we set `Pc` and `TOURNAMENT_SIZE` to the original values but we substitute the low half of the popuation (the one with lower fitness) with a set of newly generated individuals (generated half with full method and half with grow method)<br>

All these parameters, including the ones expressing probabilities, have been optimized with a trial and error strategy, by changing them in different ways and comparing the obtained results in terms of final fitness and convergence speed.

## Crossover and mutation
The algorithm uses one type of crossover (with a specific variation) and two types of mutation. In particular:
- the **crossover** acts on two parents and is based on subtrees swapping, where two children are produced, each one of which corresponds to one parent with the subtree swapped from the other parent;<br>
if the two parents to be used for the crossover are both composed by just one (terminal) node, then we use a terminal crossover that connects the two parent nodes with a new parent function that requires two children (we do this because, in this case, the default crossover would not change the two parents)
- the **mutation** can either be a simple swap mutation, where we randomly turn a function node into another function node or a terminal node into another therminal node, or it can be a collapse mutation, where we select a random subtree and we collapse it, substituting its root node with a random terminal node

If the tree resulting from a crossover has a depth that is higher than the maximum one, its depth is reduced by a function that collapses the subtrees that are too deep.

## General behavior
The algorithm, being that it implements a steady-state approach where there is only one crossover (or two mutations) at each generation, has a slow convergence, which on the other hand is more precise and gradual. The main iteration can sometimes be stuck in a long stagnation phase (1000 - 4000 stagnation level), especially in the second half of the execution (after 10.000 generations), but after this phases the algorithm usually finds new good solutions, reason for which there is not a maximum value of stagnation after which the iteration stops before reaching the maximum number of generations (20.000 generations).<br>
However, these stagnation phases have been significantly reduced with the methods of stagnation avoidance explained before.<br>
Th execution time can vary according to the dimension of the dataset and is usually around 40 minutes up to a few hours.<br><br><br>

**N.B:** The algorithm doesn't have a strict result consistency, meanining that, by running it multiple times on the same dataset, it can produce final results with a significant difference in the MSE, which, however, is usually (but not always) in the same order of magnitude.
