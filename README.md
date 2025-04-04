# Project work (Symbolic regression with GP)
**The code related to this project was written on my own with a sporadic support from ChatGPT for handling the math functions unexpected behaviors and overflows**<br><br>
The genetic programming algorithm used to solve the symbolic regression problem has the following main characteristics:
- **Population size**: 600
- **Maximum generations**: 500
- **Crossover**: tree swap crossover
- **Mutation**: node mutation, collapse mutation, hoist mutation
- **Population model**: generational model with elitism
- **Parent selection**: tournament selection with diversity enforcement
- **Survivor selection**: deterministic and structure-based
<br>


## Individuals structure
Each individual is characterized by a tree structure where each node is an instance of the class `Node`, that specifies the value of the node, its parent node and the left and right children (if present).
The whole tree is embedded by an instance of the class `Tree`, that contains the root node and the fitness value of the tree.<br>
Each node can either be:
- a function node, expressing one of the math functions in the set `["add", "sub", "mul", "div", "log", "sin", "cos", "exp", "sqrt", "square"]`
- a terminal (leaf) node, expressing a variable `x1, x2,...` or a random float constant with a value in the range `[-5,5]`

A tree has a minimum depth of 2 and a maximum depth of 6.<br>
All the math functions (excluding sine and cosine) implement a "safe" version that allows handling overflows and/or other unexpected behaviors.<br><br>


## Fitness function
The chosen fitness function consists of the reciprocal of the mean square error, meaning that the fitness is larger when the MSE is smaller.<br><br>


## Algorithm initialization
The algorithm starts initializing the population with 600 individuals, half generated with grow method and the other half with full method, in order to have a starting population as structually diverse as possible.
The grow method consists of expanding the tree with a probability of 0.5. Therefore, for each node, we have a 50% probability of inserting a terminal node as a child and 50% probability of inserting a function node as a child (this is repeated for both left and right child if both required).<br><br>


## Crossover and mutation
The implemented crossover is a simple tree-sawp crossover and the related crossover rate `Pc` is initially equal to 0.9.<br>
The available mutation operators are the following:
- **Node mutation**: the mutation is applied to a single random node of the individual. In particular:
  - If the selected node is a function node that requires two children than it is replaced by another randon function node that requires two children.
  - If the selected node is a function node that requires only one child it is replaced with another random child that requires only one child.
  - If the selected node is a terminal variable node it is replaced with a terminal constant node or another terminal variable node.
  - If the selected node is a terminal constant node it is replaced with a terminal variable node or it is perturbated with a random gaussian noise.
- **Collapse mutation**: a random node of the individual is selected and its subtree gets collapsed and replaced with a terminal node (either a variable or a constant).
- **Hoist mutation**: a random node of the individual is selected and its subtree gets extracted and becomes a new individual. The rest if the original tree is discarded.<br><br>


## Parent selection
The parent selection is performed using a modified version of the tournament selection.<br>
Given a value `TOURNAMENT_SIZE`:
- The first parent is selected with the classical approach, that is, selecting `TOURNAMENT_SIZE` random individuals from the population and choosing the one with the highest fitness value.
- The second parent is selected by taking randomly one tenth of the individuals of the population and choosing the one that is most structurally diverse from the first parent. The similarity between the selected individuals and the first parent is calculated, for each induvidual, as the ratio between the number of common subtrees between the individual and the first parent and the number of nodes of the individual.

Basing the calculation on common subtrees allows to perform crossovers among trees with different structure (useful for exploration and diversity) and dividing it by the number of nodes of the individual reduces the probability of performing crossovers between trees that have a huge difference in depth (in fact, it is easier to have few common subtrees with respect to a tree that has the minimum depth).<br><br>


## Algorithm structure
The algorithm has the following initial values of the parameters:
- `ELIITISM_SIZE`: 2% of the population
- `TOURNAMENT_SIZE`: 20
- `Pc` (probability of performing a crossover): 90%
-  `Pm_1` (probability of performing a node mutation): 80%
-  `Pm_2` (probability of performing a collapse mutation): 10%
-  `Pm_3` (probability of performing an hoist mutation): 10%
  
After the population gets initialized, the algorithm procedes as follows:
- If the current generation is not the first one and it is a multiple of 10, we inject in the offsprings a random set of tree (generated with grow method) whose size is equal to 5% of the population. This is made to maintain a stable diversity among generations.
- We insert into the offsprings the top `ELITISM_SIZE` of the individuals of the previous generation
- While the offsprings have not the same size as the population:
  - We perform a crossover with probability `Pc`
  - We perform a mutation with probability `1 - Pc` (it's mutually exclusive with crossover). In this case 

