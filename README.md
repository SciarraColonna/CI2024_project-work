# Project work (Symbolic regression with GP)
The genetic programming algorithm used to solve the symbolic regression problem has the following main characteristics:

## Individuals structure
Each individual is characterized by a tree structure where each node is an instance of the class `Node`, that specifies the value of the node, its parent node and the left and right children (if present).
The whole tree is embedded by an instance of the class `Tree`, that contains the root node and the fitness value of the tree.
