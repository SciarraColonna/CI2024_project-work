import numpy as np
import random

problem = np.load('../data/problem_3.npz')
x = problem['x']
y = problem['y']


# arbitrary small value used to manage overflows
EPSILON = 1e-10


# safe addition function handling overflow
def safe_add(a, b):
    result = np.add(a, b)
    if result == float("inf"):
        return np.finfo(float).max
    elif result == float("-inf"):
        return -np.finfo(float).max
    return result

# safe subtraction function handling overflow
def safe_subtract(a, b):
    result = np.subtract(a, b)
    if result == float("inf"):
        return np.finfo(float).max
    elif result == float("-inf"):
        return -np.finfo(float).max
    return result

# safe multiplication function handling overflow
def safe_multiply(a, b):
    result = np.multiply(a, b)
    if result == float("inf"):
        return np.finfo(float).max
    elif result == float("-inf"):
        return -np.finfo(float).max
    return result

# safe division function handling the division by zero
def safe_divide(numerator, denominator):
    return np.divide(numerator, denominator + (denominator == 0) * EPSILON)

# safe logarithm function handling the logarithm of negative values
def safe_log(val):
    return np.log(np.maximum(val, EPSILON))

# safe tangent function handling the tangent of tangent of pi/2 + kn
def safe_tan(val):
    cos_val = np.cos(val)
    if np.abs(cos_val) > EPSILON:
        return np.tan(val)
    else:
        np.sign(cos_val) * np.finfo(float).max

# safe exponential function handling overflow
def safe_exp(val):
    result = np.exp(val)
    if result == float("inf"):
        return np.finfo(float).max
    elif result == float("-inf"):
        return -np.finfo(float).max
    return result

# safe square root function handling the case of negative values 
def safe_sqrt(val):
    if val == 0:
        return EPSILON
    return np.sqrt(np.abs(val))

# safe square function handling overflow
def safe_square(val):
    result = np.square(val)
    if result == float("inf"):
        return np.finfo(float).max
    return result


# list of possible functions to insert in the non-terminal tree nodes
function_set = {
    "add": safe_add,
    "sub": safe_subtract,
    "mul": safe_multiply,
    "div": safe_divide,
    "log": safe_log,
    "sin": np.sin,
    "tan": safe_tan,
    "exp": safe_exp,
    "sqrt": safe_sqrt,
    "square": safe_square,
}

# list of possible values for the terminal nodes
terminal_values = [("x" + str(i)) for i in range(1, x.shape[0] + 1)]
# list of possible values for the function nodes
function_values = ["add", "sub", "mul", "div", "log", "sin", "tan", "exp", "sqrt", "square"]



# class representing a single node of the trees (i.e. the individuals of the population)
class Node:
    def __init__(self, value, parent=None, left_child=None, right_child=None):
        self.value = value
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child

# function for duplicating the parent tree when performing a crossover
def duplicate_tree(node, parent=None):
    if node is None:
        return None
    new_node = Node(node.value, node.parent)

    new_node.left_child = duplicate_tree(node.left_child, new_node)
    new_node.right_child = duplicate_tree(node.right_child, new_node)

    return new_node


# function for visiting the tree of a single individual and extracting the related mathematical expression
def visit_tree(node, point):
    if node.left_child == None and node.right_child == None:
        if str(node.value)[0] != "x":
            return node.value
        else:
            return point[int(node.value[1]) - 1]
    
    left_val = None
    right_val = None
    if node.left_child != None:
        left_val = visit_tree(node.left_child, point)
    if node.right_child != None:
        right_val = visit_tree(node.right_child, point)

    if right_val == None:
        return function_set[node.value](left_val)
    return function_set[node.value](left_val, right_val)

# function for evalutaing the MSE of an individual with respect to an input dataset
def evaluate_expression(root, x, y):
    square_err = 0
    for i in range(x.shape[1]):
        y_eval = visit_tree(root, x[:, i].reshape((1, x.shape[0])).ravel())
        square_err += (y_eval - y[i])**2
    
    msq = (1 / x.shape[1]) * square_err
    return msq


# function for counting the number of nodes in the tree
def count_nodes(node):
    if node.left_child is None and node.right_child is None:
        return 1
    
    left_counter = count_nodes(node.left_child)
    right_counter = 0
    if node.right_child is not None:
        right_counter = count_nodes(node.right_child)
    
    return left_counter + right_counter + 1


# function for evaluating the fitness of an individual with respect to an input dataset 
def fitness(root, x, y):
    return - evaluate_expression(root, x, y)


# function for listing the tree nodes according to a pre-order visit
def get_nodes_list(node, nodes_list):
    nodes_list.append(node)
    if node.left_child is not None:
        get_nodes_list(node.left_child, nodes_list)
    if node.right_child is not None:
        get_nodes_list(node.right_child, nodes_list)


# class representing an individual (tree)
class Tree:
    def __init__(self, root):
        self.root = root
        self.fitness = fitness(root, x, y)


MAX_DEPTH = 6
GROWTH_PROBABILITY = 0.4


# function for generating a single individual tree
def fill_tree(depth, node, prob):
    # when the tree has reached the maximum depth we stop the recursion
    if depth == MAX_DEPTH + 1:
        return
    elif random.random() < prob:
        fill_tree(MAX_DEPTH, node, prob)
        return
    # filling the leafs (terminals)
    elif depth == MAX_DEPTH:
        # the current parent node requires only one (left) child
        if node.value not in ["add", "sub", "mul", "div"]:
            if random.random() < 0.9:
                left_index = np.random.randint(0, x.shape[0])
                node.left_child = Node(terminal_values[left_index], node)
            else:
                node.left_child = Node(random.uniform(-5, 5), node)

            fill_tree(depth + 1, node.left_child, prob)
        # the current parent node requires both the children
        else:
            if random.random() < 0.9:
                left_index = np.random.randint(0, x.shape[0])
                node.left_child = Node(terminal_values[left_index], node)
            else:
                node.left_child = Node(random.uniform(-5, 5), node)
            
            if random.random() < 0.9:
                right_index = np.random.randint(0, x.shape[0])
                node.right_child = Node(terminal_values[right_index], node)
            else:
                node.right_child = Node(random.uniform(-5, 5), node)
            
            fill_tree(depth + 1, node.left_child, prob)
            fill_tree(depth + 1, node.right_child, prob)
    # filling the internal nodes (functions)
    else:
        # the current parent node requires only one (left) child
        if node.value not in ["add", "sub", "mul", "div"]:
            left_index = np.random.randint(0, len(function_values))
            node.left_child = Node(function_values[left_index], node)

            if random.random() < prob:
                depth = MAX_DEPTH - 1
            fill_tree(depth + 1, node.left_child, prob)
        # the current parent node requires both the children
        else:
            left_index = np.random.randint(0, len(function_values))
            right_index = np.random.randint(0, len(function_values))

            node.left_child = Node(function_values[left_index], node)
            node.right_child = Node(function_values[right_index], node)
    
            old_depth = depth
            if random.random() < prob:
                depth = MAX_DEPTH - 1
            fill_tree(depth + 1, node.left_child, prob)

            depth = old_depth
            if random.random() < prob:
                depth = MAX_DEPTH - 1
            fill_tree(depth + 1, node.right_child, prob)


# function for generating individuals using the full method
def full_method_init(size, population):
    for _ in range(size):
        depth = 1
        rand_index = np.random.randint(0, len(function_values))
        root = Node(function_values[rand_index])
        fill_tree(depth + 1, root, 0)

        population.append(Tree(root))

# function for generating individuals using the growth method
def grow_method_init(size, population):
    for _ in range(size):
        depth = 1

        if random.random() < GROWTH_PROBABILITY:
            depth = MAX_DEPTH
            if random.random() < 0.9:
                t_index = np.random.randint(0, x.shape[0])
                root = Node(terminal_values[t_index])
                fill_tree(depth + 1, root, GROWTH_PROBABILITY)
            else:
                root = Node(random.uniform(-5, 5))
                fill_tree(depth + 1, root, GROWTH_PROBABILITY)
        else:
            rand_index = np.random.randint(0, len(function_values))
            root = Node(function_values[rand_index])
            fill_tree(depth + 1, root, GROWTH_PROBABILITY)

        population.append(Tree(root))


# function for initializing the individuals population
def init_population(size, population):
    # FULL METHOD initialization
    full_method_init(size // 2, population)
    # GROW MRTHOD initialization
    grow_method_init(size // 2, population)


# probability of performing crossover
Pc = 0.95
# probability of performing one of the two types of mutations
Pm = 0.5
CROSSOVER_MAX_DEPTH = 6


# function used for collapse the subtrees of a tree that has a depth level that is higher than the maximum allowed one
def limit_tree(node, depth):
    if node is None:
        return
    if depth == CROSSOVER_MAX_DEPTH and node.value in function_values:
        if random.random() < 0.9:
            t_index = np.random.randint(0, x.shape[0])
            node.value = terminal_values[t_index]
        else:
            node.value = random.uniform(-5, 5)
        
        node.left_child = None
        node.right_child = None
        return
    
    limit_tree(node.left_child, depth + 1)
    limit_tree(node.right_child, depth + 1)


# function for switching the subtree of a parent with the subtree of another parent
def modify_tree_crossover(node, subtree_1_node, subtree_2_node):
    if node is None:
        return
    if node.left_child is subtree_1_node:
        node.left_child = subtree_2_node
        subtree_2_node.parent = node
        return
    elif node.right_child is subtree_1_node:
        node.right_child = subtree_2_node
        subtree_2_node.parent = node
        return
    
    modify_tree_crossover(node.left_child, subtree_1_node, subtree_2_node)
    modify_tree_crossover(node.right_child, subtree_1_node, subtree_2_node)

# function for performing the crossover between two parents in order to produce two 
# offsprings by swapping two randomly chosen subtrees
def crossover(parent_1, parent_2):
    nodes_1 = []
    nodes_2 = []

    parent_1_root_copy = duplicate_tree(parent_1.root)
    parent_2_root_copy = duplicate_tree(parent_2.root)
    get_nodes_list(parent_1_root_copy, nodes_1)
    get_nodes_list(parent_2_root_copy, nodes_2)

    index_1 = random.randint(0, len(nodes_1) - 1)
    index_2 = random.randint(0, len(nodes_2) - 1)
    while index_1 == 0 and index_2 == 0:
        index_1 = random.randint(0, len(nodes_1) - 1)
        index_2 = random.randint(0, len(nodes_2) - 1)

    subtree_1 = nodes_1[index_1]
    subtree_2 = nodes_2[index_2]
    

    subtree_1_parent = subtree_1.parent
    subtree_2_parent = subtree_2.parent

    if subtree_1_parent is None:
        parent_1_root_copy = subtree_2
    else:
        modify_tree_crossover(parent_1_root_copy, subtree_1, subtree_2)

    if subtree_2_parent is None:
        parent_2_root_copy = subtree_1
    else:
        modify_tree_crossover(parent_2_root_copy, subtree_2, subtree_1)

    limit_tree(parent_1_root_copy, 1)
    limit_tree(parent_2_root_copy, 1)
    return Tree(parent_1_root_copy), Tree(parent_2_root_copy)


# variation of the default crossover used to connect two parents that are composed by a single node
def terminal_crossover(parent_1, parent_2):
    parent_1_root_copy_1 = duplicate_tree(parent_1.root)
    parent_2_root_copy_1 = duplicate_tree(parent_2.root)
    parent_1_root_copy_2 = duplicate_tree(parent_1.root)
    parent_2_root_copy_2 = duplicate_tree(parent_2.root)

    two_children_functions = ["add", "sub", "mul", "div"]
    index_1 = random.randint(0, len(two_children_functions) - 1)
    index_2 = random.randint(0, len(two_children_functions) - 1)
    while index_1 == index_2:
        index_2 = random.randint(0, len(two_children_functions) - 1)

    new_parent_1 = Node(two_children_functions[index_1], None, parent_1_root_copy_1, parent_2_root_copy_1)
    parent_1_root_copy_1.parent = new_parent_1
    parent_2_root_copy_1.parent = new_parent_1

    new_parent_2 = Node(two_children_functions[index_1], None, parent_1_root_copy_2, parent_2_root_copy_2)
    parent_1_root_copy_2.parent = new_parent_2
    parent_2_root_copy_2.parent = new_parent_2

    return Tree(new_parent_1), Tree(new_parent_2)


# function for performing a simple swap mutation
def mutation(parent):
    nodes = []
    parent_root_copy = duplicate_tree(parent.root)
    get_nodes_list(parent_root_copy, nodes)

    node = nodes[random.randint(0, len(nodes) - 1)]
    
    # if the chosen node is a function node, we swap it with another function node, also adjusting the number
    # of children in case they are supposed in a different number
    if node.value in function_values:
        new_value = function_values[random.randint(0, len(function_values) - 1)]
        while new_value == node.value:
            new_value = function_values[random.randint(0, len(function_values) - 1)]
        
        if new_value not in ["add", "sub", "mul", "div"]:
            node.right_child = None
        if new_value in ["add", "sub", "mul", "div"] and node.value not in ["add", "sub", "mul", "div"]:
            if random.random() < 0.9:
                right_index = np.random.randint(0, x.shape[0])
                node.right_child = Node(terminal_values[right_index], node)
            else:
                node.right_child = Node(random.uniform(-5, 5), node)
        node.value = new_value
    # if the chosen node is a terminal node, we swap it with another terminal node
    else:
        new_value = None
        while True:            
            if random.random() < 0.9:
                m_index = np.random.randint(0, x.shape[0])
                new_value = terminal_values[m_index]
            else:
                new_value = random.uniform(-5, 5)
            if new_value != node.value:
                break
        node.value = new_value
        
    return Tree(parent_root_copy)


# function for performing a collapse mutation 
def collapse_mutation(parent):
    nodes = []
    parent_root_copy = duplicate_tree(parent.root)
    get_nodes_list(parent_root_copy, nodes)

    node = nodes[random.randint(0, len(nodes) - 1)]
    if random.random() < 0.9:
        index = np.random.randint(0, x.shape[0])
        new_value = terminal_values[index]
    else:
        new_value = random.uniform(-5, 5)
    node.value = new_value
    node.left_child = None
    node.right_child = None

    return Tree(parent_root_copy)
    

# function for printing the tree structure of a solution
def print_tree(node):
    if node == None:
        return
    else:
        print(str(node.value))
        print_tree(node.left_child)
        print_tree(node.right_child)


# parent selection function 
def tournament_selection(population, tau):
    tournament_individuals = np.random.choice(population, size=tau, replace=False).tolist()
    tournament_individuals.sort(key=lambda tree: - tree.fitness)

    return tournament_individuals[0]


# parameters initialization
POPULATION_SIZE = 300
MAX_GENERATIONS = 20_000
TOURNAMENT_SIZE = 2
Pi = 0.9

best_solution = None
best_fitness = float('inf')
stagnation = 0

# population initialization
population = []
grow_method_init(POPULATION_SIZE, population)
#init_population(POPULATION_SIZE, population)


# main loop
for i in range(0, MAX_GENERATIONS):

    # stagnation avoidance methods
    # variation of the crossover proobability and tournament size
    if stagnation >= 500 and stagnation < 1000:
        if stagnation % 50 == 0:
            Pc = Pc - 0.05
            TOURNAMENT_SIZE += 4

    # injection of new individuals
    if stagnation >= 1000 and stagnation % 500 == 0:
        Pc = 0.95
        TOURNAMENT_SIZE = 2
        new_population = []
        #init_population(POPULATION_SIZE // 2, new_population)
        grow_method_init(POPULATION_SIZE // 2, new_population)
        for j in range(0, POPULATION_SIZE // 2):
            population[j] = new_population[j]
    
    # resetting of crossover probability and tournament size to the default levels
    if stagnation == 0:
        Pc = 0.95
        TOURNAMENT_SIZE = 2

    # parent selection
    parent_1 = tournament_selection(population, TOURNAMENT_SIZE)
    parent_2 = tournament_selection(population, TOURNAMENT_SIZE)
    child_1 = None
    child_2 = None

    # crossover or mutation is performed using the two selected parents
    if random.random() < Pc:
        if parent_1.root.left_child is None and parent_1.root.right_child is None\
            and parent_2.root.left_child is None and parent_2.root.right_child is None:
            child_1, child_2 = terminal_crossover(parent_1, parent_2)
        else:
            child_1, child_2 = crossover(parent_1, parent_2)
    else:
        if random.random() < Pm:
            child_1 = mutation(parent_1)
        else:
            child_1 = collapse_mutation(parent_1)
        if random.random() < Pm:
            child_2 = mutation(parent_2)
        else:
            child_2 = collapse_mutation(parent_2)

    # the offsprings are put in the population set
    if random.random() < Pi:
        population.sort(key=lambda node: node.fitness)
        population[0] = child_1
        population[1] = child_2
    else:
        population[random.randint(0, POPULATION_SIZE - 2)] = child_1
        population[random.randint(0, POPULATION_SIZE - 2)] = child_2

    # selection of the best generational tree and update of the stagnation level
    best_generation_fitness = max(list(map(lambda node: node.fitness, population)))
    if best_generation_fitness == best_fitness:
        stagnation += 1
    else:
        stagnation = 0
    
    best_fitness = best_generation_fitness
    print("Generation " + str(i + 1) + ": best fitness = " + str(-best_fitness))
    print("Stagnation: " + str(stagnation))
    best_tree = None
    for tree in population:
        if tree.fitness == best_fitness:
            best_solution = tree
            break


# print final solution 
print("Best expression:")
print_tree(best_solution.root)