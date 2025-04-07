import numpy as np
import random
import math

problem = np.load('../data/problem_7.npz')
x = problem['x']
y = problem['y']


# arbitrary small value used to manage overflows
EPSILON = 1e-10


# safe addition function handling overflow
def safe_add(a, b):
    result = a + b
    if result == float("inf"):
        return np.finfo(float).max
    elif result == float("-inf"):
        return -np.finfo(float).max
    return result

# safe subtraction function handling overflow
def safe_subtract(a, b):
    result = a - b
    if result == float("inf"):
        return np.finfo(float).max
    elif result == float("-inf"):
        return -np.finfo(float).max
    return result

# safe multiplication function handling overflow
def safe_multiply(a, b):
    result = a * b
    if result == float("inf"):
        return np.finfo(float).max
    elif result == float("-inf"):
        return -np.finfo(float).max
    return result

# safe division function handling the division by zero
def safe_divide(numerator, denominator):
    if denominator == 0:
        return numerator / EPSILON
    else:
        return numerator / denominator

# safe logarithm function handling the logarithm of negative values
def safe_log(val):
    if val <= 0:
        val = EPSILON 
    return math.log(val)

# safe exponential function handling overflow
def safe_exp(val):
    try:
        result = math.exp(val)
    except OverflowError:
        result = float('inf')  
    if result == float('inf'):
        return np.finfo(float).max 
    return result

# safe square root function handling the case of negative values 
def safe_sqrt(val):
    if val < 0:
        val = - val
    if val == 0:
        return EPSILON
    return math.sqrt(val)

# safe square function handling overflow
def safe_square(val):
    result = val * val
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
    "sin": math.sin,
    "cos": math.cos,
    "exp": safe_exp,
    "sqrt": safe_sqrt,
    "square": safe_square,
}

# list of possible values for the terminal nodes
terminal_values = [("x" + str(i)) for i in range(1, x.shape[0] + 1)]
# list of possible values for the function nodes
function_values = ["add", "sub", "mul", "div", "log", "sin", "cos", "exp", "sqrt", "square"]



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
    if node.left_child is None and node.right_child is None:
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


# function for counting the number of common subtrees between two individuals
def common_subtrees(node_1, node_2):
    if node_1 is None:
        return 0
    
    left_subtrees = common_subtrees(node_1.left_child, node_2)
    right_subtrees = common_subtrees(node_1.right_child, node_2)
    
    c_sub = 0
    nodes_1 = []
    nodes_2 = []
    get_nodes_list(node_1, nodes_1)
    get_nodes_list(node_2, nodes_2)

    flag = True
    if len(nodes_1) <= len(nodes_2):
        for i in range(len(nodes_2) - len(nodes_1) + 1):
            for j in range(len(nodes_1)):
                if nodes_2[i + j].value != nodes_1[j].value:
                    flag = False
                    break
            if flag == True:
                c_sub = 1
                break
            else:
                flag = True

    return left_subtrees + right_subtrees + c_sub


# function for evaluating the fitness of an individual with respect to an input dataset 
def fitness(root, x, y):
    return 1 / evaluate_expression(root, x, y)


# function for listing the tree nodes according to a pre-order visit
def get_nodes_list(node, nodes_list):
    nodes_list.append(node)
    if node.left_child is not None:
        get_nodes_list(node.left_child, nodes_list)
    if node.right_child is not None:
        get_nodes_list(node.right_child, nodes_list)


# function for verifying that two individuals have the same nodes and structure
def compareTrees(root_1, root_2):
    nodes_1 = []
    nodes_2 = []
    get_nodes_list(root_1, nodes_1)
    get_nodes_list(root_2, nodes_2)

    flag = True
    if len(nodes_1) != len(nodes_2):
        return False
    else:
        for i in range(0, len(nodes_1)):
            if nodes_1[i].value != nodes_2[i].value:
                flag = False
                break
        return flag


# class representing an individual (tree)
class Tree:
    def __init__(self, root):
        self.root = root
        self.fitness = fitness(root, x, y)



MAX_DEPTH = 6
GROWTH_PROBABILITY = 0.5

# function for generating a single individual tree
def fill_tree(depth, node, prob):
    # when the tree has reached the maximum depth we stop the recursion
    if depth == MAX_DEPTH + 1:
        return
    elif depth == 2:
        fill_tree(MAX_DEPTH, node, prob)
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
    counter = 0
    while True:
        if counter == size:
            break
        depth = 1

        rand_index = np.random.randint(0, len(function_values))
        root = Node(function_values[rand_index])
        fill_tree(depth + 1, root, GROWTH_PROBABILITY)

        duplicated = False
        for i in range(counter):
            if compareTrees(root, population[i].root):
                duplicated = True
                break
        
        if not duplicated:
            population.append(Tree(root))
            counter += 1


# function for initializing the individuals population
def init_population(size, population):
    # FULL METHOD initialization
    full_method_init(size // 2, population)
    # GROW MRTHOD initialization
    grow_method_init(size // 2, population)



# probability of performing crossover
Pc = 0.90
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


# upper bound of the constants valid only for the mutation
MAX_CONST = 2
# function for counting the constants inside an individual's tree
def count_const(nodes):
    counter = 0
    for i in range(0, len(nodes)):
        if nodes[i].value not in function_values and str(nodes[i].value)[0] != "x":
            counter += 1
    return counter 


# function for performing a node mutation
def mutation(parent):
    nodes = []
    parent_root_copy = duplicate_tree(parent.root)
    get_nodes_list(parent_root_copy, nodes)

    node = nodes[random.randint(0, len(nodes) - 1)]

    tc_functions = ["add", "sub", "mul", "div"]
    # if the node requires two children we change it with another node that requires two children
    if node.value in tc_functions:
        new_value = tc_functions[random.randint(0, len(tc_functions) - 1)]
        while new_value == node.value:
            new_value = tc_functions[random.randint(0, len(tc_functions) - 1)]
        node.value = new_value
    # if the node requires only one child we change it with another node that requires only one child
    elif node.value in function_values and node.value not in tc_functions:
        new_value = function_values[random.randint(4, len(function_values) - 1)]
        while new_value == node.value:
            new_value = function_values[random.randint(4, len(function_values) - 1)]
        node.value = new_value
    # if the node is a constant and we change it adding random gaussian noise or turning it into a variable
    elif str(node.value)[0] != "x":
        new_value = None
        if random.random() < 0.8:
            index = np.random.randint(0, x.shape[0])
            new_value = terminal_values[index]
        else:
            new_value = node.value + np.random.normal(loc=0.0, scale=1.0)
        node.value = new_value
    # if the node is a variable we change it into a constant or to another variable
    else:
        new_value = None
        if random.random() < 0.5 and count_const(nodes) < MAX_CONST:
            new_value = random.uniform(-5, 5)
        else:
            m_index = np.random.randint(0, x.shape[0])
            new_value = terminal_values[m_index]
            while len(terminal_values) > 1 and new_value == node.value:
                m_index = np.random.randint(0, x.shape[0])
                new_value = terminal_values[m_index]
        node.value = new_value

    return Tree(parent_root_copy)


# function for performing a collapse mutation 
def collapse_mutation(parent):
    if parent.root.left_child is None and parent.root.right_child is None:
        return parent
    
    nodes = []
    parent_root_copy = duplicate_tree(parent.root)
    get_nodes_list(parent_root_copy, nodes)

    node = nodes[random.randint(0, len(nodes) - 1)]
    while node.left_child is None and node.right_child is None:
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


# function for performing an hoist mutation
def hoist_mutation(parent):
    if parent.root.left_child is None and parent.root.right_child is None:
        return parent
    
    nodes = []
    parent_root_copy = duplicate_tree(parent.root)
    get_nodes_list(parent_root_copy, nodes)

    node = nodes[random.randint(0, len(nodes) - 1)]
    while node.left_child is None and node.right_child is None:
        node = nodes[random.randint(0, len(nodes) - 1)]

    return Tree(node)


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
    # selecting the first parent
    tournament_individuals = np.random.choice(population, size=tau, replace=False).tolist()
    tournament_individuals.sort(key=lambda tree: - tree.fitness)
    parent_1 = tournament_individuals[0]

    # selecting the second parent
    tournament_individuals = np.random.choice(population, size=(len(population) // 10), replace=False).tolist()
    tournament_individuals.sort(key=lambda tree: (common_subtrees(tree.root, parent_1.root) / count_nodes(tree.root)))
    parent_2 = tournament_individuals[0]

    return parent_1, parent_2



# parameters initialization
# constant parameters
POPULATION_SIZE = 600
MAX_GENERATIONS = 500
ELITISM_SIZE = int(0.02 * POPULATION_SIZE)

# probability (non-constant) paremeters
TOURNAMENT_SIZE = 20
Pc = 0.90
Pm_1 = 0.8
Pm_2 = 0.1
Pm_3 = 0.1

best_solution = None
best_fitness = float('-inf')
stagnation = 0


# population initialization
population = []
init_population(POPULATION_SIZE, population)

# main loop
for i in range(0, MAX_GENERATIONS):    
    offsprings = []

    # regular injection of random individuals in the offsprings list
    if i > 0 and i % 10 == 0:
        population.sort(key=lambda node: node.fitness)
        grow_method_init(0.05 * POPULATION_SIZE, offsprings)

    # the top individuals of the previous generation are copied in the following generation
    population.sort(key=lambda node: node.fitness)
    offsprings.extend(population[-ELITISM_SIZE:])

    # loop for filling the remaining part of the offsprings list
    while len(offsprings) < POPULATION_SIZE:
        # parent selection
        parent_1, parent_2 = tournament_selection(population, TOURNAMENT_SIZE)
        child_1 = None
        child_2 = None

        if random.random() < Pc:
            # crossover
            child_1, child_2 = crossover(parent_1, parent_2)
        else:
            # mutation
            functions = [mutation, collapse_mutation, hoist_mutation]
            probabilities = [Pm_1, Pm_2, Pm_3]
            child_1 = np.random.choice(functions, p=probabilities)(parent_1)
            child_2 = np.random.choice(functions, p=probabilities)(parent_2)

        # if the resulting offsprings have depth 1 (i.e. they have just one node) they are discarded
        if not (child_1.root.left_child is None and child_1.root.right_child is None):
            offsprings.append(child_1)
        if not (child_2.root.left_child is None and child_2.root.right_child is None):
            offsprings.append(child_2)

    # the offprings list becomes the actual population
    population = offsprings

    # selection of the best individual of the generation and update of the stagnation level
    best_generation_fitness = max(list(map(lambda node: node.fitness, population)))

    if i == 0:
        best_fitness = best_generation_fitness
    else:
        if best_generation_fitness > best_fitness:
            best_fitness = best_generation_fitness
            stagnation = 0
            # when the stagnation ends the updated parameters are set to their original values
            TOURNAMENT_SIZE = 20
            Pc = 0.90
            Pm_1 = 0.8
            Pm_2 = 0.1
            Pm_3 = 0.1
        else:
            stagnation += 1
            # crossover probability and tournament size are updated in case of stagnation
            if Pc > 0.60:
                Pc -= 0.05
            if stagnation >= 10 and TOURNAMENT_SIZE > 15:
                TOURNAMENT_SIZE -= 1

    # if the stagnation continues for more than 2 generations we also update the probabilities of the mutations
    if stagnation > 2:
        Pm_1 = 0.4
        Pm_2 = 0.3
        Pm_3 = 0.3
    
    
    print("Generation " + str(i + 1) + ": best fitness = " + str(best_fitness) + ", best MSE = " + str(1 / best_fitness))
    print("Stagnation: " + str(stagnation))
    # updating the best solution (if necessary)
    if stagnation == 0:
        for tree in population:
            if tree.fitness == best_fitness:
                best_solution = tree
                break


# print final solution 
print("Best expression:")
print_tree(best_solution.root)