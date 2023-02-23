import random

operators = ["+", "-", "*", "/"]


params = {
    "one_operator": 0.2,
    "two_operators": 0.5,
    "val_is_x": 0.25,
    "min_val": -5,
    "max_val": 5,
    "max_depth": 3,
    "is_real": False
}


class Equation:
    def __init__(self, params):
        # sanity check
        assert params["two_operators"] >= params["one_operator"]
        
        if params["is_real"]:
            randval = random.uniform(params["min_val"], params["max_val"])
        else:
            randval = random.randint(params["min_val"], params["max_val"])
        
        
        # Set root
        queue = []
        root = Node(random.choice(operators), 0)
        self.set_root(root)
        queue.append(root)
        cur_node = queue.pop()

        while len(queue) > 0:
            ops_check = random.random()
            right = ""
            left  = ""
            if ops_check < params["one_operator"]:
                # Generate one op child and one val child, add op child to queue
                right = Node(random.choice(operators))
                left  = Node(randval)
                queue.append(right)
            else if ops_check < params["two_operators"]:
                # Generate two op children, add both to queue
                right = Node(random.choice(operators))
                left  = Node(random.choice(operators))
                queue.add(right)
                queue.add(left)
            else:
                # Generate two val children
                right = Node(randval)
                left  = Node(randval)
                
            
            cur_node.set_left(left)
            cur_node.set_right(right)

            cur_node = queue.pop()
                   


    def mutate(self):
        # Get a list of all the value nodes, and change a random one
        equation = self.copy() # deep copy the tree
        val_nodes = equation.traversal(ret="val")
        # I'm not sure how the references would work here

    def crossover(self):
        pass

    def set_root(self, Node):
        self.root = Node

    # Post order traveral of the tree. Can be 
    #    set to evaluate the equation at a certain
    #    x value, or returns all of the 'value' nodes
    # IDK if this is the right way to do this
    def traversal(self, ret=None, value=None):
        pass

    def get_fitness(self, x, y):
        assert len(x) = len(y)
        total_error = 0
        

class Node:
    def __init__(self, value, depth):
        self.value = value
        self.parent = None
        self.left = None
        self.right = None
        self.depth = depth
    
    def set_parent(self, parent):
        self.parent = parent
    
    def set_left(self, left):
        self.left = left
    
    def set_right(self, right):
        self.right = right

    def get_value(self):
        return self.value


def initialize(size, is_real, max_depth, threshold):
    return [Equation(max_depth, is_real, threshold) for i in range(size)]


