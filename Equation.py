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
        # Set root
        queue = []
        root = Node(random.choice(operators), 0)
        self.set_root(root)
        queue.append(root)
        cur_node = queue.pop()

        while len(queue) > 0:
            # if random.random() > op_thresh or cur_node.depth > max_depth:
            #     # Generate values
            #     if is_real:
            #         if random.random() < x_thresh:
            #             self.generate_children(cur_node, random.randrange(MIN_VAL, MAX_VAL), random.randrange(MIN_VAL, MAX_VAL))

            #     else:
            #         if random.random() < x_thresh:
            #             left = Node(random.randint(MIN_VAL, MAX_VAL), cur_node.depth + 1)
            #             right = Node(random.randint(MIN_VAL, MAX_VAL), cur_node.depth + 1)
            #             cur_node.set_left(left)
            #             cur_node.set_right(right)
                
            # else: 
            #     # Generate operands
            #     self.generate_children(cur_node, random.choice(operators), random.choice(operators))
            #     queue.append(left)
            #     queue.append(right)


            if random.random() < params[""]:


            cur_node = queue.pop()
            

    def generate_children(cur_node, left_val, right_val):
        left = Node(left_val, cur_node.depth + 1)
        right = Node(right_val, cur_node.depth + 1)
        cur_node.set_left(left)
        cur_node.set_right(right)
        


    def mutate(self):
        pass

    def crossover(self):
        pass

    def set_root(self, Node):
        self.root = Node

    def evaluate(self, value):
        pass

    def get_fitness(self):
        pass

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


