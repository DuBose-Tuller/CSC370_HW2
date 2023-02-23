import random
import copy

operators = ["+", "-", "*", "/"]


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

        while len(queue) > 0:
            cur_node = queue.pop()

            ops_check = random.random()
            right = ""
            left  = ""
            if cur_node.depth == params["max_depth"] or ops_check > params["two_operators"]:
                # Generate two val children
                right = Node(randval, cur_node.depth + 1)
                left  = Node(randval, cur_node.depth + 1)

            elif ops_check < params["one_operator"]:
                # Generate one op child and one val child, add op child to queue
                #    randomize which order they come in
                val_is_left = random.random() < 0.5
                if val_is_left:
                    right = Node(random.choice(operators), cur_node.depth + 1)
                    left  = Node(randval, cur_node.depth + 1)
                    queue.append(right)
                else:
                    left = Node(random.choice(operators), cur_node.depth + 1)
                    right  = Node(randval, cur_node.depth + 1)
                    queue.append(left)

            else:
                # Generate two op children, add both to queue
                right = Node(random.choice(operators), cur_node.depth + 1)
                left  = Node(random.choice(operators), cur_node.depth + 1)
                queue.append(right)
                queue.append(left)
                
            
            cur_node.set_left(left)
            cur_node.set_right(right)
        


    # def generate_random(self, params):
        

    def mutate(self):
        # Get a list of all the value nodes, and change a random one
        equation = self.copy() # deep copy the tree
        val_nodes = equation.traversal(ret="val")
        # I'm not sure how the references would work here

    def crossover(self):
        pass

    def set_root(self, Node):
        self.root = Node

    def evaluate(self, x, node):
        if node.value not in operators:
            if node.value == "x":
                return x
            else:
                return node.value

        left = self.evaluate(x, node.left)
        right = self.evaluate(x, node.right)
 
        op = node.value
        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op == "*":
            return left * right
        elif op == "/":
            if right == 0: # Safe division
                return 1
            
            return left / right
        else:
            raise Exception()


    # MSE
    def get_fitness(self, xs, ys):
        assert len(x) == len(y)
        total_error = 0

        for (x, y) in zip(xs, ys):
            error = y - self.evaluate(x)
            total_error += error**2

        return total_error/len(x)

    def copy(self):
        return copy.deepcopy(self)

        

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





