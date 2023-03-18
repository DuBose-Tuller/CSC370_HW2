import random
import copy
import numpy as np

operators = ["+", "-", "*", "/"]


class Equation:
    def __init__(self, params):
        self.nodes = []

        # sanity check
        assert params["two_operators"] >= params["one_operator"]
        
        if params["is_real"]:
            randval = random.uniform
        else:
            randval = random.randint
        
        
        # Set root
        queue = []
        root = Node(random.choice(operators), params["start_depth"])
        self.set_root(root)
        queue.append(root)

        while len(queue) > 0:
            cur_node = queue.pop()

            ops_check = random.random()
            if cur_node.depth == params["max_depth"]  or ops_check > params["two_operators"]:
                # Generate two val children
                right = self.generate_val_node(params, cur_node.depth + 1, randval)
                self.nodes.append(right)
                right.set_parent(cur_node)
                left  = self.generate_val_node(params, cur_node.depth + 1, randval)
                self.nodes.append(left)
                left.set_parent(cur_node)

            elif ops_check < params["one_operator"]:
                # Generate one op child and one val child, add op child to queue
                #    randomize which order they come in
                val_is_left = random.random() < 0.5
                if val_is_left:
                    right = Node(random.choice(operators), cur_node.depth + 1)
                    self.nodes.append(right)
                    right.set_parent(cur_node)
                    queue.append(right)

                    left  = self.generate_val_node(params, cur_node.depth + 1, randval)
                    self.nodes.append(left)
                    left.set_parent(cur_node)
                    
                else:
                    left = Node(random.choice(operators), cur_node.depth + 1)
                    self.nodes.append(left)
                    left.set_parent(cur_node)
                    queue.append(left)

                    right  = self.generate_val_node(params, cur_node.depth + 1, randval)
                    self.nodes.append(right)
                    right.set_parent(cur_node)

            else:
                # Generate two op children, add both to queue
                right = Node(random.choice(operators), cur_node.depth + 1)
                self.nodes.append(right)
                right.set_parent(cur_node)
                queue.append(right)

                left  = Node(random.choice(operators), cur_node.depth + 1)
                self.nodes.append(left)
                left.set_parent(cur_node)
                queue.append(left)
                
            
            cur_node.set_left(left)
            cur_node.set_right(right)
        

    def generate_val_node(self, params, depth, randval):
        x_check = random.random()

        if x_check < params["val_is_x"]:
            var = random.choice(params["variables"])
            node = Node(var, depth)
            # node = Node("x", depth)
        else: 
            node = Node(randval(params["min_val"], params["max_val"]), depth)

        return node

    def mutate(self, params, prob_regrow=0.1):
        regrow_check = random.random()
        if regrow_check < prob_regrow:
            return self.mutate_regrow(params)
        else:
            return self.mutate_change_val(params)

    def mutate_regrow(self, params):
        # Get a list of all the value nodes, and change a random one
        equation = self.copy() # deep copy the tree
        mutation_point = random.choice(equation.nodes)

        start_depth = mutation_point.depth
        params["start_depth"] = start_depth

        new_subtree = Equation(params).root
        parent = mutation_point.parent
        new_subtree.parent = parent

        if parent.left == mutation_point:
            parent.left = new_subtree
        else:
            parent.right = new_subtree

        return equation
    
    def mutate_change_val(self, params):
        # Get a list of all the value nodes, and change a random one
        equation = self.copy() # deep copy the tree
        mutation_point = random.choice(equation.nodes)
        if mutation_point.value in operators:
            mutation_point.value = random.choice(operators)
        else:
            if params["is_real"]:
                randval = random.uniform
            else:
                randval = random.randint
            
            mutation_point.value = randval(params["min_val"], params["max_val"])
        
        return equation


    def crossover(self, other, brood=1):
        assert(type(other) == Equation)

        children = []

        for i in range(brood):
            child1 = self.copy()
            child2 = other.copy()

            child1_cross = random.choice(child1.nodes)
            child2_cross = random.choice(child2.nodes)

            # Swap the trees at the specified points
            parent1 = child1_cross.parent
            parent2 = child2_cross.parent
            child2_cross.parent = parent1
            child1_cross.parent = parent2

            if parent1.left == child1_cross:
                parent1.left = child2_cross
            else:
                parent1.right = child2_cross

            if parent2.left == child2_cross:
                parent2.left = child1_cross
            else:
                parent2.right = child1_cross

            children.append(child1)
            children.append(child2)

        return children


    def set_root(self, Node):
        self.root = Node

    def randNode(self):
        return random.choice(self.nodes)

    def evaluate(self, inputs, variables, node):
        if node.value not in operators:
            for (i, var) in enumerate(variables):
                if node.value == var:
                    return inputs[i]
            else:
                return node.value

        left = self.evaluate(inputs, variables, node.left)
        right = self.evaluate(inputs, variables, node.right)
 
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


    def set_MSE(self, inputs, outputs, variables = ["x"], set=True):
        assert len(inputs) == len(outputs) and len(inputs[0]) == len(variables)
        total_error = 0

        for (ins, out) in zip(inputs, outputs):
            error = out - self.evaluate(ins, variables, self.root)
            total_error += error**2

        if set:
            self.MSE = total_error/len(inputs)
        else:
            return total_error/len(inputs)

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

    def __repr__(self):
        ret = "\t"*self.depth+repr(self.value)+"\n"
        if not self.left:
            return ret
        
        ret += self.left.__repr__()
        ret += self.right.__repr__()
        return ret



