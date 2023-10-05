import heapq
import numpy as np


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for pos in range(len(from_state)):
        num = from_state[pos]   #get the current puzzle number

        if num == 0:            #no puzzle need to be moved here
            continue
        for to_pos in range(len(to_state)): #find the to_position of the puzzle
            if num == to_state[to_pos]:
                distance += abs(pos//3-to_pos//3) + abs(pos%3-to_pos%3)
                break
        
    return distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))

def swap_puzzles(state, pos, swap_pos):
    res = np.array(state)
    res[pos], res[swap_pos] = res[swap_pos], res[pos]

    return list(res).copy()     # return a copy of res or it'll always return the same   

def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
 
    succ_states = []
    for pos in range(len(state)):
        if state[pos] != 0:
            continue
        x = pos//3
        y = pos%3
        # not right
        swap_pos = x*3 + y+1
        if y != 2 and state[swap_pos] != 0:
            succ_states.append(swap_puzzles(state, pos, swap_pos))
        # not left
        swap_pos = x*3 + y-1
        if y != 0 and state[swap_pos] != 0:
            succ_states.append(swap_puzzles(state, pos, swap_pos))
        # not bottom  
        swap_pos = (x+1)*3 + y
        if x != 2 and state[swap_pos] != 0:
            succ_states.append(swap_puzzles(state, pos, swap_pos))
        # not top
        swap_pos = (x-1)*3 + y
        if x != 0 and state[swap_pos] != 0:
            succ_states.append(swap_puzzles(state, pos, swap_pos))

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    open = []       # priority queue
    close = []      # store the node that already visited
    close_len = 0
    max_queue_len = -1
    h = get_manhattan_distance(state, goal_state)
    heapq.heappush(open, (h, state, (0, h, -1)))    # initialize
    
    while len(open) != 0:
        # open the lowest priority node and add to close
        node = heapq.heappop(open)
        # print(node)
        node_state = node[1].copy()
        close.append(node)
        if node_state == goal_state:
            close_len += 1
            break

        # expand the node
        succ_states = get_succ(node_state)
        parent = close_len
        close_len += 1
        
        for succ in succ_states:
            #culculate g(n) and h(n)
            g = node[2][0] + 1
            h = get_manhattan_distance(succ, goal_state)

            # check if is closed
            closed = False
            for item in close:
                if succ == item[1]:
                    closed = True
                    break
            if closed == False:
                heapq.heappush(open, (g+h, succ, (g, h, parent)))
            # else:
            #     if g < node[2][0]:
            #         heapq.heappush(open, (g+h, succ, (g, h, parent)))
        if len(open) > max_queue_len:
            max_queue_len = len(open)

    if len(open) == 0:
        return     #failure to find solution

    # trace the path with the close list and the parent index (position in close list)
    path = []
    parent = close_len-1
    while parent != -1:
        node = close[parent]
        path.insert(0, node)
        parent = node[2][2]
    
    # print the solution of the puzzle
    move = 0
    for item in path:
        print(item[1], "h={} moves: {}".format(item[2][1], move))
        move += 1
    print("Max queue length: %d" % max_queue_len)

                    

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([3, 4, 6, 0, 0, 1, 7, 2, 5])
    print()
