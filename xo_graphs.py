from collections import defaultdict
from collections import deque
from binary_arrays_for_xo_graphs import all_binary_arrays_no_mirroring_gen
from xo_graph_helper_functions import *
    
def xo_dicts_gen(n, isAlternating=True):
    if isAlternating:
        x_gap = 2
        v_gap = 1
        step = 2
    else:
        x_gap = n
        v_gap = n
        step = 1

    for array in all_binary_arrays_no_mirroring_gen(n - 3):

        # Append 1's on the extremes of the array
        array = deque(array)
        array.appendleft(1)
        array.append(1)
        array = list(array)

        # Initialize dictionaries 
        graph_dict_left = defaultdict(list)
        graph_dict_right = defaultdict(list)

        # Add top chain
        add_simple_chain(graph_dict_left, 0, step*(n - 1), step)
        add_simple_chain(graph_dict_right, 0, step*(n - 1), step)

        # Add bottom chain
        add_simple_chain(graph_dict_left, v_gap, 2*n - 1, step)
        add_simple_chain(graph_dict_right, v_gap, 2*n - 1, step)

        for i in range(len(array)):
            if array[i] == 1:
                # Add X
                add_X_edges_in_block_num(graph_dict_left, i*step, x_gap)
                add_X_edges_in_block_num(graph_dict_right, i*step, x_gap)

                # Add vertical to the left of X
                add_edge(graph_dict_left, i*step, i*step + v_gap)
                # Add vertical to the right of X
                add_edge(graph_dict_right, (i + 1)*step, (i + 1)*step + v_gap)

        yield (graph_dict_left, graph_dict_right, array)

def obtain_positions_for_xo_graph(luke_graph_dict, isAlternating=False):
    if isAlternating:
        step = 2
    else:
        step = 1

    positions = {}
    for t in range(0, step*(len(luke_graph_dict) // 2), step):
        positions[t] = (t, 1)
    x = 0
    for b in range(len(luke_graph_dict) - 1 - t, len(luke_graph_dict), step):
        positions[b] = (x, 0)
        x += step
    return positions
