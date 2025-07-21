from collections import defaultdict
from collections import deque
from binary_arrays_for_xo_graphs import all_binary_arrays_no_mirroring_gen
from xo_graph_helper_functions import *
    
def xo_dicts_gen(n_rows, n_cols, twin_connectivity='cycle'):
    h_step = n_rows
    v_step = 1

    for array in all_binary_arrays_no_mirroring_gen(n_cols - 3):

        # Append 1's on the extremes of the array
        array = deque(array)
        array.appendleft(1)
        array.append(1)
        array = list(array)

        # Initialize dictionaries 
        graph_dict_left = defaultdict(list)
        graph_dict_right = defaultdict(list)

        # Add chain for each row
        for r in range(n_rows):
            add_simple_chain(graph_dict_left, r*v_step, h_step*(n_cols - 1), h_step)
            add_simple_chain(graph_dict_right, r*v_step, h_step*(n_cols - 1), h_step)

        for i in range(len(array)):
            if array[i] == 1:
                vertical_twin_group_first_node = i * h_step

                # Add X
                add_complete_bipartite_edges_in_block(graph_dict_left, vertical_twin_group_first_node, h_step)
                add_complete_bipartite_edges_in_block(graph_dict_right, vertical_twin_group_first_node, h_step)

                # Add edges among twin vertex groups
                if twin_connectivity == 'cycle':
                    for j in range(n_rows):
                        add_edge(graph_dict_left, vertical_twin_group_first_node + j, vertical_twin_group_first_node + (j + v_step) % n_rows)
                        add_edge(graph_dict_right, vertical_twin_group_first_node + h_step + j, vertical_twin_group_first_node + h_step + (j + v_step) % n_rows)
       
        yield (graph_dict_left, graph_dict_right, array)

def obtain_positions_for_xo_graph(n_rows, n_cols):
    positions = {}
    for c in range(0, n_cols):
        for r in range(0, n_rows):
            positions[c*n_rows + r] = (c, n_rows - r)
    return positions
