from collections import defaultdict
from sage.all import show
from sage.all import Graph

def store_decimal_value_in_binary_array(array, val, reverse=False):
    if reverse:
        start = 0
        stop = len(array)
        step = 1
    else:
        start = len(array) - 1
        stop = -1
        step = -1

    for i in range(start, stop, step):
        if val % 2 == 0:
            array[i] = 0
        else:
            array[i] = 1
        val //= 2

def merge_left_right(left_subarray,right_subarray, array):
    for i in range(len(left_subarray)):
        array[i] = left_subarray[i]
    for j in range(len(right_subarray)):
        array[-1 - j] = right_subarray[-1 - j]

def all_odd_length_binary_arrays_no_mirroring_gen(size):
    """
    size: odd length of array
    """
    halved_size = size // 2
    binary_array = [0]*size
    left_array = [0]*halved_size
    right_array = [0]*halved_size

    for i in range(2**halved_size):
        store_decimal_value_in_binary_array(left_array, i)
        for j in range(i + 1, 2**halved_size):
            store_decimal_value_in_binary_array(right_array, j, reverse=True)
            merge_left_right(left_array, right_array, binary_array)
            binary_array[halved_size] = 0
            yield binary_array
            binary_array[halved_size] = 1
            yield binary_array

def all_even_length_binary_arrays_no_mirroring_gen(size):
    """
    size: even length of array
    """
    halved_size = size // 2
    binary_array = [0]*size
    left_array = [0]*halved_size
    right_array = [0]*halved_size

    for i in range(2**halved_size):
        store_decimal_value_in_binary_array(left_array, i)
        for j in range(i + 1, 2**halved_size):
            store_decimal_value_in_binary_array(right_array, j, reverse=True)
            merge_left_right(left_array, right_array, binary_array)
            yield binary_array

def all_binary_arrays_no_mirroring_gen(size):
    if size % 2 == 0:
        return all_even_length_binary_arrays_no_mirroring_gen(size)
    else:
        return all_odd_length_binary_arrays_no_mirroring_gen(size)

def add_edge(graph_dict, vertex_1, vertex_2):
    graph_dict[vertex_1].append(vertex_2)
    graph_dict[vertex_2].append(vertex_1)

def add_X_edges_in_block_num(graph_dict, block_num, n):
    add_edge(graph_dict, block_num, block_num + 1 + n)
    add_edge(graph_dict, block_num + 1, block_num + n)

def add_simple_chain(graph_dict, start_vertex, end_vertex):
    for i in range(start_vertex, end_vertex):
        add_edge(graph_dict, i, i + 1)
    
def luke_graphs_gen(n):
    for array in all_binary_arrays_no_mirroring_gen(n - 3):
        graph_dict_left = defaultdict(list)
        graph_dict_right = defaultdict(list)

        # Left X
        add_X_edges_in_block_num(graph_dict_left, 0, n)
        add_X_edges_in_block_num(graph_dict_right, 0, n)

        # Add vertical to the left of X
        add_edge(graph_dict_left, 0, n)
        # Add vertical to the right of X
        add_edge(graph_dict_right, 1, n + 1)

        # Right X
        add_X_edges_in_block_num(graph_dict_left, n - 2, n)
        add_X_edges_in_block_num(graph_dict_right, n - 2, n)

        # Add vertical to the left of X
        add_edge(graph_dict_left, n - 2, 2*n - 2)
        # Add vertical to the right of X
        add_edge(graph_dict_right, n - 1, 2*n - 1)

        # Add top chain
        add_simple_chain(graph_dict_left, 0, n - 1)
        add_simple_chain(graph_dict_right, 0, n - 1)

        # Add bottom chain
        add_simple_chain(graph_dict_left, n, 2*n - 1)
        add_simple_chain(graph_dict_right, n, 2*n - 1)

        for i in range(len(array)):
            if array[i] == 1:
                # Add X
                add_X_edges_in_block_num(graph_dict_left, i + 1, n)
                add_X_edges_in_block_num(graph_dict_right, i + 1, n)

                # Add vertical to the left of X
                add_edge(graph_dict_left, i + 1, i + 1 + n)
                # Add vertical to the right of X
                add_edge(graph_dict_right, i + 2, i + 2 + n)

        yield (graph_dict_left, graph_dict_right)

