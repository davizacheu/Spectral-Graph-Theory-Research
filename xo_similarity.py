from sage.matrix.special import block_matrix, ones_matrix
from sage.graphs.graph import Graph
from sage.all import Matrix
from sage.all import QQ
from xo_graphs import xo_dicts_gen
from xo_graphs import obtain_positions_for_xo_graph


def xo_graphs_and_similarity_matrix_gen(n, similarity_matrix_constructor, isAlternating=True):
    xo_dict_pairs = xo_dicts_gen(n, isAlternating)
    for graph_dict_left, graph_dict_right, xo_array in xo_dict_pairs:
        if isAlternating:
            yield (
                (Graph(graph_dict_left),
                 obtain_positions_for_xo_graph(graph_dict_left, isAlternating)
                 ),
                (Graph(graph_dict_right),
                 obtain_positions_for_xo_graph(graph_dict_right, isAlternating)
                 ),
                similarity_matrix_constructor(xo_array)
            )
        else:
            yield (graph_dict_left, graph_dict_right, None)


def laplacian_similarity_matrix_for_xo_graphs_alternating_labels(xo_array):
    n = len(xo_array) + 1
    # Initialize similarity 2d array
    O = Matrix(2)
    I = Matrix.identity(2)
    J = ones_matrix(2)
    A = J*(1/2)
    B = J*(-1/2) + I
    C = [Matrix(QQ, [[-1, 1], [1, -1]]), Matrix(QQ, [[1, -1], [-1, 1]])]
    block_arr = [[O for i in range(n)] for j in range(n)]
    block_arr[0][0] = A
    block_arr[-1][-1] = A
    block_arr[0][-1] = B
    block_arr[-1][0] = B
    o_count = 0
    c_index = 0
    for i in range(1, n - 1):
        block_arr[i][i] = I
        if xo_array[i] == 1:
            o_count = 0
        else:
            for j in range(o_count + 1):
                block_arr[i + 1][i - j] = C[c_index]*(3**j)
                c_index = (c_index + 1) % 2
            c_index = 0
            o_count += 1
    return block_matrix(block_arr)
