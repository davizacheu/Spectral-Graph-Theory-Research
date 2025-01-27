import sys

from torch import combinations
from sage.all import (
    block_matrix, 
    ones_matrix, 
    identity_matrix, 
    diagonal_matrix, 
    matrix,
    Matrix, 
    Graph, 
    QQ, 
    QQbar,
    copy
)

from xo_graphs import xo_dicts_gen
from xo_graphs import obtain_positions_for_xo_graph
import itertools
from concurrent.futures import ThreadPoolExecutor
import os

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
    return block_matrix(QQ, block_arr)

I = identity_matrix(2)
J = ones_matrix(2)
O = Matrix(2)
A = J * (1/2)
B = J * (-1/2) + I

def check_combination(tup):
    """
    The function that will run in worker processes.
    `tup` should be (combination, target_l_small, target_r_small).
    """

    combination, target_l_small, target_r_small = tup
    
    s_1 = block_matrix(2,2, combination[:4])
    s_3 = block_matrix(2,2, combination[4:])
    s_4 = s_1 + s_3*block_matrix(2,2, [O,O,O,J - I])
    s_2 = (-1)*s_3*block_matrix(2,2, [I,O,O, 2*I])
    sim_candidate = block_matrix(QQ, 2, 2,
                                 [s_1,s_2,
                                  s_3,s_4])
    left_result = target_l_small * sim_candidate
    rigth_result = sim_candidate * target_r_small
    return left_result, rigth_result, sim_candidate

def brute_force_correct_corner_blocks(target_l, target_r):
    """
    Example showing how to use a generator + ProcessPoolExecutor to
    avoid building an enormous list in memory.
    """

    # ---------------------------------------------------------------
    # 1) Reduce target_l, target_r to 4×4 block-matrices as you do now
    # ---------------------------------------------------------------
    def extract_block(matrix, row_start, col_start):
        return matrix.submatrix(row_start, col_start, 2, 2)
    
    positions = [(i, j) for i in [0, 8, 10, 18] for j in [0, 8, 10, 18]]
    blocks_l = [extract_block(target_l, row, col) for row, col in positions]
    blocks_r = [extract_block(target_r, row, col) for row, col in positions]
    target_l_small = block_matrix(QQ, 4, 4, blocks_l)
    target_r_small = block_matrix(QQ, 4, 4, blocks_r)
    print(f'target_l_small \n')
    print(target_l_small)
    print(f'target_r_small \n')
    print(target_r_small)

    # target_l_second_quadrant = target_l.submatrix(0, 0, 10, 10)
    # target_r_second_quadrant = target_r.submatrix(0, 0, 10, 10)

    # print('target_l_second_quadrant')
    # print(target_l_second_quadrant)
    # print('target_r_second_quadrant')
    # print(target_r_second_quadrant)
 
    # similar, P = target_l_small.is_similar(target_r_small, transformation= True)
    # print( 'Are they similar? ', similar)
    # if similar:
    #     P = P.change_ring(QQ)
    #     P_inv = P.inverse()
    #     print('P')
    #     print(P)
    #     print('P inverse')
    #     print(P_inv)

    # sim_candidate = block_matrix(4,4, [
    #     [A,O,O,O],
    #     [O,A,O,O],
    #     [O,O,B,O],
    #     [O,O,O,B]

    # ])
    # print('sim_candidate')
    # print(sim_candidate)
    # left_result = (target_l_small * sim_candidate).change_ring(QQ)
    # rigth_result = (sim_candidate * target_r_small).change_ring(QQ)
    # print('left_result')
    # print(left_result)
    
    # print('right_result')
    # print(rigth_result)

    # if left_result == rigth_result:
    #     print("Sim candidate found!")
        
        

    # ---------------------------------------------------------------
    # 2) Define the 5 possible 2×2 blocks
    # ---------------------------------------------------------------
    
    possible_matrices = [A, -A, B, -B, O]

    # ---------------------------------------------------------------
    # 3) Create a generator of all possible 16-block combinations
    #    *without* converting to a huge list
    # ---------------------------------------------------------------
    flexible_blocks = itertools.product(possible_matrices, repeat=8)

    # ---------------------------------------------------------------
    # 4) Submit these tasks in parallel using ProcessPoolExecutor
    # ---------------------------------------------------------------
    # Using executor.map(...) consumes the generator on-the-fly
    # and returns an iterator of results in the same order.
    with ThreadPoolExecutor() as executor:
        print(f'PID: {os.getpid()}') 
        # Map each combination to (combination, target_l_small, target_r_small)
        futures = []
        # so check_combination() knows what to do.
        step = 1
        for combination in flexible_blocks:
            if step % 10000 == 0: print(f'combination: {step}')
            f = executor.submit(check_combination, (combination, target_l_small, target_r_small))
            futures.append(f)
            step += 1
        for f in futures:
            left_result, rigth_result, sim_candidate = f.result()
            if left_result == rigth_result:
                print("Sim candidate found!")
                print(sim_candidate)


def find_ihara_similarity(n):
    """ Returns Ihara Matrix of G """

    """ Returns the degree matrix D of graph G"""
    def deg_matrix(G):
        return diagonal_matrix([G.degree(v) for v in G.vertices(sort=True)], sparse=False)
    
    def ihara_matrix(G):
        A = G.adjacency_matrix()
        D = deg_matrix(G)
        I = identity_matrix(G.order())
        Z = matrix.zero(G.order())
        return block_matrix(QQbar, [[A, D-I], [-I, Z]], subdivide=False)
    
    graph_pairs_and_sim_matrix = xo_graphs_and_similarity_matrix_gen(
        n, laplacian_similarity_matrix_for_xo_graphs_alternating_labels
        )
    
    for graph_left_and_pos, graph_right_and_pos, similarity_matrix in graph_pairs_and_sim_matrix:
        graph_left, pos_left = graph_left_and_pos
        graph_right, pos_right = graph_right_and_pos

        target_l = ihara_matrix(graph_left)
        target_r = ihara_matrix(graph_right)
        print('target_l = ')
        print(target_l)
        print('target_r= ')
        print(target_r)
        print('similarity_matrix')
        print(similarity_matrix)
    
        brute_force_correct_corner_blocks(target_l, target_r)

with open('xo_ihara_similarity_corners.txt', 'w') as f:
    sys.stdout = f  # Redirect standard output to the file
    find_ihara_similarity(5)
