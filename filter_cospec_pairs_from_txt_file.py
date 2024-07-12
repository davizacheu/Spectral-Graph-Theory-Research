from ast import literal_eval
from sage.graphs.graph import Graph
from sage.all import var
from sage.matrix.constructor import identity_matrix, diagonal_matrix
import sys
from os import dup2, write
from concurrent.futures import ProcessPoolExecutor

def create_graph_from_dict_string(dict_string):
    return Graph(literal_eval(dict_string))

def generalized_charac_poly_from_graph(G : Graph):
    u, t = var('u t')
    return (u * identity_matrix(G.num_verts()) - (G.adjacency_matrix() - t * diagonal_matrix(G.degree()))).determinant()

def read_cospectral_pairs(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    i = 1
    while i <= len(lines) - 3:
        graph1_str = lines[i].strip()
        graph2_str = lines[i + 1].strip()
        i += 3
        yield (graph1_str, graph2_str)

def read_cospectral_groups(file_path):
    divider = "Cospectral group:"

    with open(file_path, 'r') as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == divider:
            i += 1
            ntuple = []
            while lines[i].strip() != divider :
                ntuple.append(lines[i].strip())
                i += 1
            n = len(ntuple)
            for i in range(n):
                for j in range(i + 1, n):
                    yield (ntuple[i], ntuple[j])
        else:
            i += 1
            
def check_not_equal_general_poly(pair_str: tuple):
    if generalized_charac_poly_from_graph(create_graph_from_dict_string(pair_str[0])) \
            != generalized_charac_poly_from_graph(create_graph_from_dict_string(pair_str[1])):
        return pair_str
    else:
        return False

def check_not_equal_ihara_zeta_inverse_function(pair_str):
    g1_str, g2_str = pair_str
    if create_graph_from_dict_string(g1_str).ihara_zeta_function_inverse()\
          != create_graph_from_dict_string(g2_str).ihara_zeta_function_inverse():
        return pair_str
    return False

def filter_cospec_tuples_with_desired_property(pairs_str, property_verification_function):
    filtered_tuples = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(property_verification_function, pair_str) for pair_str in pairs_str]
        for f in futures:
            tuple = f.result()
            if tuple:
                filtered_tuples.append(tuple)
    return filtered_tuples

def write_pairs_of_graphs_to_file(pairs_of_graphs, file_path):
    # Saving the reference of the standard output
    original_stdout = sys.stdout 

    with open(file_path, 'w') as file:
        sys.stdout = file

        for g1_str, g2_str in pairs_of_graphs:
            print('Pair:')
            print(g1_str)
            print(g2_str)

        # Reset the standard output
        sys.stdout = original_stdout

if __name__ == "__main__":
    file_path = '10v_cospec_pairs.txt'
    cospectral_pairs_str = read_cospectral_pairs(file_path)
    filtered_cospec_pairs = filter_cospec_tuples_with_desired_property(cospectral_pairs_str, check_not_equal_general_poly)
    for g1_str, g2_str in filtered_cospec_pairs:
        print('Pair:')
        print(g1_str)
        print(g2_str)
    print('All done')
