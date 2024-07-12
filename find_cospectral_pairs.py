from collections import defaultdict
from time import time
from sage.graphs.graph import Graph
from sage.graphs.graph_generators import graphs
from concurrent.futures import ProcessPoolExecutor
from sys import argv

def return_tuple_graph_and_function(G):
    return (G.ihara_zeta_function_inverse(), G)

def submit_task_to_process_pool_exec(graph_generator):
    result_dict= defaultdict(Graph)
    with ProcessPoolExecutor() as executor:
        futures = []
        for G in graph_generator:
            future = executor.submit(return_tuple_graph_and_function, G)
            futures.append(future)
        for future in futures:
            f, g = future.result()
            if f in result_dict:
                print('Cospectral pair:')
                print(g.to_dictionary())
                print(result_dict[f].to_dictionary())
            else:
                result_dict[f] = g
    print('Length of list:', len(result_dict))

def main(nauty_geng_flags):
    # Generate graphs
    graph_gen = graphs.nauty_geng(nauty_geng_flags)
    t = time()
    result = submit_task_to_process_pool_exec(graph_gen)
    dt = time() - t
    print('Time taken:', dt)

if __name__ == "__main__":
    if len(argv) > 1:
        main(argv[1])
    else:
        print('Invalid usage of the module')
