from collections import defaultdict
from time import time
from sage.graphs.graph import Graph
from sage.graphs.graph_generators import graphs
from concurrent.futures import ProcessPoolExecutor

def return_tuple_graph_and_function(G):
    return (G.ihara_zeta_function_inverse(), G)

def submit_task_to_process_pool_exec(graph_generator):
    result_dict= defaultdict(Graph)
    with ProcessPoolExecutor() as executor:
        futures = []
        for G in graph_generator:
            future = executor.submit(return_tuple_graph_and_function, G)
            futures.append(future)
        for f in futures:
            tuple = f.result()
            if tuple[0] in result_dict:
                print('Cospectral pair:')
                print(tuple[1].to_dictionary()) 
                print(result_dict[tuple[0]].to_dictionary())
            result_dict[tuple[0]] = tuple[1]
    print('Length of list:', len(result_dict))

def main():
    # Generate graphs
    graph_gen = graphs.nauty_geng("10 -c -d2")
    t = time()
    result = submit_task_to_process_pool_exec(graph_gen)
    dt = time() - t
    print('Time taken:', dt)

if __name__ == "__main__":
    main()
