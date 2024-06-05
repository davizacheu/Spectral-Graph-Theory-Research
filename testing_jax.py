import numpy as np
import jax.numpy as jnp
from time import time
import networkx as nx
from sage.graphs.graph import Graph
from sage.graphs.graph_generators import graphs
import concurrent.futures
from multiprocessing import Pool, cpu_count, Manager, Queue, Pipe
from multiprocessing.pool import ThreadPool
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict 
from sys import getsizeof

def sample_code():
    #### Generate 1 million random matrices in chunks of 100k ####
    k = 30
    chunk_size = int(1e5)
    num_chunks = 90
    t = time()
    AA = []
    for ii in range(num_chunks):
        A_nonsym = np.random.randint(0, 2, (chunk_size, k, k))  # random 0-1 matrices
        A = np.tril(A_nonsym) + np.tril(A_nonsym, -1).swapaxes(1, 2)  # symmetrize
        AA.append(A)
    dt_generate_matrices = time() - t
    print('dt_generate_matrices=', dt_generate_matrices)

    #### Compute eigenvalues and check if small ####
    bad_matrices_all_chunks = []
    for ii, A in enumerate(AA):
        # move chunk of 100k matrices from memory to GPU
        t = time()
        A_gpu = jnp.array(A)
        dt_move = time() - t

        # compute eigenvalues
        t = time()
        eigs = jnp.linalg.eigvalsh(A_gpu)
        dt_eig = time() - t

        # Check for small eigenvalues
        t = time()
        bad_matrices = jnp.any(np.abs(eigs) < 1.0, axis=1)
        bad_matrices_all_chunks.append(bad_matrices)
        dt_check = time() - t

        print('chunk: ', ii, ', dt_move=', dt_move, ', dt_eig=', dt_eig)

def submit_task_to_process_pool_exec(graph_generator):
    result_dict= defaultdict(Graph)
    with concurrent.futures.ProcessPoolExecutor() as executor:
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
    

def parallel_threads_processor(process_element, generator):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the generator to the executor 
        future_to_element = {executor.submit(process_element, element): element for element in generator}
        for future in concurrent.futures.as_completed(future_to_element):
            element = future_to_element[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'Element {element} generated an exception: {exc}')
            else:
                results.append(result)
    return results

def return_tuple_graph_and_function(G):
    return (G.ihara_zeta_function_inverse(), G)

def pool_process(graph_gen):
    pool = Pool()

    list = pool.map(return_tuple_graph_and_function, graph_gen)
    print('Length of list:',len(list))

def thread_pool(graph_gen):
    pool = ThreadPool()
    map = pool.map(return_tuple_graph_and_function, graph_gen, chunksize=10000)
    pool.close()
    pool.join()
    return map


def map_zeta_inverse_function_to_graphs(G: Graph):
    key = G.ihara_zeta_function_inverse()
    if key not in shared_dict:
        shared_dict[key] = list()
    else:
        shared_list_of_keys.append(key)
    shared_dict[key].append(G)

def display_cospectral_graphs():
    # Create a PDF with the plots and text
    with PdfPages('ihara_cospectral_graphs_on_10v.pdf') as pdf:
        for key in shared_list_of_keys:
            # Create a figure for the text
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "THE FOLLOWING GRAPHS ARE IHARA COSPECTRAL",
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.axis('off')  # Turn off the axis
            # Save the figure with text to the PDF
            pdf.savefig(fig)
            plt.close(fig)  # Close the figure to free up memory

            for G in shared_dict[key]:
                # Create a plot of the graph
                graph_plot = G.plot(vertex_size=300, vertex_labels=True, edge_color='blue')
                # Convert the Sage plot to a Matplotlib figure
                fig = graph_plot.matplotlib()

                # Save the figure to the PDF
                pdf.savefig(fig)
                plt.close(fig)

def populate_queue_with_graph_and_function(graph : Graph, queue : Queue, conn):
    data = queue.get()
    if data[graph.ihara_zeta_function_inverse()] is not None :
        conn.send([graph, data[graph.ihara_zeta_function_inverse()]])
    data[graph.ihara_zeta_function_inverse()] = graph
    queue.put(data)

def populate_dict_with_graph_and_function(graph : Graph, shared_data, conn):
    if graph.ihara_zeta_function_inverse() in shared_data :
        conn.send([graph, shared_data[graph.ihara_zeta_function_inverse()]])
    shared_data[graph.ihara_zeta_function_inverse()] = graph
    

def async_apply(graph_generator):
    pool = Pool()
    manager = Manager()
    shared_dict = manager.dict()
    parent_conn, child_conn = Pipe()
    for G in graph_generator:
        pool.apply_async(func=populate_dict_with_graph_and_function, args=(G, shared_dict, child_conn))
    pool.close()
    pool.join()
    while parent_conn.poll():
        input = parent_conn.recv()
        print(input)

def main():
    print("Number of cpus from cpu_count in multiprocessing module:", os.cpu_count())
    print("Number of cpus from cpu_count in os module:", cpu_count())
    print("Number of logical cpus the calling thread is restriced to:", len(os.sched_getaffinity(0)))

    # Generate graphs
    graph_gen = graphs.nauty_geng("10 -c -d2")
    t = time()
    result = list(graph_gen)
    print('Memory space taken:', getsizeof(result))
    dt = time() - t
    print('Time taken:', dt)


# Create a global shared dictionary
shared_dict = dict()

# Create a global shared list of keys with multiple values
shared_list_of_keys = list()

if __name__ == "__main__":
    main()
     