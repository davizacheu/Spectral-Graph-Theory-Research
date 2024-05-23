import numpy as np
import jax.numpy as jnp
import time
import networkx as nx
from sage.graphs.graph import Graph
from sage.graphs.graph_generators import graphs
import concurrent.futures
from multiprocessing import Pool, cpu_count, Manager
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def deg_matrix(G):
    return np.diag([G.degree(v) for v in G.nodes()])

def ihara_matrix(G):
    A = nx.adjacency_matrix(G).todense()
    D = deg_matrix(G)
    I = np.identity(len(G.nodes()))
    Z = np.zeros([len(G.nodes()), len(G.nodes())])
    return np.block([[A, D - I], [-I, Z]])


def sample_code():
    #### Generate 1 million random matrices in chunks of 100k ####
    k = 30
    chunk_size = int(1e5)
    num_chunks = 10
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

def ihara_matrix_gen(graph_gen):
    for G in networkx_graph_gen(graph_gen):
        yield ihara_matrix(G)

def networkx_graph_gen(gen):
    for G in gen:
        yield G.networkx_graph()

def collect_ihara_matrices(matrix_gen):
    list = []
    for M in matrix_gen:
        list.append(M)
    return list

def hash_eigenvalues(M):
    return np.linalg.eigvals(M)

def parallel_processes_generator(processing_function, generator):
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        map = executor.map(processing_function, generator)
    return map

def process_graph_to_ihara_matrix(G):
    return ihara_matrix(G.networkx_graph())

def populate_dict_ihara_zeta_to_graphs(graph: Graph, shared_dict):
    zeta_function_inverse = graph.ihara_zeta_function_inverse()
    shared_dict[zeta_function_inverse]

def multiprocess_generator(processing_function, generator, chunksize):
    with Pool() as pool:
        list = pool.starmap(processing_function, generator, chunksize)
    return list

def map_zeta_inverse_function_to_graphs(G: Graph):
    key = G.ihara_zeta_function_inverse()
    if key not in shared_dict:
        shared_dict[key] = manager.list()
    else:
        shared_list_of_keys.append(key)
    shared_dict[key].append(G)

def main():
    print("Number of cpus from cpu_count in multiprocessing module:", os.cpu_count())
    print("Number of cpus from cpu_count in os module:", cpu_count())
    print("Number of logical cpus the calling thread is restriced to:", len(os.sched_getaffinity(0)))

    # Generate graphs
    graph_gen = graphs.nauty_geng("10 -c -d2")

    pool = Pool()

    pool.map(map_zeta_inverse_function_to_graphs, graph_gen)

    # Close pool.
    pool.close()

    # Wait for all thread.
    pool.join()

    print("Displaying graphs in pdf file")

    # Display graphs
    display_cospectral_graphs()

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

# Manager to create shared object.
manager = Manager()

# Create a global shared dictionary
shared_dict = manager.dict()

# Create a global shared list of keys with multiple values
shared_list_of_keys = manager.list()

if __name__ == "__main__":
    main()
     