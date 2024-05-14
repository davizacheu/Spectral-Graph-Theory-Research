import numpy as np
import jax.numpy as jnp
from time import time
import networkx as nx
from sage.graphs.graph import Graph
from sage.graphs.graph_generators import graphs

def deg_matrix(G):
    return np.diag([G.degree(v) for v in G.nodes()])

def ihara_matrix(G):
    A = nx.adjacency_matrix(G).todense(); D=deg_matrix(G); I = np.identity(len(G.nodes())); Z=np.zeros([len(G.nodes()),len(G.nodes())])
    return np.block([[A,D-I],[-I,Z]]) 

def sample_code():
    #### Generate 1 million random matrices in chunks of 100k ####
    k = 30
    chunk_size = int(1e5)
    num_chunks = 10
    t = time()
    AA = []
    for ii in range(num_chunks):
        A_nonsym = np.random.randint(0, 2, (chunk_size, k, k)) # random 0-1 matrices
        A = np.tril(A_nonsym) + np.tril(A_nonsym, -1).swapaxes(1,2) # symmetrize
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

def matrix_list(gen):
    list = []  # Use the correct variable name
    for G in gen:  # Iterate over the generator directly
        if type(G) is Graph:
            list.append(ihara_matrix(G))
    return list

gen = graphs.nauty_geng("9 -c -d2")
results = matrix_list(gen)
print(len(results))
