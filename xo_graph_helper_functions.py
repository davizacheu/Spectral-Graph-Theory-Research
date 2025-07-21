def add_edge(graph_dict, vertex_1, vertex_2):
    graph_dict[vertex_1].append(vertex_2)
    graph_dict[vertex_2].append(vertex_1)

def add_complete_bipartite_edges_in_block(graph_dict, first_vertex, gap):
    for i in range(gap):
        for j in range(gap):
            if i != j:
                add_edge(graph_dict, first_vertex + i, first_vertex + gap + j)

def add_simple_chain(graph_dict, start_vertex, end_vertex, step):
    for i in range(start_vertex, end_vertex, step):
        add_edge(graph_dict, i, i + step)

