def add_edge(graph_dict, vertex_1, vertex_2):
    graph_dict[vertex_1].append(vertex_2)
    graph_dict[vertex_2].append(vertex_1)

def add_X_edges_in_block_num(graph_dict, first_vertex, gap):
    add_edge(graph_dict, first_vertex, first_vertex + 1 + gap)
    add_edge(graph_dict, first_vertex + 1, first_vertex + gap)

def add_simple_chain(graph_dict, start_vertex, end_vertex, step):
    for i in range(start_vertex, end_vertex, step):
        add_edge(graph_dict, i, i + step)