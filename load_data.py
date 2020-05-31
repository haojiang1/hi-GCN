import networkx as nx
import numpy as np
import scipy as sc
import os
import re
from scipy.io import loadmat


def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:

            # count = 0
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
                # count = count + 1
                # print(count)
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    print(filename_node_attrs)
    try:

        with open(filename_node_attrs) as f:

            for line in f:

                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val == 0:
                label_has_zero = True
            graph_labels.append(val - 1)
    graph_labels = np.array(graph_labels)
    if label_has_zero:
        graph_labels += 1
    filename_graphs2 = prefix + '_graph_labels2.txt'
    graph_labels2 = []
    with open(filename_graphs2) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val == 0:
                label_has_zero = True
            graph_labels2.append(val - 1)
    graph_labels2 = np.array(graph_labels2)
    if label_has_zero:
        graph_labels2 += 1

    filename_adj = prefix + '_A.txt'
    filename_edg = prefix + '_edge_attributes.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        # count = 0
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))

            adj_list[graph_indic[e0]].append((e0, e1))

            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1

    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]
    # input the brian network of ASD
    data_dict = loadmat('\\opt_BrainNet.mat')  # ASD866

    data_array = data_dict['opt_BrainNet']
    add = 0
    print('shape: ', data_array.shape)

    print(data_array[::, ::, 0].shape)

    graph_array = data_array[::, ::, 0]
    graphs = []
    for i in range(1, 1 + len(adj_list)):
        print('i', i)
        G = nx.from_edgelist(adj_list[i])
        for graph_number in range(data_array.shape[2]):
          if (i == (graph_number + 1)):
            print('g_n', graph_number)
            # if ((graph_number != 53) & (graph_number != 101)):
            single_graph = data_array[::, ::, graph_number]

            single_graph = np.nan_to_num(single_graph)
            for row in range(len(single_graph)):
                for j in range(len(single_graph)):
                    if (row == j):
                        single_graph[row][j] = 0


            value = 0
            obj = 0

            for lines in range(graph_array.shape[0]):
                for columns in range(graph_array.shape[0]):
                    if (abs(single_graph[lines - obj][columns]) <= 0.75):
                        value = value + 1
                if (value == 116):
                    single_graph = np.delete(single_graph, lines - obj, axis=0)

                    value = 0
                    obj = obj + 1
                else:
                    value = 0
                    # print(lines+1)

            value = 0
            item = 0
            items = 0
            for columns in range(single_graph.shape[1]):
                for lines in range(single_graph.shape[0]):
                    if (abs(single_graph[lines][columns - item]) <= 0.75):
                        value = value + 1
                if (value == single_graph.shape[0]):
                    single_graph = np.delete(single_graph, columns - items, axis=1)
                    value = 0
                    item = item + 1
                    items = items + 1

                else:
                    value = 0

            add = add + single_graph.shape[1]
            for lines in range(single_graph.shape[0]):
                for columns in range(single_graph.shape[1]):


                    single_graph[lines][columns] = abs(single_graph[lines][columns])
                    if (abs(single_graph[lines][columns]) <= 0.75):
                        single_graph[lines][columns] = 0


            where_are_nan = np.isnan(single_graph)
            where_are_inf = np.isinf(single_graph)
            single_graph[where_are_nan] = 0
            single_graph[where_are_inf] = 0.8
            # single_graph = np.nan_to_num(single_graph)
            # if graph_number == 53:
            np.set_printoptions(threshold=np.inf)
            #    print(single_graph)

            G = nx.from_numpy_matrix(single_graph)

        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        G.graph['label2'] = graph_labels2[i - 1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs

