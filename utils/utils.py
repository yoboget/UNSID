import os
import networkx as nx
import torch
import numpy as np
from torch_geometric.utils import to_dense_batch, to_dense_adj, to_networkx


def get_networkx_from_dense(X, A, mask, x_min=None):
    graphs = []
    if X is not None:
        X.squeeze_()
        assert X.dim() == 2, 'The annotation matrix must be of shape (batch_size, n)'
        assert mask.size() == X.size(), 'The annotation matrix and the mask must be of same shape'
    assert A.dim() == 3, 'The adjacency matrix must be of shape (batch_size, n, n)'

    batch_size = A.size(0)
    for b in range(batch_size):
        G = nx.Graph()

        # Add nodes with annotations
        for i, node in enumerate(X[b, mask[b]]):
            if x_min is not None:
                node += x_min
            G.add_node(i, label=node.int().item())

        # Add edges with attributes
        for i in range(mask[b].sum()):
            for j in range(i + 1, mask[b].sum()):  # Only consider the upper triangle for undirected graphs
                if A[b, i, j] > 0:  # Check if there is an edge
                    G.add_edge(i, j, label=A[b, i, j].int().item())
                    G.add_edge(j, i, label=A[b, i, j].int().item())  # Add reverse for undirected

        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        graphs.append(G)
    return graphs

def batch_to_networkx(batch):
    data_list = batch.to_data_list()
    nx_list = []
    for j, data in enumerate(data_list):

        if data.edge_attr is not None:
            is_edge = data.edge_attr[..., :-1].sum(-1) > 0
            assert data.edge_attr.dim() == 2, 'Edge_attr should be 2-dimensional'
            data.edge_index = data.edge_index[:, is_edge]
        nx_graph = to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True,remove_self_loops=True)

        if data.x is not None and data.x.size(-1) != 0:
            node_attrs = {i: {'label': x.int().item()} for i, x in enumerate(data.x[..., 0])}
            nx.set_node_attributes(nx_graph, node_attrs)
        if data.edge_attr is not None and data.edge_attr.size(-1) > 2:
            edge_attr = data.edge_attr[is_edge].argmax(-1) + 1
            edge_attrs = {(x.item(), y.item()): {'label': z.item()} for (x, y), z in zip(data.edge_index.T, edge_attr)}
        else:
            edge_attrs = {(x.item(), y.item()): {'label': 1} for x, y in data.edge_index.T}
        nx.set_edge_attributes(nx_graph, edge_attrs)

        nx_list.append(nx_graph)
    return nx_list

def batch_to_dense(batch, max_num_nodes):
    n_max = max(batch.batch.bincount())
    X, mask = to_dense_batch(batch.x, batch=batch.batch, max_num_nodes=n_max)
    A = to_dense_adj(batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch,
                     max_num_nodes=n_max)
    A = torch.cat((A.sum(-1, keepdim=True) == 0, A), dim=-1)
    return X, A, mask

def get_num_nodes_distribution(loader, max_num_nodes, dataset):
    filepath = f'./data/{dataset}/node_distribution.pt'
    if os.path.exists(filepath):
        num_nodes_distribution = torch.load(filepath)
    else:
        num_nodes = 0
        for batch in loader['train']:
            n = batch.batch.bincount()
            num_nodes += n.bincount(minlength=max_num_nodes+1)
        num_nodes_distribution = num_nodes / num_nodes.sum()
        torch.save(num_nodes_distribution, filepath)
    return num_nodes_distribution

def get_conditional_input(cond, X_noisy, mask, max_num_nodes, device, training=True, conditional=False):
    if training:
        is_cond = torch.bernoulli(0.25 * torch.ones(X_noisy.shape[0], 1)).to(device)
    else:
        if conditional:
            is_cond = torch.ones(X_noisy.shape[0], 1).to(device)
        else:
            is_cond = torch.zeros(X_noisy.shape[0], 1).to(device)
    cond = cond * is_cond
    is_cond = is_cond.repeat(1, max_num_nodes).unsqueeze(-1)
    cond = cond.repeat(max_num_nodes, 1).reshape(*X_noisy.shape[:2], -1) * mask.unsqueeze(-1)
    return torch.cat((X_noisy, cond, is_cond), dim=-1)

def print_eval(runs, dataset):
    keys = runs[0].keys()
    latex_format = {}
    for key in keys:
        val = []
        for run in runs:
            val.append(run[key])
        mean = np.asarray(val).mean()
        std = np.asarray(val).std()
        print(f'mean {key}: {mean}')
        print(f'std {key}: {std}')
        if key in ['valid', 'unique', 'novel']:
            mean *= 100
            std *= 100
            latex_format[key] = f'${mean:.2f} \pm {std:.2f}$'
        elif key in ['nspdk', 'degree', 'cluster', 'orbit', 'spectral']:
            mean *= 1000
            std *= 1000
            latex_format[key] = f'${mean:.3f} \pm {std:.3f}$'
        else:
            latex_format[key] = f'${mean:.3f} \pm {std:.3f}$'

    if dataset in ['qm9', 'qm9_cc', 'qm9H', 'zinc']:
        print(
            f'{latex_format["valid"]} &  {latex_format["fcd"]} & {latex_format["nspdk"]} & {latex_format["unique"]} & {latex_format["novel"]} \\')
    else:
        if 'orbit' in latex_format.keys():
            latex_format["valid"], latex_format['novel'] = 0, 0
            print(
                f'{latex_format["valid"]} &  {latex_format["degree"]} & {latex_format["cluster"]} & {latex_format["orbit"]}& {latex_format["spectral"]} & {latex_format["novel"]} \\')
        else:
            print(
                f'{latex_format["valid"]} &  {latex_format["degree"]} & {latex_format["cluster"]} & - & {latex_format["spectral"]} & {latex_format["novel"]} \\')