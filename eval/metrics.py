import torch
from matplotlib.pyplot import legend

from eval.mol_metrics import gen_mol, mol_metric, get_molecular_properties
from eval.stats3 import eval_graph_list, eval_fraction_unique_non_isomorphic_valid, is_planar_graph, is_sbm_graph
from utils.utils import get_networkx_from_dense, batch_to_dense
from eval.molset import get_all_metrics
from utils.plot import plot_batch_networkx_graphs
import pickle
import wandb

from eval.mol_metrics import load_smiles

from rdkit import Chem
from rdkit.Chem import Draw


class SamplingMetrics:
    def __init__(self, dataset, max_num_nodes, sampling, ref_loader=None):
        self.dataset = dataset
        self.max_num_nodes = max_num_nodes
        self.sampling = sampling
        print(dataset)
        if dataset in ['planar', 'sbm', 'enzymes']:
            self.val_loader = ref_loader
            batch = next(iter(ref_loader))
            X, A, mask = batch_to_dense(batch, max_num_nodes=max_num_nodes)
            self.ref_graphs = get_networkx_from_dense(X.argmax(-1), A.argmax(-1), mask)
            # plot_batch_networkx_graphs(self.ref_graphs[:20], filename='./misc/plots/sbm/sbm_test')
            # pickle.dump(self.ref_graphs, open(f'./misc/dump/sbm_test.pkl', 'wb'))
        if dataset == 'qm9_cc' or dataset == 'qm9_dg':
            self.mean = torch.tensor([123.0129, 0.1307, 0.4617])
            self.std = torch.tensor([7.6222, 1.1716, 0.0769])



    def __call__(self, X, A, mask, mask_adj, step=None, fname=None,
                 ref_values=None, conditioning=None):
        if ref_values is not None:
            self.ref_values=ref_values

        distrib_metric = self.eval_general_stats(X, A, mask, mask_adj)

        if self.dataset in ['qm9', 'zinc', 'qm9_cc', 'qm9_dg', 'qm9H']:
            metrics = self.molecular_metrics((X, A, mask), conditioning=conditioning)

        elif self.dataset in ['planar', 'sbm', 'enzymes']:
            gen_graphs = get_networkx_from_dense(X, A, mask)
            metrics = self.generic_graph_metrics(gen_graphs, self.dataset)
        else:
            NotImplementedError('Metrics not implemented for this dataset.')
        if not self.sampling:
            metrics['epoch'] = step
        wandb.log({f'sampling/': metrics}, step=step)
        wandb.log({f'distributions/': distrib_metric}, step=step)
        print(f'sampling: ', metrics)
        print(f'distributions: ', distrib_metric)
        return metrics

    def molecular_metrics(self, gen_graphs, conditioning=None):
        annots, adjs, mask = gen_graphs
        gen_mols, num_no_correct = gen_mol(annots, adjs, mask, self.dataset)
        # for i, mol in enumerate(gen_mols):
        #     Draw.MolToFile(mol, f'./misc/plots/zinc/zinc_gen_{i}.png')

        if self.dataset == 'qm9_cc':
            dataset = 'qm9'
        else:
            dataset = self.dataset

        if conditioning is not None:
            values = get_molecular_properties(gen_mols)
            values = values.to(conditioning.device)
            values = (values -self.mean.to(values.device))/self.std.to(values.device)
            #
            # from misc.plots import plot_scatter_properties, plot_histogram_properties
            # plot_scatter_properties(1, 2, values, conditioning, self.ref_values)
            # plot_scatter_properties(0, 1, values, conditioning, self.ref_values)
            # plot_scatter_properties(0, 2, values, conditioning, self.ref_values)
            # plot_histogram_properties(1, values, self.ref_values)
            # plot_histogram_properties(2, values, self.ref_values)
            # plot_histogram_properties(0, values, self.ref_values)

            conditional_mae = torch.abs(values - conditioning).mean(0)
            conditional_metrics = {}
            for k, v in zip(('mol_weight', 'logP', 'QED'), conditional_mae):
                conditional_metrics[k] = v.item()
            print(conditional_metrics)

        # train_smiles, test_smiles = load_smiles(dataset=dataset)
        # metrics = get_all_metrics(gen_mols, test=test_smiles, train=train_smiles)
        metrics = mol_metric(gen_mols, dataset, num_no_correct, test_metrics=True)
        if conditioning is not None:
            metrics = metrics | conditional_metrics
        return metrics

    def generic_graph_metrics(self, gen_graphs, dataset):
        metrics = eval_graph_list(self.ref_graphs, gen_graphs, methods=['degree', 'cluster', 'orbit', 'spectral'])
        avg = sum(metrics.values()) / len(metrics.values())
        metrics['avg'] = avg
        if dataset == 'planar':
            u, n, v = eval_fraction_unique_non_isomorphic_valid(gen_graphs, self.ref_graphs,
                                                                validity_func=is_planar_graph)
            metrics['unique'], metrics['novel'], metrics['valid'] = u, n, v
            # plot_batch_networkx_graphs(gen_graphs[:30], filename='./misc/plots/planar/plan_gen')
        elif dataset == 'sbm':
            pass
            # u, n, v = eval_fraction_unique_non_isomorphic_valid(gen_graphs, self.ref_graphs,
            #                                                     validity_func=is_sbm_graph)
            # metrics['unique'], metrics['novel'], metrics['valid'] = u, n, v
            # plot_batch_networkx_graphs(gen_graphs[:20], filename='./misc/plots/sbm/sbm_gen')
        return metrics


    def eval_general_stats(self, X, A, mask, mask_adj):
        if self.dataset == 'zinc':
            node_names = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset == 'qm9':
            node_names = ['C', 'N', 'O', 'F']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset in ['qm9_cc', 'qm9_dg', 'qm9H']:
            node_names = ['H', 'C', 'N', 'O', 'F']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset in ['planar', 'sbm', 'enzymes']:
            node_names = []
            edge_names = ['no_edge', 'edge']

        if len(node_names) > 0:
            node_distrib = X[mask].bincount(minlength=X.shape[-1])
            node_distrib = node_distrib / node_distrib.sum()
            node_dict = {node_name: node_ratio for node_name, node_ratio in zip(node_names, node_distrib)}
        else:
            node_dict = {}

        edge_distrib = A[mask_adj].bincount(minlength=A.shape[-1])
        edge_distrib = edge_distrib / edge_distrib.sum()
        edge_dict = {edge_name: edge_ratio for edge_name, edge_ratio in zip(edge_names, edge_distrib)}

        return node_dict | edge_dict

class SpectralMetrics:
    def __init__(self, val_loader, max_num_nodes, test=None):
        self.val_loader = val_loader
        batch = next(iter(val_loader))
        X, A, mask = batch_to_dense(batch, max_num_nodes=max_num_nodes)
        self.val_graphs = get_networkx_from_dense(X.argmax(-1), A.argmax(-1), mask)
        # plot_batch_networkx_graphs(self.val_graphs[:13])

        if test is not None:
            batch = next(iter(test))
            X, A, mask = batch_to_dense(batch, max_num_nodes=max_num_nodes)
            self.test_graphs = get_networkx_from_dense(X.argmax(-1), A.argmax(-1), mask)
            #plot_batch_networkx_graphs(self.val_graphs[:40])
            pickle.dump(self.test_graphs, open(f'./dump/sbm_test', 'wb'))
            # for batch in self.val_loader:
            #     X, A, mask = batch_to_dense(batch, max_num_nodes=max_num_nodes)
            #     self.val_graphs = get_networkx_from_dense(X.argmax(-1), A.argmax(-1), mask)
            #     eval_graph_list(self.val_graphs, self.test_graphs, methods=['degree', 'cluster', 'orbit', 'spectral'])




    def eval_samples(self, X, A, mask, mask_adj, ema=False, n_run=0, conditional_values=None):
        ema_string = '_ema' if ema else ''
        distrib_metric = eval_general_stats(X, A, mask, mask_adj, self.dataset)
        if self.dataset in ['qm9', 'zinc', 'qm9_conditional']:
            metrics = self.get_metrics((X, A, mask), self.dataset, conditional_values)
        else:
            gen_graphs = get_networkx_from_dense(X, A, mask)
            metrics = self.get_metrics(gen_graphs, self.dataset, conditional_values)
            if self.save_graphs:
                import pickle
                fname = f'./dump/{self.dataset}_{self.transition}_{self.prior}_{self.id}{n_run + 1}{ema_string}'
                pickle.dump(gen_graphs, open(fname, 'wb'))
        if self.sampling:
            self.step = None
        else:
            metrics['epoch'] = self.epoch
        wandb.log({f'sampling{ema_string}/': metrics}, step=self.step)
        wandb.log({f'distributions{ema_string}/': distrib_metric}, step=self.step)
        print(f'sampling{ema_string}: ', metrics)
        print(f'distributions{ema_string}: ', distrib_metric)
        return metrics


    def eval_general_stats(self, X, A, mask, mask_adj):
        if self.dataset == 'zinc':
            node_names = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset == 'qm9':
            node_names = ['C', 'N', 'O', 'F']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset == 'qm9_conditional':
            node_names = ['H', 'C', 'N', 'O', 'F']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset in ['planar', 'sbm']:
            node_names = []
            edge_names = ['no_edge', 'edge']

        if len(node_names) > 0:
            node_distrib = X[mask].bincount(minlength=X.shape[-1])
            node_distrib = node_distrib / node_distrib.sum()
            node_dict = {node_name: node_ratio for node_name, node_ratio in zip(node_names, node_distrib)}
        else:
            node_dict = {}

        edge_distrib = A[mask_adj].bincount(minlength=A.shape[-1])
        edge_distrib = edge_distrib / edge_distrib.sum()
        edge_dict = {edge_name: edge_ratio for edge_name, edge_ratio in zip(edge_names, edge_distrib)}

        return node_dict | edge_dict