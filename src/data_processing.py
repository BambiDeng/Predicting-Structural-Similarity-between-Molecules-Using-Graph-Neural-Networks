import os
import random
import dgl
import torch
import pysmiles
import itertools
import numpy as np
from time import time
from collections import defaultdict
from networkx.algorithms.similarity import graph_edit_distance
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader

random.seed(0)
attribute_names = ['element', 'charge', 'aromatic', 'hcount']
train_ratio = 0.8
val_ratio = 0.1


class GEDPredDataset(dgl.data.DGLDataset):
    def __init__(self, args):
        self.args = args
        self.path = '../data/' + args.dataset + '/'
        self.graphs1 = []
        self.graphs2 = []
        self.targets = []
        self.feature_len = 0
        super().__init__(name='ged_pred_' + args.dataset)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.args.dataset + ' dataset to GPU')
            self.graphs1 = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.graphs1]
            self.graphs2 = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.graphs2]

    def save(self):
        print('saving ' + self.args.dataset + ' dataset to ' + self.path + 'ged0.bin and ' + self.path + 'ged1.bin')
        dgl.save_graphs(self.path + 'ged0.bin', self.graphs1, {'target': self.targets})
        dgl.save_graphs(self.path + 'ged1.bin', self.graphs2, {'feature_len': torch.Tensor([self.feature_len])})

    def load(self):
        print('loading ' + self.args.dataset + ' dataset from ' + self.path + 'ged0.bin and ' + self.path + 'ged1.bin')
        self.graphs1, self.targets = dgl.load_graphs(self.path + 'ged0.bin')
        self.graphs2, self.feature_len = dgl.load_graphs(self.path + 'ged1.bin')
        self.targets = self.targets['target']
        self.feature_len = int(self.feature_len['feature_len'])
        self.to_gpu()

    def process(self):
        molecule_list, all_values = self.read_data()
        print(all_values)
        exit(0)
        feature_encoder = get_feature_encoder(all_values)
        self.feature_len = sum([len(feature_encoder[key]) for key in attribute_names])
        samples = self.sample(molecule_list)
        res = calculate_ged(samples)

        with open(self.path + 'pairwise_ged.csv', 'w') as f:
            f.writelines('smiles1,smiles2,ged\n')
            for g1, g2, s1, s2, ged in res:
                self.graphs1.append(networkx_to_dgl(g1, feature_encoder))
                self.graphs2.append(networkx_to_dgl(g2, feature_encoder))
                self.targets.append(ged)
                f.writelines(s1 + ',' + s2 + ',' + str(ged) + '\n')
        self.targets = torch.Tensor(self.targets)
        self.to_gpu()

    def has_cache(self):
        if os.path.exists(self.path + 'ged0.bin') and os.path.exists(self.path + 'ged1.bin'):
            print('cache found')
            return True
        else:
            print('cache not found')
            return False

    def __getitem__(self, i):
        return self.graphs1[i], self.graphs2[i], self.targets[i]

    def __len__(self):
        return len(self.graphs1)

    def read_data(self):
        print('retrieving the first %d molecules from %s dataset' % (self.args.n_molecules, self.args.dataset))
        molecule_list = []
        all_values = defaultdict(set)
        with open(self.path + self.args.dataset + '.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0 or line == '\n':
                    continue
                if idx > self.args.n_molecules:
                    break
                items = line.strip().split(',')

                if self.args.dataset == 'QM9':
                    smiles = items[1]
                else:
                    raise ValueError('unknown dataset')

                raw_graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
                molecule_list.append((raw_graph, smiles))

                for attr in attribute_names:
                    for _, value in raw_graph.nodes(data=attr):
                        all_values[attr].add(value)

        return molecule_list, all_values

    def sample(self, molecule_list):
        print('sampling %d pairs' % self.args.n_pairs)
        all_pairs = list(itertools.combinations(molecule_list, 2))
        samples = random.sample(all_pairs, self.args.n_pairs)
        return samples


def calculate_ged(samples):
    def node_match(n1, n2):
        return n1['element'] == n2['element'] and n1['charge'] == n2['charge']

    def edge_match(e1, e2):
        return e1['order'] == e2['order']

    print('calculating GED')
    t = time()
    res = []
    for i, graph_pair in enumerate(samples):
        g1, g2 = graph_pair
        graph1, smiles1 = g1
        graph2, smiles2 = g2
        ged = graph_edit_distance(graph1, graph2, node_match=node_match, edge_match=edge_match)
        res.append((graph1, graph2, smiles1, smiles2, ged))
        if i % 100 == 0:
            print('%d / %d' % (i, len(samples)))
    print('%.1f s' % (time() - t))
    return res


def get_feature_encoder(all_values):
    feature_encoder = {}
    idx = 0
    # key: attribute; values: all possible values of the attribute
    for key, values in all_values.items():
        feature_encoder[key] = {}
        for value in values:
            feature_encoder[key][value] = idx
            idx += 1
        # for each attribute, we add an "unknown" key to handle unknown values during inference
        feature_encoder[key]['unknown'] = idx
        idx += 1

    return feature_encoder


def networkx_to_dgl(raw_graph, feature_encoder):
    # add edges
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    # add node features
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if raw_feature[j] in feature_encoder[j]:
                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]['unknown'])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata['feature'] = node_features
    # transform to bi-directed graph with self-loops
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    return graph


def split_dataset(args, data):
    dataset_size = len(data)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:int(train_ratio * dataset_size)]
    val_indices = indices[int(train_ratio * dataset_size):int((train_ratio + val_ratio) * dataset_size)]
    test_indices = indices[int((train_ratio + val_ratio) * dataset_size):]
    train_loader = GraphDataLoader(data, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = GraphDataLoader(data, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_indices))
    test_loader = GraphDataLoader(data, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_indices))

    return train_loader, val_loader, test_loader


def load_data(args):
    data = GEDPredDataset(args)
    train_loader, val_loader, test_loader = split_dataset(args, data)
    return train_loader, val_loader, test_loader, data.feature_len
