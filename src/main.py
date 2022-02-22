import os
import argparse
import data_processing
import train


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')

    parser.add_argument('--dataset', type=str, default='QM9', help='dataset name')
    parser.add_argument('--n_molecules', type=int, default=1000, help='the number of molecules to be sampled')
    parser.add_argument('--n_pairs', type=int, default=10000, help='the number of molecule pairs to be sampled')

    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--gnn', type=str, default='gcn', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=3, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=2048, help='dimension of molecule embeddings')
    parser.add_argument('--mlp_hidden_unit', type=int, default=100, help='unit of the hidden layer in MLP')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--feature_mode', type=str, default='concat', help='how to construct the input feature')

    args = parser.parse_args()
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    data = data_processing.load_data(args)
    train.train(args, data)


if __name__ == '__main__':
    main()
