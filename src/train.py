import math
import torch
from model import GNN


def train(args, data):
    train_loader, val_loader, test_loader, feature_len = data
    model = GNN(args.gnn, args.layer, feature_len, args.dim, args.mlp_hidden_unit, args.feature_mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda(args.gpu)

    print('start training\n')

    print('initial case:')
    evaluate(model, 'train', train_loader)
    evaluate(model, 'val', val_loader)
    evaluate(model, 'test', test_loader)
    print()

    for i in range(args.epoch):
        print('epoch %d:' % i)

        # train
        model.train()
        for graph1, graph2, target in train_loader:
            pred = torch.squeeze(model(graph1, graph2))
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        evaluate(model, 'train', train_loader)
        evaluate(model, 'val', val_loader)
        evaluate(model, 'test', test_loader)
        print()


def evaluate(model, mode, data):
    pred_list = []
    target_list = []
    model.eval()
    with torch.no_grad():
        for graph1, graph2, target in data:
            pred = torch.squeeze(model(graph1, graph2))
            pred_list.append(pred)
            target_list.append(target)
    pred_list = torch.concat(pred_list)
    target_list = torch.concat(target_list)
    mae = torch.linalg.norm(pred_list - target_list, ord=1) / len(pred_list)
    rmse = torch.linalg.norm(pred_list - target_list, ord=2) / math.sqrt(len(pred_list))
    print('%s  mae: %.4f  rmse: %.4f' % (mode, float(mae), float(rmse)))
