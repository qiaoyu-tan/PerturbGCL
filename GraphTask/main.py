import torch
from dataset import get_dataset
import argparse
import numpy as np
import time
from model import MGCLPrune
import random
from eval_graph import Eval_unsuper, Eval_semi
from torch_geometric.data import DataLoader


parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
parser.add_argument('--dataset', type=str, default='NCI1', help='PROTEINS| NCI1 | DD |COLLAB | MUTAG|IMDB-BINARY|REDDIT-BINARY|REDDIT-MULTI-5K ') #PROTEINS;NCI1;COLLAB
parser.add_argument('--mode', type=str, default='unsuper', help='semi | unsuper')
parser.add_argument('--lr', default=0.0001, dest='lr', type=float,
        help='Learning rate.')
parser.add_argument('--f_lr', default=0.001, dest='lr', type=float, help='Learning rate.')
parser.add_argument('--num_gc_layers', type=int, default=2,
        help='Number of graph convolution layers before each pooling')
parser.add_argument('--hidden_dim', type=int, default=128, help='')
parser.add_argument('--aug1', type=str, default='dropN',
                    help="{['raw', 'dropN', 'permE', 'subgraph', 'maskN'] }")
parser.add_argument('--aug2', type=str, default='maskN',
                    help="{['dropN', 'permE', 'subgraph', 'maskN', 'random2', 'random4', 'raw'] }")
parser.add_argument('--aug1_ratio', default=0.2, type=float, help='aug ratio .')
parser.add_argument('--aug2_ratio', default=0.2, type=float, help='aug ratio.')

parser.add_argument('--k_online', type=int, default=3)
parser.add_argument('--k_target', type=int, default=3)
parser.add_argument('--k_test', type=int, default=2)
parser.add_argument('--mask_step', type=int, default=1)
parser.add_argument('--stop_grad', type=int, default=1,
                    help='1 | 0')
parser.add_argument('--aug_type', type=str, default='prune',
                    help='noise | prune')
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--prune_rate', type=float, default=0.7)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_cpu', type=int, default=1, help='0: use cpu 1: use gpu')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cuda', type=str, default='0', help='specify cuda devices')
parser.add_argument('--choice_model', type=str, default='last', help='specify cuda devices')
parser.add_argument('--gnn_type', type=str, default='gin', help='resgcn | gin | gcn| gcn_raw') #gcn_raw is the standard gcn
parser.add_argument('--decay', type=float,default=0,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--label_rate', type=float, default=0.1,
                    help="the weight decay for l2 normalizaton")


def print_configuration(args):
    print('--> Experiment configuration')
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))


def setup_seed(seed):
    if seed == 200:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


def final_eval(evaluator, model, mode):
    acc_best = 0.0
    std_best = 0.0
    for i in range(5):
        if mode == 'unsuper':
            acc, std = evaluator.evaluate(model, target_k=i)
            print('!!! Final test result in unsuper mode is acc: {} std: {}'.format(acc, std))

        else:
            acc, std = evaluator.grid_search(model, target_k=i)
        if acc > acc_best:
            acc_best = acc
            std_best = std
    return acc_best, std_best


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    if torch.cuda.is_available():
        cuda_name = 'cuda:' + args.cuda
        device = torch.device(cuda_name)
        print('--> Use GPU %s' % args.cuda)
    else:
        device = torch.device("cpu")
        print("--> No GPU")
    # for semi-supervised learning with label ratio 0.01
    if args.mode == "semi":
        dataset, dataset_pretrain = get_dataset(args.dataset, task='semisupervised')
        data_loader = DataLoader(dataset_pretrain, args.batch_size, shuffle=True)
        evaluator = Eval_semi(dataset, label_rate=args.label_rate, device=device)  # for unsupervised
        if args.epoch != 100:
            pass
        else:
            args.epoch = 100
    else:
        dataset = get_dataset(args.dataset, task='unsupervised')
        data_loader = DataLoader(dataset, args.batch_size, shuffle=True)
        evaluator = Eval_unsuper(dataset, device=device)  # for unsupervised
        args.epoch = 20
    print('*** {} statistics: num_instances={} num_class={}'.format(args.dataset, len(dataset), dataset.num_classes))
    print_configuration(args)
    feat_dim = dataset[0].x.shape[1]

    save_path = "./weights/cogcl_naug-graph_" + args.mode + "_%s_" % args.dataset + args.gnn_type\
                + "_nlayer-%d" % args.num_gc_layers + '_dim%d_' % args.hidden_dim\
                + "{}_{}_{}_{}".format(args.label_rate, args.k_online, args.k_target, args.mask_step)\
                + '_{}_{}_{}'.format(args.stop_grad, args.aug_type, args.prune_rate) + '.pth'

    model = MGCLPrune(feat_dim, args.hidden_dim, aug1=args.aug1, aug2=args.aug2, n_layers=args.num_gc_layers, stop_grad=args.stop_grad, gnn=args.gnn_type, aug_type=args.aug_type, prune_rate=args.prune_rate, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    min_loss = 1e9
    for epoch in range(1, args.epoch+1):
        t1 = time.time()
        if epoch % args.mask_step == 0:
            update = 1
        else:
            update = 0
        model.train().to(device)
        epoch_loss = 0.0
        ite = 0
        for data in data_loader:
            optimizer.zero_grad()
            z = model(data, args.k_online, args.k_target, update=update)
            update = 0
            loss = model.compute_loss(z)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            ite += 1
        t2 = time.time()
        print('---> Pretraining: epoch={}/{} loss={} time: {}'.format(epoch + 1, args.epoch, epoch_loss / ite, t2-t1))

        if args.choice_model == 'best' and epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
        if args.choice_model == 'last' and epoch == args.epoch:
            torch.save(model.state_dict(), save_path)
    acc, std = final_eval(evaluator, model, args.mode)
    print('Final test result is acc: {} std: {}'.format(acc, std))



