import argparse
import os.path as osp
import time
import numpy as np
import torch
from gssl.augment import GraphAugmentor
from model import GNNEncoderPrune, MGCLPruneAug
from pGRACE.eval import log_regressionNew as log_regression, MulticlassEvaluator
from pGRACE.dataset import get_dataset
from gssl.utils import seed

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=20)
# model parameter
parser.add_argument('--lr_base', type=float, default=5.e-4)
parser.add_argument('--tau', type=float, default=0.4)
parser.add_argument('--weight_decay', type=float, default=1.e-5)
parser.add_argument('--gnn_type', type=str, default='GCN',
                    help='GCN | GAT')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--emb_dim', type=int, default=512)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--pred_dim', type=int, default=512)
parser.add_argument('--use_pred', type=int, default=0)
parser.add_argument('--activation', type=str, default='relu')
# augmentation parameters
parser.add_argument('--k_online', type=int, default=2)
parser.add_argument('--k_target', type=int, default=2)
parser.add_argument('--k_test', type=int, default=2)
parser.add_argument('--prune_rate', type=float, default=0.9)
parser.add_argument('--dataset', type=str, default='Coauthor-CS')
parser.add_argument('--loss_type', type=str, default='cl')
parser.add_argument('--sym_loss', type=int, default=0)
parser.add_argument('--mask_step', type=int, default=1)
parser.add_argument('--stop_grad', type=int, default=1,
                    help='1 | 0')
parser.add_argument('--aug_type', type=str, default='prune',
                    help='noise | prune')
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--p_f_1', type=float, default=0.5)
parser.add_argument('--p_f_2', type=float, default=0.5)
parser.add_argument('--p_e_1', type=float, default=0.4)
parser.add_argument('--p_e_2', type=float, default=0.4)
# data split parameter
parser.add_argument('--train_examples_per_class', type=int, default=20,
                    help='hidden dimension.')
parser.add_argument('--val_examples_per_class', type=int, default=30,
                    help='hidden dimension.')
# evaluation parameter
parser.add_argument('--concat', type=int, default=0)


def train(model, data, optimizer, device, update, k_online, k_target, args):
    model.train()
    optimizer.zero_grad()

    if args.use_pred:
        if args.sym_loss:
            _h1, _h2, _h1_pred, _h2_pred = model(data.to(device), k_online, k_target, update)
            loss = model.compute_loss_sym(_h1, _h2, _h1_pred, _h2_pred)
        else:
            _h1, _h2, _h1_pred = model(data.to(device), k_online, k_target, update)
            loss = model.compute_loss(_h1_pred, _h2)
    else:
        _h1, _h2 = model(data.to(device), k_online, k_target, update)
        loss = model.compute_loss(_h1, _h2)

    loss.backward()
    optimizer.step()

    return loss.item()


def test_onetime(model, data, dataset, k_test, args):
    model.eval()
    z = model.testforward(data, k_test)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            result = log_regression(z, dataset, evaluator, split='wikics:{}'.format(i))
            accs.append(result)
        acc_val = np.mean([acc_['val_acc'] for acc_ in accs])
        acc_test = np.mean([acc_['test_acc'] for acc_ in accs])
        result = dict()
        result['val_acc'] = acc_val
        result['test_acc'] = acc_test
    else:
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            result = log_regression(z, dataset, evaluator, split='preloaded')
        else:
            result = log_regression(z, dataset, evaluator)
    return result


def test(model, dataset, data, args):

    # model.eval()
    # z = model.testforward(data, args.k_test)
    best_val = 0
    best_result = None
    for i in range(5):
        result = test_onetime(model, data, dataset, i, args)
        if result['val_acc'] > best_val:
            best_result = result
            best_val = result['val_acc']
    return best_result


def train_eval(dataset, data, augmentor, num_node_features, model_path, device, run, args):
    encoder = GNNEncoderPrune(in_channels=num_node_features,
                              hidden_channels=args.hidden_dim,
                              out_channels=args.emb_dim,
                              num_layers=args.num_layers,
                              gnn_type=args.gnn_type,
                              fix_k=args.k_test,
                              aug_type=args.aug_type,
                              act=args.activation)
    model = MGCLPruneAug(encoder=encoder,
                         augmentor=augmentor,
                         stop_grad=args.stop_grad,
                         out_dim=args.emb_dim,
                         pred_dim=args.pred_dim,
                         eta=args.eta,
                         sym_loss=args.sym_loss,
                         prune_rate=args.prune_rate,
                         use_pred=args.use_pred,
                         aug_type=args.aug_type,
                         loss_type=args.loss_type).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_base,
        weight_decay=args.weight_decay
    )

    best_epoch = 0
    best_val = 0
    best_test = 0
    best_k = 0
    for epoch in range(1, args.total_epochs + 1):
        t1 = time.time()
        if epoch % args.mask_step == 0:
            update = 1
        else:
            update = 0

        loss = train(model, data, optimizer, device, update, args.k_online, args.k_target, args)
        print('Epoch: {} loss:{} time: {}'.format(epoch, loss, time.time()-t1))

        if epoch % args.log_interval == 0 or epoch == args.total_epochs - 1:
            result = test(model, dataset, data, args)
            print(
                'current epoch: {} val_acc: {} test_acc:{} best_k:{}'.format(epoch,result['val_acc'], result['test_acc'], result['test_k']))
            if result['val_acc'] > best_val:
                best_val = result['val_acc']
                best_test = result['test_acc']
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print('Run: {} current epoch: {} val_acc: {} test_acc:{} best_acc_val:{} best_acc_test:{}'.format(run, epoch,
                                  result['val_acc'], result['test_acc'], best_val, best_test))

    print('Run: {} final result best_epoch: {} best_val: {} test_acc:{}'.format(run, best_epoch, best_val, best_test))
    return best_val, best_test


def main():
    args = parser.parse_args()
    seed(args.seed)
    # Read params
    # with open("experiments/configs/full_batch/train_bgrl_final.yaml", "r") as fin:
    #     params = yaml.safe_load(fin)[dataset_name]
    # params['dataset'] = dataset_name
    params = dict()
    print(args)

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]
    data = data.to(device)
    num_node_features = data.x.shape[1]

    val_metrics = []
    test_metrics = []

    for run in range(args.runs):
        augmentor = GraphAugmentor(
            p_x_1=args.p_f_1,
            p_x_2=args.p_f_2,
            p_e_1=args.p_e_1,
            p_e_2=args.p_e_2,
        )
        model_path = 'weight/cogclfinal_{}_{}_{}_{}_{}_{}_{}_{}_model_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(
            args.dataset, args.p_f_1, args.p_f_2, args.p_e_1, args.p_e_2,
            args.emb_dim, args.hidden_dim, args.pred_dim, args.k_online,
            args.k_target, args.k_test, args.loss_type, args.stop_grad,
            args.prune_rate, args.sym_loss, args.activation, args.aug_type, run)

        best_val, best_test = train_eval(dataset, data, augmentor, num_node_features, model_path, device, run, args)
        val_metrics.append(best_val)
        test_metrics.append(best_test)

    if args.runs > 1:
        mean_val = np.mean(val_metrics)
        std_val = np.std(val_metrics)
        mean_test = np.mean(test_metrics)
        std_test = np.std(test_metrics)
        print(
            'mean_val: {} std_val: {} mean_test:{} std_test:{}'.format(
                mean_val, std_val, mean_test, std_test))
    else:
        mean_val = val_metrics[0]
        std_val = 0.0
        mean_test = test_metrics[0]
        std_test = 0.0
        print(
            'mean_val: {} std_val: {} mean_test:{} std_test:{}'.format(
                mean_val, std_val, mean_test, std_test))


if __name__ == '__main__':
    main()


