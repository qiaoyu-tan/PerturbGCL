import os.path as osp
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')
    print (root_path)
    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
#        return Coauthor(root=path, name='cs')
    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
#        return Amazon(root=path, name='computers')

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
#        return Amazon(root=path, name='photo')

    if name == 'ogbn-arxiv':
        data = read_ogb_dataset(name=name, path=osp.join(root_path, 'OGB'))
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        return [data]

    if name == "ogbn-products":
        data = read_ogb_dataset(name=name, path=osp.join(root_path, 'OGB'))
        return [data]

        # return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def read_ogb_dataset(name: str, path: str) -> Data:
    dataset = PygNodePropPredDataset(root=path, name=name)
    split_idx = dataset.get_idx_split()

    data = dataset[0]

    data.train_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.train_mask[split_idx["train"]] = True

    data.val_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.val_mask[split_idx["valid"]] = True

    data.test_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.test_mask[split_idx["test"]] = True

    data.y = data.y.squeeze(dim=-1)

    return data

def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)
