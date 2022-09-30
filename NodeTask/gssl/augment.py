from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.data import Data
from torch_geometric.data.sampler import EdgeIndex
from pGRACE.functional import drop_feature
from torch_geometric.utils import dropout_adj


class GraphAugmentor:
    """Masks node features (same for all nodes) and drops edges."""

    def __init__(
        self,
        p_x_1: float,
        p_e_1: float,
        p_x_2: Optional[float] = None,
        p_e_2: Optional[float] = None,
    ):
        self._p_x_1 = p_x_1
        self._p_e_1 = p_e_1

        self._p_x_2 = p_x_2 if p_x_2 is not None else p_x_1
        self._p_e_2 = p_e_2 if p_e_2 is not None else p_e_1

    def __call__(self, data: Data):
        """Augment full-batch graph."""
        # x_a = mask_features(data.x, p=self._p_x_1)
        # x_b = mask_features(data.x, p=self._p_x_2)

        x_a = drop_feature(data.x, self._p_x_1)
        x_b = drop_feature(data.x, self._p_x_2)

        edge_index_a = dropout_adj(data.edge_index, p=self._p_e_1)[0]
        edge_index_b = dropout_adj(data.edge_index, p=self._p_e_2)[0]

        return (x_a, edge_index_a), (x_b, edge_index_b)

    def augment_batch(
        self,
        x: torch.Tensor,
        adjs: List[EdgeIndex],
    ):
        """Augment batch from NeighborSampler."""
        x_a = mask_features(x, p=self._p_x_1)
        x_b = mask_features(x, p=self._p_x_2)

        edge_indexes_a = [
            drop_edges(adj.edge_index, p=self._p_e_1)
            for adj in adjs
        ]
        edge_indexes_b = [
            drop_edges(adj.edge_index, p=self._p_e_2)
            for adj in adjs
        ]

        return (x_a, edge_indexes_a), (x_b, edge_indexes_b)


def mask_features(x: torch.Tensor, p: float) -> torch.Tensor:
    num_features = x.size(-1)
    device = x.device

    return bernoulli_mask(size=(1, num_features), prob=p).to(device) * x


def drop_edges(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    num_edges = edge_index.size(-1)
    device = edge_index.device

    mask = bernoulli_mask(size=num_edges, prob=p).to(device) == 1.

    return edge_index[:, mask]


def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return torch.bernoulli((1 - prob) * torch.ones(size))
