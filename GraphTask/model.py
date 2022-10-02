from functools import partial
import torch
import torch.nn.functional as F
from layers import ResGCNNew, GINNew
from torch.nn import Parameter
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_scatter import scatter_add
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
import numpy as np
from method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, \
    UniformSample, RWSample, RandomView, RawView


class GCL_model(torch.nn.Module):
    r"""A wrapped :class:`torch.nn.Module` class for the convinient instantiation of
    pre-implemented graph encoders.

    Args:
        feat_dim (int): The dimension of input node features.
        hidden_dim (int): The dimension of node-level (local) embeddings.
        n_layer (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
        pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
            (default: :obj:`sum`)
        gnn (string, optional): The type of GNN layer, :obj:`gcn`, :obj:`gin` or
            :obj:`resgcn`. (default: :obj:`gin`)
        bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
        act (string, optional): The activation function, :obj:`relu` or :obj:`prelu`.
            (default: :obj:`relu`)
        bias (bool, optional): Whether to include bias term in Linear. (default: :obj:`True`)
        xavier (bool, optional): Whether to apply xavier initialization. (default: :obj:`True`)
        node_level (bool, optional): If :obj:`True`, the encoder will output node level
            embedding (local representations). (default: :obj:`False`)
        graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
            embeddings (global representations). (default: :obj:`True`)
        edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
            compute the aggregation. (default: :obj:`False`)

    Note
    ----
    For GCN and GIN encoders, the dimension of the output node-level (local) embedding will be
    :obj:`hidden_dim`, whereas the node-level embedding will be :obj:`hidden_dim` * :obj:`n_layers`.
    For ResGCN, the output embeddings for boths node and graphs will have dimensions :obj:`hidden_dim`.

    Examples
    """

    def __init__(self, feat_dim, hidden_dim, n_layers=5, pool='sum',
                 gnn='gin', bn=True, act='relu', bias=True, proj_head='MLP', xavier=True, tau=0.5, device=None,
                 object_loss='NCE', node_level=False, graph_level=True, edge_weight=False):
        super(GCL_model, self).__init__()

        self.tau = tau
        self.proj_head = proj_head
        self.hidden_dim = hidden_dim
        self.proj_out_dim = hidden_dim
        self.object_loss = object_loss
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        if gnn == 'gin':
            self.encoder = GIN(feat_dim, hidden_dim, n_layers, pool, bn, act)
        elif gnn == 'gcn':
            self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'gcn_raw':
            self.encoder = GCNRaw(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'resgcn':
            self.encoder = ResGCN(feat_dim, hidden_dim, num_conv_layers=n_layers,
                                  global_pool=pool)
        # self.views_fn = self.aug_funs()
        self.node_level = node_level
        self.graph_level = graph_level
        if gnn == "gin" or gnn == "gcn":
            self.d_out = self.hidden_dim * n_layers
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim * n_layers)
        else:
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim)
            self.d_out = self.hidden_dim

    def _get_proj(self, proj_head, in_dim):

        if callable(proj_head):
            return proj_head

        assert proj_head in ['linear', 'MLP']

        # out_dim = self.proj_out_dim
        out_dim = self.proj_out_dim

        if proj_head == 'linear':
            proj_nn = nn.Linear(in_dim, out_dim)
            self._weights_init(proj_nn)
        elif proj_head == 'MLP':
            proj_nn = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_dim, out_dim))
            for m in proj_nn.modules():
                self._weights_init(m)

        return proj_nn

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def encode_view(self, data):
        z_g = self.encoder(data)
        return z_g

    def forward(self, data, sampler):
        zs = []
        # views = [v_fn(data).to(self.device) for v_fn in self.views_fn]
        x = self.encode_view(data.to(self.device))
        data_v1, data_v2, _ = sampler.sample(data, x)
        z = self.encode_view(data_v1.to(self.device))
        z = self.proj_head_g(z)
        zs.append(z)
        z = self.encode_view(data_v2.to(self.device))
        z = self.proj_head_g(z)
        zs.append(z)
        return zs

    def compute_pred_and_logits(self, data, sampler):
        xx = self.encode_view(data.to(self.device))
        data_v1, data_v2, actions, logits = sampler.sample(data, xx, return_logits=True)
        z1 = self.encode_view(data_v1.to(self.device))
        z1 = self.proj_head_g(z1)
        z2 = self.encode_view(data_v2.to(self.device))
        z2 = self.proj_head_g(z2)
        loss = self.compute_reward(z1, z2, tau=self.tau, reduce=False)
        return loss, actions, logits


    def compute_loss(self, zs_n):
        loss = self.NT_Xent(zs_n[0], zs_n[1], tau=self.tau)
        return loss

    def NT_Xent(self, z1, z2, tau=0.5, norm=True, reduce=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        if reduce:
            loss = - torch.log(loss).mean()
        return loss

    def compute_reward(self, z1, z2, tau=0.5, norm=True, reduce=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # pos_sim = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # if reduce:
        #     loss = - torch.log(loss).mean()
        return pos_sim


class GCL_SOTA(torch.nn.Module):
    r"""A wrapped :class:`torch.nn.Module` class for the convinient instantiation of
    pre-implemented graph encoders.

    Args:
        feat_dim (int): The dimension of input node features.
        hidden_dim (int): The dimension of node-level (local) embeddings.
        n_layer (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
        pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
            (default: :obj:`sum`)
        gnn (string, optional): The type of GNN layer, :obj:`gcn`, :obj:`gin` or
            :obj:`resgcn`. (default: :obj:`gin`)
        bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
        act (string, optional): The activation function, :obj:`relu` or :obj:`prelu`.
            (default: :obj:`relu`)
        bias (bool, optional): Whether to include bias term in Linear. (default: :obj:`True`)
        xavier (bool, optional): Whether to apply xavier initialization. (default: :obj:`True`)
        node_level (bool, optional): If :obj:`True`, the encoder will output node level
            embedding (local representations). (default: :obj:`False`)
        graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
            embeddings (global representations). (default: :obj:`True`)
        edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
            compute the aggregation. (default: :obj:`False`)

    Note
    ----
    For GCN and GIN encoders, the dimension of the output node-level (local) embedding will be
    :obj:`hidden_dim`, whereas the node-level embedding will be :obj:`hidden_dim` * :obj:`n_layers`.
    For ResGCN, the output embeddings for boths node and graphs will have dimensions :obj:`hidden_dim`.

    Examples
    """

    def __init__(self, feat_dim, hidden_dim, n_layers=5, pool='sum', aug1='random2', aug2='random2',
                 gnn='gin', bn=True, act='relu', bias=True, proj_head='MLP', xavier=True, tau=0.5, device=None,
                 object_loss='NCE', node_level=False, graph_level=True, edge_weight=False):
        super(GCL_SOTA, self).__init__()

        self.tau = tau
        self.proj_head = proj_head
        self.hidden_dim = hidden_dim
        self.proj_out_dim = hidden_dim
        self.object_loss = object_loss
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        if gnn == 'gin':
            self.encoder = GIN(feat_dim, hidden_dim, n_layers, pool, bn, act)
        elif gnn == 'gcn':
            self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'gcn_raw':
            self.encoder = GCNRaw(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'resgcn':
            self.encoder = ResGCN(feat_dim, hidden_dim, num_conv_layers=n_layers,
                                  global_pool=pool)
        # self.views_fn = self.aug_funs()
        self.node_level = node_level
        self.graph_level = graph_level
        if gnn == "gin" or gnn == "gcn":
            self.d_out = self.hidden_dim * n_layers
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim * n_layers)
        else:
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim)
            self.d_out = self.hidden_dim
        self.views_fn = self.view_fns(aug1, aug2)

    def view_fns(self, aug_1, aug_2, aug_ratio=0.2):
        views_fn = []

        for aug in [aug_1, aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'dropN':
                views_fn.append(UniformSample(ratio=aug_ratio))
            elif aug == 'permE':
                views_fn.append(EdgePerturbation(ratio=aug_ratio))
            elif aug == 'subgraph':
                views_fn.append(RWSample(ratio=aug_ratio))
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(mask_ratio=aug_ratio))
            elif aug == 'random2':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random4':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random3':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio),
                              NodeAttrMask(mask_ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            else:
                raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                        'maskN', 'random2', 'random3', 'random4'] or None.")
        return views_fn

    def _get_proj(self, proj_head, in_dim):

        if callable(proj_head):
            return proj_head

        assert proj_head in ['linear', 'MLP']

        # out_dim = self.proj_out_dim
        out_dim = self.proj_out_dim

        if proj_head == 'linear':
            proj_nn = nn.Linear(in_dim, out_dim)
            self._weights_init(proj_nn)
        elif proj_head == 'MLP':
            proj_nn = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_dim, out_dim))
            for m in proj_nn.modules():
                self._weights_init(m)

        return proj_nn

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def encode_view(self, data):
        z_g = self.encoder(data)
        return z_g

    def forward(self, data):
        zs = []
        views = [v_fn(data).to(self.device) for v_fn in self.views_fn]
        z = self.encode_view(views[0])
        z = self.proj_head_g(z)
        zs.append(z)
        z = self.encode_view(views[1])
        z = self.proj_head_g(z)
        zs.append(z)
        return zs

    def compute_pred_and_logits(self, data, sampler):
        xx = self.encode_view(data.to(self.device))
        data_v1, data_v2, actions, logits = sampler.sample(data, xx, return_logits=True)
        z1 = self.encode_view(data_v1.to(self.device))
        z1 = self.proj_head_g(z1)
        z2 = self.encode_view(data_v2.to(self.device))
        z2 = self.proj_head_g(z2)
        loss = self.compute_reward(z1, z2, tau=self.tau, reduce=False)
        return loss, actions, logits

    def compute_loss(self, zs_n):
        loss = self.NT_Xent(zs_n[0], zs_n[1], tau=self.tau)
        return loss

    def NT_Xent(self, z1, z2, tau=0.5, norm=True, reduce=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        if reduce:
            loss = - torch.log(loss).mean()
        return loss

    def compute_reward(self, z1, z2, tau=0.5, norm=True, reduce=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # pos_sim = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # if reduce:
        #     loss = - torch.log(loss).mean()
        return pos_sim


class MGCLPruneAug(torch.nn.Module):
    r"""A wrapped :class:`torch.nn.Module` class for the convinient instantiation of
    pre-implemented graph encoders.

    Args:
        feat_dim (int): The dimension of input node features.
        hidden_dim (int): The dimension of node-level (local) embeddings.
        n_layer (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
        pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
            (default: :obj:`sum`)
        gnn (string, optional): The type of GNN layer, :obj:`gcn`, :obj:`gin` or
            :obj:`resgcn`. (default: :obj:`gin`)
        bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
        act (string, optional): The activation function, :obj:`relu` or :obj:`prelu`.
            (default: :obj:`relu`)
        bias (bool, optional): Whether to include bias term in Linear. (default: :obj:`True`)
        xavier (bool, optional): Whether to apply xavier initialization. (default: :obj:`True`)
        node_level (bool, optional): If :obj:`True`, the encoder will output node level
            embedding (local representations). (default: :obj:`False`)
        graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
            embeddings (global representations). (default: :obj:`True`)
        edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
            compute the aggregation. (default: :obj:`False`)

    Note
    ----
    For GCN and GIN encoders, the dimension of the output node-level (local) embedding will be
    :obj:`hidden_dim`, whereas the node-level embedding will be :obj:`hidden_dim` * :obj:`n_layers`.
    For ResGCN, the output embeddings for boths node and graphs will have dimensions :obj:`hidden_dim`.

    Examples
    """

    def __init__(self, feat_dim, hidden_dim, n_layers=5, pool='sum', aug1='random2', aug2='random2',
                 gnn='gin', bn=True, act='relu', bias=True, proj_head='MLP', xavier=True, tau=0.5, device=None,
                 object_loss='NCE', node_level=False, stop_grad=1, aug_type='prune', prune_rate=0.3, eta=0.1, graph_level=True, edge_weight=False):
        super(MGCLPruneAug, self).__init__()

        self.tau = tau
        self.proj_head = proj_head
        self.hidden_dim = hidden_dim
        self.proj_out_dim = hidden_dim
        self.object_loss = object_loss
        self.aug_type = aug_type
        self.prune_rate = prune_rate
        self.eta = eta
        self.stop_grad = stop_grad
        self.n_layers = n_layers

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        self.gnn_type = gnn
        if gnn == 'gin':
            self.encoder = GINNew(feat_dim, hidden_dim, n_layers, pool, bn, act, aug_type=aug_type)
        elif gnn == 'gcn':
            self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'gcn_raw':
            self.encoder = GCNRaw(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'resgcn':
            self.encoder = ResGCNNew(feat_dim, hidden_dim, num_conv_layers=n_layers,
                                  global_pool=pool, aug_type=aug_type)
        # self.views_fn = self.aug_funs()
        self.node_level = node_level
        self.graph_level = graph_level
        if gnn == "gin" or gnn == "gcn":
            self.d_out = self.hidden_dim * n_layers
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim * n_layers)
        else:
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim)
            self.d_out = self.hidden_dim
        self.views_fn = self.view_fns(aug1, aug2)

    def view_fns(self, aug_1, aug_2, aug_ratio=0.2):
        views_fn = []

        for aug in [aug_1, aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'dropN':
                views_fn.append(UniformSample(ratio=aug_ratio))
            elif aug == 'permE':
                views_fn.append(EdgePerturbation(ratio=aug_ratio))
            elif aug == 'subgraph':
                views_fn.append(RWSample(ratio=aug_ratio))
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(mask_ratio=aug_ratio))
            elif aug == 'random2':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random4':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random3':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio),
                              NodeAttrMask(mask_ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            else:
                raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                        'maskN', 'random2', 'random3', 'random4'] or None.")
        return views_fn

    def _get_proj(self, proj_head, in_dim):

        if callable(proj_head):
            return proj_head

        assert proj_head in ['linear', 'MLP']

        # out_dim = self.proj_out_dim
        out_dim = self.proj_out_dim

        if proj_head == 'linear':
            proj_nn = nn.Linear(in_dim, out_dim)
            self._weights_init(proj_nn)
        elif proj_head == 'MLP':
            proj_nn = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_dim, out_dim))
            for m in proj_nn.modules():
                self._weights_init(m)

        return proj_nn

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def encode_view(self, data, layer_K, prune: bool = False, finetune=False):
        z_g = self.encoder(data, layer_K, prune, finetune=finetune)
        return z_g

    def get_target_encoder(self):

        if self.gnn_type == 'resgcn':
            if self.aug_type == 'noise':
                for i in range(self.n_layers):
                    weight_, _, bias_, _ = self.encoder.convs[i].get_weight()
                    bias_ = torch.from_numpy(bias_)
                    weight_ = torch.from_numpy(weight_)
                    weight_1_noise = self.eta * torch.normal(0, torch.ones_like(weight_.data) * weight_.data.std())
                    bias_1_noise = self.eta * torch.normal(0, torch.ones_like(bias_.data) * bias_.data.std())
                    self.encoder.convs[i].update_weight2noise(weight_1_noise, bias_1_noise)
            else:
                weight_all = []
                weight_all_mask = []
                for i in range(self.n_layers):
                    weight_1, weight1_mask, bias_1, bias_1_mask = self.encoder.convs[i].get_weight()
                    weight_all.extend([weight_1, bias_1])
                    weight_all_mask.extend([weight1_mask, bias_1_mask])
                weight_vector = np.concatenate([v.flatten() for v in weight_all])

                # number_of_remaining_weights = np.sum([np.sum(v) for v in weight_all_mask])
                number_of_remaining_weights = weight_vector.shape[0]

                number_of_weights_to_prune_magnitude = np.ceil(
                    self.prune_rate * number_of_remaining_weights).astype(
                    int)
                threshold = np.sort(np.abs(weight_vector))[
                    min(number_of_weights_to_prune_magnitude,
                        len(weight_vector) - 1)]
                for i in range(self.n_layers):
                    self.encoder.convs[i].update_weight2prune(threshold)
        elif self.gnn_type == 'gin':
            if self.aug_type == 'noise':
                for i in range(self.n_layers):
                    weight_1, bias_1, weight_2, bias_2 = self.encoder.convs[i].get_weight()
                    weight_1 = torch.from_numpy(weight_1)
                    bias_1 = torch.from_numpy(bias_1)
                    weight_2 = torch.from_numpy(weight_2)
                    bias_2 = torch.from_numpy(bias_2)
                    weight_1_noise = self.eta * torch.normal(0, torch.ones_like(weight_1.data) * weight_1.data.std())
                    bias_1_noise = self.eta * torch.normal(0, torch.ones_like(bias_1.data) * bias_1.data.std())
                    weight_2_noise = self.eta * torch.normal(0, torch.ones_like(weight_2.data) * weight_2.data.std())
                    bias_2_noise = self.eta * torch.normal(0, torch.ones_like(bias_2.data) * bias_2.data.std())
                    self.encoder.convs[i].update_weight2noise([weight_1_noise, weight_2_noise],
                                                              [bias_1_noise, bias_2_noise])
            else:
                weight_all = []
                for i in range(self.n_layers):
                    weight_1, bias_1, weight_2, bias_2 = self.encoder.convs[i].get_weight()
                    weight_all.extend([weight_1, bias_1, weight_2, bias_2])
                weight_vector = np.concatenate([v.flatten() for v in weight_all])

                # number_of_remaining_weights = np.sum([np.sum(v) for v in weight_all_mask])
                number_of_remaining_weights = weight_vector.shape[0]

                number_of_weights_to_prune_magnitude = np.ceil(
                    self.prune_rate * number_of_remaining_weights).astype(
                    int)
                threshold = np.sort(np.abs(weight_vector))[
                    min(number_of_weights_to_prune_magnitude,
                        len(weight_vector) - 1)]
                for i in range(self.n_layers):
                    self.encoder.convs[i].update_weight2prune(threshold)
        else:
            raise ValueError('the gnn type: {} is not supported'.format(self.gnn_type))

    def forward(self, data, online_k, target_k, update=0, finetune=False):
        zs = []
        views = [v_fn(data).to(self.device) for v_fn in self.views_fn]
        z = self.encode_view(views[0], online_k, finetune=finetune)
        z = self.proj_head_g(z)
        zs.append(z)

        if update:
            self.get_target_encoder()

        if self.stop_grad:
            z = self.encode_view(views[1], target_k, prune=True, finetune=finetune).detach()
        else:
            z = self.encode_view(views[1], target_k, prune=True, finetune=finetune)

        z = self.proj_head_g(z)
        zs.append(z)
        return zs

    def compute_pred_and_logits(self, data, sampler):
        xx = self.encode_view(data.to(self.device))
        data_v1, data_v2, actions, logits = sampler.sample(data, xx, return_logits=True)
        z1 = self.encode_view(data_v1.to(self.device))
        z1 = self.proj_head_g(z1)
        z2 = self.encode_view(data_v2.to(self.device))
        z2 = self.proj_head_g(z2)
        loss = self.compute_reward(z1, z2, tau=self.tau, reduce=False)
        return loss, actions, logits

    def compute_loss(self, zs_n):
        loss = self.NT_Xent(zs_n[0], zs_n[1], tau=self.tau)
        return loss

    def NT_Xent(self, z1, z2, tau=0.5, norm=True, reduce=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        if reduce:
            loss = - torch.log(loss).mean()
        return loss

    def compute_reward(self, z1, z2, tau=0.5, norm=True, reduce=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # pos_sim = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # if reduce:
        #     loss = - torch.log(loss).mean()
        return pos_sim


class MGCLPrune(torch.nn.Module):
    r"""A wrapped :class:`torch.nn.Module` class for the convinient instantiation of
    pre-implemented graph encoders.

    Args:
        feat_dim (int): The dimension of input node features.
        hidden_dim (int): The dimension of node-level (local) embeddings.
        n_layer (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
        pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
            (default: :obj:`sum`)
        gnn (string, optional): The type of GNN layer, :obj:`gcn`, :obj:`gin` or
            :obj:`resgcn`. (default: :obj:`gin`)
        bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
        act (string, optional): The activation function, :obj:`relu` or :obj:`prelu`.
            (default: :obj:`relu`)
        bias (bool, optional): Whether to include bias term in Linear. (default: :obj:`True`)
        xavier (bool, optional): Whether to apply xavier initialization. (default: :obj:`True`)
        node_level (bool, optional): If :obj:`True`, the encoder will output node level
            embedding (local representations). (default: :obj:`False`)
        graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
            embeddings (global representations). (default: :obj:`True`)
        edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
            compute the aggregation. (default: :obj:`False`)

    Note
    ----
    For GCN and GIN encoders, the dimension of the output node-level (local) embedding will be
    :obj:`hidden_dim`, whereas the node-level embedding will be :obj:`hidden_dim` * :obj:`n_layers`.
    For ResGCN, the output embeddings for boths node and graphs will have dimensions :obj:`hidden_dim`.

    Examples
    """

    def __init__(self, feat_dim, hidden_dim, n_layers=5, pool='sum', aug1='random2', aug2='random2',
                 gnn='gin', bn=True, act='relu', bias=True, proj_head='MLP', xavier=True, tau=0.5, device=None,
                 object_loss='NCE', node_level=False, stop_grad=1, aug_type='prune', prune_rate=0.3, eta=0.1, graph_level=True, edge_weight=False):
        super(MGCLPrune, self).__init__()

        self.tau = tau
        self.proj_head = proj_head
        self.hidden_dim = hidden_dim
        self.proj_out_dim = hidden_dim
        self.object_loss = object_loss
        self.aug_type = aug_type
        self.prune_rate = prune_rate
        self.eta = eta
        self.stop_grad = stop_grad
        self.n_layers = n_layers

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        if gnn == 'gin':
            self.encoder = GINNew(feat_dim, hidden_dim, n_layers, pool, bn, act, aug_type=aug_type)
        elif gnn == 'gcn':
            self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'gcn_raw':
            self.encoder = GCNRaw(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'resgcn':
            self.encoder = ResGCNNew(feat_dim, hidden_dim, num_conv_layers=n_layers,
                                  global_pool=pool, aug_type=aug_type)
        # self.views_fn = self.aug_funs()
        self.gnn_type = gnn
        self.node_level = node_level
        self.graph_level = graph_level
        if gnn == "gin" or gnn == "gcn":
            self.d_out = self.hidden_dim * n_layers
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim * n_layers)
        else:
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim)
            self.d_out = self.hidden_dim
        self.views_fn = self.view_fns(aug1, aug2)

    def view_fns(self, aug_1, aug_2, aug_ratio=0.2):
        views_fn = []

        for aug in [aug_1, aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'dropN':
                views_fn.append(UniformSample(ratio=aug_ratio))
            elif aug == 'permE':
                views_fn.append(EdgePerturbation(ratio=aug_ratio))
            elif aug == 'subgraph':
                views_fn.append(RWSample(ratio=aug_ratio))
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(mask_ratio=aug_ratio))
            elif aug == 'random2':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random4':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random3':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio),
                              NodeAttrMask(mask_ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            else:
                raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                        'maskN', 'random2', 'random3', 'random4'] or None.")
        return views_fn

    def _get_proj(self, proj_head, in_dim):

        if callable(proj_head):
            return proj_head

        assert proj_head in ['linear', 'MLP']

        # out_dim = self.proj_out_dim
        out_dim = self.proj_out_dim

        if proj_head == 'linear':
            proj_nn = nn.Linear(in_dim, out_dim)
            self._weights_init(proj_nn)
        elif proj_head == 'MLP':
            proj_nn = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_dim, out_dim))
            for m in proj_nn.modules():
                self._weights_init(m)

        return proj_nn

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def encode_view(self, data, layer_K, prune: bool = False, finetune=False):
        z_g = self.encoder(data, layer_K, prune, finetune=finetune)
        return z_g

    def get_target_encoder(self):
        if self.gnn_type == 'resgcn':
            if self.aug_type == 'noise':
                for i in range(self.n_layers):
                    weight_, _, bias_, _ = self.encoder.convs[i].get_weight()
                    bias_ = torch.from_numpy(bias_)
                    weight_ = torch.from_numpy(weight_)
                    weight_1_noise = self.eta * torch.normal(0, torch.ones_like(weight_.data) * weight_.data.std()).data
                    bias_1_noise = self.eta * torch.normal(0, torch.ones_like(bias_.data) * bias_.data.std()).data
                    self.encoder.convs[i].update_weight2noise(weight_1_noise, bias_1_noise)
            else:
                weight_all = []
                weight_all_mask = []
                for i in range(self.n_layers):
                    weight_1, weight1_mask, bias_1, bias_1_mask = self.encoder.convs[i].get_weight()
                    weight_all.extend([weight_1, bias_1])
                    weight_all_mask.extend([weight1_mask, bias_1_mask])
                weight_vector = np.concatenate([v.flatten() for v in weight_all])

                # number_of_remaining_weights = np.sum([np.sum(v) for v in weight_all_mask])
                number_of_remaining_weights = weight_vector.shape[0]

                number_of_weights_to_prune_magnitude = np.ceil(
                    self.prune_rate * number_of_remaining_weights).astype(
                    int)
                threshold = np.sort(np.abs(weight_vector))[
                    min(number_of_weights_to_prune_magnitude,
                        len(weight_vector) - 1)]
                for i in range(self.n_layers):
                    self.encoder.convs[i].update_weight2prune(threshold)
        elif self.gnn_type == 'gin':
            if self.aug_type == 'noise':
                for i in range(self.n_layers):
                    weight_1, bias_1, weight_2, bias_2 = self.encoder.convs[i].get_weight()
                    weight_1 = torch.from_numpy(weight_1)
                    bias_1 = torch.from_numpy(bias_1)
                    weight_2 = torch.from_numpy(weight_2)
                    bias_2 = torch.from_numpy(bias_2)
                    weight_1_noise = self.eta * torch.normal(0, torch.ones_like(weight_1.data) * weight_1.data.std())
                    bias_1_noise = self.eta * torch.normal(0, torch.ones_like(bias_1.data) * bias_1.data.std())
                    weight_2_noise = self.eta * torch.normal(0, torch.ones_like(weight_2.data) * weight_2.data.std())
                    bias_2_noise = self.eta * torch.normal(0, torch.ones_like(bias_2.data) * bias_2.data.std())
                    self.encoder.convs[i].update_weight2noise([weight_1_noise, weight_2_noise], [bias_1_noise, bias_2_noise])
            else:
                weight_all = []
                for i in range(self.n_layers):
                    weight_1, bias_1, weight_2, bias_2 = self.encoder.convs[i].get_weight()
                    weight_all.extend([weight_1, bias_1, weight_2, bias_2])
                weight_vector = np.concatenate([v.flatten() for v in weight_all])

                # number_of_remaining_weights = np.sum([np.sum(v) for v in weight_all_mask])
                number_of_remaining_weights = weight_vector.shape[0]

                number_of_weights_to_prune_magnitude = np.ceil(
                    self.prune_rate * number_of_remaining_weights).astype(
                    int)
                threshold = np.sort(np.abs(weight_vector))[
                    min(number_of_weights_to_prune_magnitude,
                        len(weight_vector) - 1)]
                for i in range(self.n_layers):
                    self.encoder.convs[i].update_weight2prune(threshold)
        else:
            raise ValueError('the gnn type: {} is not supported'.format(self.gnn_type))

    def forward(self, data, online_k, target_k, update=0, finetune=False):
        zs = []
        # views = [v_fn(data).to(self.device) for v_fn in self.views_fn]

        z = self.encode_view(data.to(self.device), online_k, finetune=finetune)
        z = self.proj_head_g(z)
        zs.append(z)

        if update:
            self.get_target_encoder()

        if self.stop_grad:
            z = self.encode_view(data, target_k, prune=True, finetune=finetune).detach()
        else:
            z = self.encode_view(data, target_k, prune=True, finetune=finetune)

        z = self.proj_head_g(z)
        zs.append(z)
        return zs

    def compute_pred_and_logits(self, data, sampler):
        xx = self.encode_view(data.to(self.device))
        data_v1, data_v2, actions, logits = sampler.sample(data, xx, return_logits=True)
        z1 = self.encode_view(data_v1.to(self.device))
        z1 = self.proj_head_g(z1)
        z2 = self.encode_view(data_v2.to(self.device))
        z2 = self.proj_head_g(z2)
        loss = self.compute_reward(z1, z2, tau=self.tau, reduce=False)
        return loss, actions, logits

    def compute_loss(self, zs_n):
        loss = self.NT_Xent(zs_n[0], zs_n[1], tau=self.tau)
        return loss

    def NT_Xent(self, z1, z2, tau=0.5, norm=True, reduce=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        if reduce:
            loss = - torch.log(loss).mean()
        return loss

    def compute_reward(self, z1, z2, tau=0.5, norm=True, reduce=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # pos_sim = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # if reduce:
        #     loss = - torch.log(loss).mean()
        return pos_sim


class GIN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False,
                 act='relu', bias=True, xavier=True):
        super(GIN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        self.act = torch.nn.PReLU() if act == 'prelu' else torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            nn = Sequential(Linear(start_dim, hidden_dim, bias=bias),
                            self.act,
                            Linear(hidden_dim, hidden_dim, bias=bias))
            if xavier:
                self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep
        # return global_rep, x


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False,
                 act='relu', bias=True, xavier=True, edge_weight=False):
        super(GCN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool
        self.edge_weight = edge_weight
        self.normalize = not edge_weight
        self.add_self_loops = not edge_weight

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GCNConv(start_dim, hidden_dim, bias=bias,
                           add_self_loops=self.add_self_loops,
                           normalize=self.normalize)
            if xavier:
                self.weights_init(conv)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GCNConv):
            try:
                torch.nn.init.xavier_uniform_(m.weight.data)
            except AttributeError:
                torch.nn.init.xavier_uniform_(m.lin.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_weight:
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.acts[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep
        # return global_rep, x


class GCNRaw(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False,
                 act='relu', bias=True, xavier=True, edge_weight=False):
        super(GCNRaw, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool
        self.edge_weight = edge_weight
        self.normalize = not edge_weight
        self.add_self_loops = not edge_weight

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GCNConv(start_dim, hidden_dim, bias=bias,
                           add_self_loops=self.add_self_loops,
                           normalize=self.normalize)
            if xavier:
                self.weights_init(conv)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GCNConv):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_weight:
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.acts[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        # global_rep = torch.cat(xpool, 1)
        global_rep = xpool[-1]

        return global_rep


class ResGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 improved=False, cached=False, bias=True, edge_norm=True, gfn=False):
        super(ResGCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.weights_init()

    def weights_init(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes,),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        row, col = edge_index

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = ResGCNConv.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class ResGCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim,
                 num_feat_layers=1,
                 num_conv_layers=3,
                 num_fc_layers=2, xg_dim=None, bn=True,
                 gfn=False, collapse=False, residual=False,
                 global_pool="sum", dropout=0, edge_norm=True):

        super(ResGCN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.collapse = collapse
        self.bn = bn

        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        GConv = partial(ResGCNConv, edge_norm=edge_norm, gfn=gfn)

        if xg_dim is not None:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(xg_dim)
            self.lin1_xg = Linear(xg_dim, hidden_dim)
            self.bn2_xg = BatchNorm1d(hidden_dim)
            self.lin2_xg = Linear(hidden_dim, hidden_dim)
        else:
            self.use_xg = False

        if collapse:
            self.bn_feat = BatchNorm1d(feat_dim)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(feat_dim, feat_dim),
                    torch.nn.ReLU(),
                    Linear(feat_dim, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden_dim))
                hidden_in = hidden_dim
        else:
            self.bn_feat = BatchNorm1d(feat_dim)
            feat_gfn = True  # set true so GCNConv is feat transform
            self.conv_feat = ResGCNConv(feat_dim, hidden_dim, gfn=feat_gfn)
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    Linear(hidden_dim, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden_dim))
                self.convs.append(GConv(hidden_dim, hidden_dim))
            self.bn_hidden = BatchNorm1d(hidden_dim)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_dim))
                self.lins.append(Linear(hidden_dim, hidden_dim))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is of shape [n_graphs, feat_dim]
            xg = self.bn1_xg(data.xg) if self.bn else xg
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg) if self.bn else xg
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x) if self.bn else x
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x) if self.bn else x
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        local_rep = x
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg

        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x) if self.bn else x
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x