import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_scatter import scatter_add
import torch
from torch import Tensor
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul


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


class ResGCNConvPrune(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 improved=False, cached=False, bias=True, skip: int = 1, aug_type='prune', edge_norm=True, gfn=False):
        super(ResGCNConvPrune, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.add_skip = skip

        self.gfn = gfn
        self.aug_type = aug_type
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_mask = Parameter(torch.ones_like(self.weight),
                                     requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias_mask = Parameter(torch.ones_like(self.bias), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.weights_init()

    def weights_init(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    def get_weight(self):
        return self.weight.clone().cpu().detach().numpy(), self.weight_mask.clone().cpu().detach().numpy(),\
               self.bias.clone().cpu().detach().numpy(), self.bias_mask.clone().cpu().detach().numpy()

    def update_weight2prune(self, threshold):
        self.weight_mask = Parameter((torch.abs(self.weight) >= threshold).float(), requires_grad=False).to(self.weight.device)
        self.bias_mask = Parameter((torch.abs(self.bias) >= threshold).float(), requires_grad=False).to(self.weight.device)

    def update_weight2noise(self, weight1_noise, bias1_noise):
        # self.weight_mask = Parameter(weight1_noise, requires_grad=False).to(self.weight.device)
        # self.bias_mask = Parameter(bias1_noise, requires_grad=False).to(self.weight.device)
        self.weight_mask = Parameter(torch.ones_like(self.weight), requires_grad=False).to(self.weight.device)
        self.bias_mask = Parameter(torch.ones_like(self.bias), requires_grad=False).to(self.weight.device)
        self.weight_mask.weight = weight1_noise.to(self.weight.device)
        self.bias_mask.weight = bias1_noise.to(self.weight.device)

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

    def forward(self, x, edge_index, k, edge_weight=None, prune=False):
        if prune:
            if self.aug_type == 'prune':
                x = torch.matmul(x, self.weight * self.weight_mask)
            else:
                x = torch.matmul(x, self.weight + self.weight_mask)
        else:
            x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = ResGCNConv.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        if k == 0:
            out = self.propagate(edge_index, x=x, norm=norm)
        else:
            for i in range(k):
                if self.add_skip:
                    x = (x + self.propagate(edge_index, x=x, norm=norm)) * 0.5
                else:
                    x = self.propagate(edge_index, x=x, norm=norm)
            out = x

        if self.bias is not None:
            if prune:
                if self.aug_type == 'prune':
                    out = out * (self.bias * self.bias_mask)
                else:
                    out = out + (self.bias + self.bias_mask)
            else:
                out += self.bias

        return out

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    # def update(self, aggr_out):
    #     if self.bias is not None:
    #         aggr_out = aggr_out + self.bias
    #     return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GINNew(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False,
                 act='relu', bias=True, xavier=True, aug_type='prune'):
        super(GINNew, self).__init__()

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
            # nn = Sequential(Linear(start_dim, hidden_dim, bias=bias),
            #                 self.act,
            #                 Linear(hidden_dim, hidden_dim, bias=bias))
            # if xavier:
            #     self.weights_init(nn)
            conv = GINConvPrune(start_dim, hidden_dim, self.act)
            # conv = GINConv(nn)
            self.convs.append(conv)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def generate_step(self, K, finetune=False):
        if self.training and finetune == False:
            k = np.random.randint(0, K)
        else:
            k = K
        return k

    def forward(self, data, layer_K, prune: bool = False, finetune=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, k=self.generate_step(layer_K, finetune), prune=prune)
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


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class LinearPrune(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, aug_type='prune', bias: bool = True) -> None:
        super(LinearPrune, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aug_type = aug_type
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_mask = Parameter(torch.ones_like(self.weight), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias_mask = Parameter(torch.ones_like(self.bias), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def getweight(self):
        return self.weight.clone().cpu().detach().numpy(), self.bias.clone().cpu().detach().numpy()

    def update_prune(self, threshold):
        self.weight_mask = Parameter((torch.abs(self.weight) >= threshold).float(), requires_grad=False).to(self.weight.device)
        self.bias_mask = Parameter((torch.abs(self.bias) >= threshold).float(), requires_grad=False).to(self.weight.device)

    def update_noise(self, weight1_noise, bias1_noise):
        # self.weight_mask = Parameter(weight1_noise, requires_grad=False).to(self.weight.device)
        # self.bias_mask = Parameter(bias1_noise, requires_grad=False).to(self.weight.device)
        self.weight_mask = Parameter(torch.ones_like(self.weight), requires_grad=False).to(self.weight.device)
        self.bias_mask = Parameter(torch.ones_like(self.bias), requires_grad=False).to(self.weight.device)
        self.weight_mask.weight = weight1_noise.to(self.weight.device)
        self.bias_mask.weight = bias1_noise.to(self.weight.device)

    def forward(self, input: Tensor, prune=False) -> Tensor:
        if prune:
            if self.aug_type == 'prune':
                x = F.linear(input, self.weight * self.weight_mask, self.bias * self.bias_mask)
            else:
                x = F.linear(input, self.weight + self.weight_mask, self.bias + self.bias_mask)
        else:
            x = F.linear(input, self.weight, self.bias)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class GINConvPrune(MessagePassing):
    def __init__(self, input_dim, hidden_dim, act, bias=True, eps: float = 0., skip: int = 1, train_eps: bool = False, aug_type='prune',
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConvPrune, self).__init__(**kwargs)
        self.nn1 = LinearPrune(input_dim, hidden_dim, bias=bias, aug_type=aug_type)
        # activations = {
        #     'relu': F.relu,
        #     'hardtanh': F.hardtanh,
        #     'elu': F.elu,
        #     'leakyrelu': F.leaky_relu,
        #     'prelu': torch.nn.PReLU(),
        #     'rrelu': F.rrelu
        # }
        self.act = act
        self.nn2 = LinearPrune(hidden_dim, hidden_dim, bias=bias, aug_type=aug_type)
        self.aug_type = aug_type
        self.add_skip = skip

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def reset_parameters(self):
        # reset(self.nn)
        self.nn1.reset_parameters()
        self.nn2.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def get_weight(self):
        weight_1, bias_1 = self.nn1.getweight()
        weight_2, bias_2 = self.nn2.getweight()
        return weight_1, bias_1, weight_2, bias_2

    def update_weight2prune(self, threshold):
        self.nn1.update_prune(threshold)
        self.nn2.update_prune(threshold)

    def update_weight2noise(self, weight1_noise, bias1_noise):
        self.nn1.update_noise(weight1_noise[0], bias1_noise[0])
        self.nn2.update_noise(weight1_noise[1], bias1_noise[1])

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, k, prune=False,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if k == 0:
            out = self.propagate(edge_index, x=x, size=size)
        else:
            out = x[0]
            for i in range(k):
                if self.add_skip:
                    out = (out + self.propagate(edge_index, x=(out, out), size=size)) * 0.5
                else:
                    out = self.propagate(edge_index, x=(out, out), size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        out = self.nn1(out, prune)
        out = self.act(out)
        out = self.nn2(out, prune)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class ResGCNNew(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim,
                 num_feat_layers=1,
                 num_conv_layers=3,
                 num_fc_layers=2, xg_dim=None, bn=True,
                 gfn=False, collapse=False, residual=False,
                 global_pool="sum", dropout=0, edge_norm=True, aug_type='prune'):

        super(ResGCNNew, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.collapse = collapse
        self.bn = bn
        self.aug_type = aug_type

        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        # GConv = partial(ResGCNConvPrune, edge_norm=edge_norm, gfn=gfn)

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
                # self.convs.append(GConv(hidden_dim, hidden_dim))
                self.convs.append(ResGCNConvPrune(hidden_dim, hidden_dim, aug_type=self.aug_type, edge_norm=edge_norm, gfn=False))
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

    def generate_step(self, K, finetune=False):
        if self.training and finetune == False:
            k = np.random.randint(0, K)
        else:
            k = K
        return k

    def forward(self, data, layer_K, prune: bool = False, finetune=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xg = None

        x = self.bn_feat(x) if self.bn else x
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x) if self.bn else x
            x_ = F.relu(conv(x_, edge_index, k=self.generate_step(layer_K, finetune), prune=prune))
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