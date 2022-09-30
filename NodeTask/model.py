import torch
from copy import deepcopy
from layers import GCNLayer, GATLayer, GCNLayerPrune
import numpy as np
import torch.nn.functional as F
from torch import nn
from gssl.augment import GraphAugmentor


class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_type='GCN',
                 dropout=0.5, fix_k=2):
        super().__init__()

        if gnn_type == 'GCN':
            self._conv1 = GCNLayer(in_channels, hidden_channels)
            self._conv2 = GCNLayer(hidden_channels, out_channels)
        else:
            self._conv1 = GATLayer(in_channels, hidden_channels)
            self._conv2 = GATLayer(hidden_channels, out_channels)

        self._act1 = nn.ReLU()
        # self._act2 = nn.ReLU()
        self.fix_k = fix_k
        self.dropout = dropout

    def generate_step(self, K):
        if self.training:
            k = np.random.randint(0, K)
        else:
            k = self.fix_k
        return k

    def forward(self, x, edge_index, layer_K):
        x = self._conv1(x, edge_index, k=self.generate_step(layer_K))
        # x = self._bn1(x)
        x = self._act1(x)
        x = self._conv2(x, edge_index, k=self.generate_step(layer_K))
        return x


class GNNEncoderPrune(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, aug_type='prune', gnn_type='GCN',
                 dropout=0.5, fix_k=2, act='relu'):
        super().__init__()

        if gnn_type == 'GCN':
            self.convs = torch.nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNLayerPrune(in_channels, hidden_channels, aug_type))
                else:
                    self.convs.append(GCNLayerPrune(hidden_channels, out_channels, aug_type))
        else:
            self.conv1 = GATLayer(in_channels, hidden_channels)
            self.conv2 = GATLayer(hidden_channels, out_channels)

        activations = {
            'relu': F.relu,
            'hardtanh': F.hardtanh,
            'elu': F.elu,
            'leakyrelu': F.leaky_relu,
            'prelu': torch.nn.PReLU(),
            'rrelu': F.rrelu
        }
        self.num_layer = num_layers
        self.act1 = activations[act]
        self.fix_k = fix_k
        self.dropout = dropout

    def generate_step(self, K):
        if self.training:
            k = np.random.randint(0, K)
        else:
            k = K
        return k

    def forward(self, x, edge_index, layer_K, prune: bool = False):
        for _, conv in enumerate(self.convs[0:-1]):
            x = conv(x, edge_index, k=self.generate_step(layer_K), prune=prune)
            x = self.act1(x)
        x = self.convs[-1](x, edge_index, k=self.generate_step(layer_K), prune=prune)

        # x = self.conv1(x, edge_index, k=self.generate_step(layer_K), prune=prune)
        # # x = self._bn1(x)
        # x = self.act1(x)
        # x = self.conv2(x, edge_index, k=self.generate_step(layer_K), prune=prune)
        return x


class MGCL(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        aug_type: str,
        out_dim: int,
        pred_dim: int,
        eta: float = 0.1,
        loss_type: str = 'cl',
    ):
        super().__init__()
        self._augmentor = augmentor
        self.aug_type = aug_type
        self.eta = eta
        self.loss_type = loss_type

        self._online_encoder = encoder
        self._target_encoder = deepcopy(self._online_encoder)
        # self.get_target_encoder()

        self._predictor = torch.nn.Sequential(
            nn.Linear(out_dim, pred_dim),
            # nn.BatchNorm1d(pred_dim, momentum=0.01),
            nn.ELU(),
            nn.Linear(pred_dim, out_dim),
            # nn.BatchNorm1d(out_dim, momentum=0.01),
            # nn.ELU(),
        )

    def get_target_encoder(self):
        self._target_encoder = deepcopy(self._online_encoder)
        if self.aug_type == 'Noise':
            for p in self._target_encoder.parameters():
                p.data = p.data + self.eta * torch.normal(0, torch.ones_like(p.data) * p.data.std())
                p.requires_grad = False
        else:
            pass
        #     implement lottery ticket

        return self._target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(
            self.get_target_encoder().parameters(),
            self._online_encoder.parameters(),
        ):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, data, online_k, target_k):
        # (x1, edge_index1), (x2, edge_index2) = self._augmentor(data=data)
        h1 = self._online_encoder(data.x, data.edge_index, online_k)
        h1_pred = self._predictor(h1)
        with torch.no_grad():
            h2 = self.get_target_encoder()(data.x, data.edge_index, target_k)
        # h2_pred = self._predictor(h2)
        return h1, h2, h1_pred

    def compute_loss(self, h1, h2):
        if self.loss_type == 'cl':
            loss = loss_cl(h1, h2)
        elif self.loss_type == 'br':
            loss = bt_loss(h1, h2)
        else:
            raise ValueError('the loss funs {} should be cl or bt'.format(self.loss_type))
        return loss

    def predict(self, data, final_k=2):
        with torch.no_grad():
            h1 = self._online_encoder(data.x, data.edge_index, final_k).cpu()
            h2 = self.get_target_encoder()(data.x, data.edge_index, final_k).cpu()
        return h1, h2


class MGCLPrune(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        stop_grad: int,
        out_dim: int,
        pred_dim: int,
        sym_loss: int = 0,
        use_pred: int = 1,
        eta: float = 0.1,
        loss_type: str = 'cl',
        aug_type: str = 'prune',
        prune_rate: float = 0.3,
    ):
        super().__init__()
        self._augmentor = augmentor
        self.stop_grad = stop_grad
        self.eta = eta
        self.prune_rate = prune_rate
        self.loss_type = loss_type
        self.sym_loss = sym_loss
        self.use_pred = use_pred
        self.aug_type = aug_type

        self._online_encoder = encoder

        if self.use_pred:
            self._predictor = torch.nn.Sequential(
                nn.Linear(out_dim, pred_dim),
                nn.ELU(),
                nn.Linear(pred_dim, out_dim),
            )

    def get_target_encoder(self):
        weight_1, weight1_mask, bias_1, bias_1_mask = self._online_encoder.conv1.get_weight()
        weight_2, weight2_mask, bias_2, bias_2_mask = self._online_encoder.conv2.get_weight()

        if self.aug_type == 'noise':
            weight_1 = torch.from_numpy(weight_1)
            weight_2 = torch.from_numpy(weight_2)
            bias_1 = torch.from_numpy(bias_1)
            bias_2 = torch.from_numpy(bias_2)
            weight_1_noise = self.eta * torch.normal(0, torch.ones_like(weight_1.data) * weight_1.data.std())
            weight_2_noise = self.eta * torch.normal(0, torch.ones_like(weight_2.data) * weight_2.data.std())
            bias_1_noise = self.eta * torch.normal(0, torch.ones_like(bias_1.data) * bias_1.data.std())
            bias_2_noise = self.eta * torch.normal(0, torch.ones_like(bias_2.data) * bias_2.data.std())
            self._online_encoder.conv1.update_weight2noise(weight_1_noise, bias_1_noise)
            self._online_encoder.conv2.update_weight2noise(weight_2_noise, bias_2_noise)
        else:
            weight_vector = np.concatenate([v.flatten() for v in [weight_1, weight_2, bias_1, bias_2]])

            number_of_remaining_weights = np.sum([np.sum(v) for v in [weight1_mask, weight2_mask, bias_1_mask, bias_2_mask]])

            number_of_weights_to_prune_magnitude = np.ceil(
                self.prune_rate * number_of_remaining_weights).astype(
                int)
            threshold = np.sort(np.abs(weight_vector))[
                min(number_of_weights_to_prune_magnitude,
                    len(weight_vector) - 1)]
            self._online_encoder.conv1.update_weight2prune(threshold)
            self._online_encoder.conv2.update_weight2prune(threshold)
        #     implement lottery ticket

    def get_weight(self):
        weight_1, weight1_mask, bias_1, bias_1_mask = self._online_encoder.conv1.get_weight()
        weight_2, weight2_mask, bias_2, bias_2_mask = self._online_encoder.conv2.get_weight()
        return weight_1, weight1_mask, weight_2, weight2_mask

    def forward(self, data, online_k, target_k, update=0):
        # (x1, edge_index1), (x2, edge_index2) = self._augmentor(data=data)
        h1 = self._online_encoder(data.x, data.edge_index, online_k)
        if update:
            self.get_target_encoder()

        if self.stop_grad:
            h2 = self._online_encoder(data.x, data.edge_index, target_k, prune=True).detach()
        else:
            h2 = self._online_encoder(data.x, data.edge_index, target_k, prune=True)

        if self.use_pred:
            h1_pred = self._predictor(h1)
            if self.sym_loss:
                h2_pred = self._predictor(h2)
                return h1, h2, h1_pred, h2_pred
            else:
                return h1, h2, h1_pred
        else:
            return h1, h2

    def testforward(self, data, target_k):
        z = self._online_encoder(data.x, data.edge_index, [target_k, target_k])
        return z

    def testforward_metric(self, x, edge_index, target_k=1):
        z = self._online_encoder(x, edge_index, [target_k, target_k])
        z_pos = self._online_encoder(x, edge_index, [target_k, target_k], prune=True)
        return z, z_pos

    def compute_loss(self, h1, h2):
        if self.loss_type == 'cl':
            loss = loss_cl(h1, h2)
        elif self.loss_type == 'cl2':
            loss = loss_cl2(h1, h2)

        elif self.loss_type == 'br':
            loss = bt_loss(h1, h2)
        else:
            raise ValueError('the loss funs {} should be cl or bt'.format(self.loss_type))
        return loss

    def compute_loss_sym(self, h1, h2, h1_pred, h2_pred):
        if self.loss_type == 'cl':
            loss1 = loss_cl(h1_pred, h2)
            loss2 = loss_cl(h2_pred, h1)
            loss = (loss1 + loss2) * 0.5

        elif self.loss_type == 'cl2':
            loss1 = loss_cl2(h1_pred, h2)
            loss2 = loss_cl2(h2_pred, h1)
            loss = (loss1 + loss2) * 0.5

        elif self.loss_type == 'br':
            loss1 = bt_loss(h1_pred, h2)
            loss2 = bt_loss(h2_pred, h1)
            loss = (loss1 + loss2) * 0.5
        else:
            raise ValueError(
                'the loss funs {} should be cl or bt'.format(self.loss_type))
        return loss

    def predict(self, data, final_k=1):
        with torch.no_grad():
            h1 = self._online_encoder(data.x, data.edge_index, final_k).cpu()
            h2 = self._online_encoder(data.x, data.edge_index, final_k, prune=True).cpu()
        return h1, h2


class MGCLPruneAug(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        stop_grad: int,
        out_dim: int,
        pred_dim: int,
        sym_loss: int = 0,
        use_pred: int = 1,
        eta: float = 0.1,
        loss_type: str = 'cl',
        aug_type: str = 'prune',
        prune_rate: float = 0.3,
    ):
        super().__init__()
        self._augmentor = augmentor
        self.stop_grad = stop_grad
        self.eta = eta
        self.prune_rate = prune_rate
        self.loss_type = loss_type
        self.sym_loss = sym_loss
        self.use_pred = use_pred
        self.aug_type = aug_type

        self._online_encoder = encoder

        if self.use_pred:
            self._predictor = torch.nn.Sequential(
                nn.Linear(out_dim, pred_dim),
                nn.ELU(),
                nn.Linear(pred_dim, out_dim),
            )

    def get_target_encoder(self):

        if self.aug_type == 'noise':
            for i in range(self._online_encoder.num_layer):
                weight_, _, bias_, _ = self._online_encoder.convs[i].get_weight()
                bias_ = torch.from_numpy(bias_)
                weight_ = torch.from_numpy(weight_)
                weight_1_noise = self.eta * torch.normal(0, torch.ones_like(weight_.data) * weight_.data.std())
                # print('weight type is: {}'.format(type(weight_1_noise)))
                bias_1_noise = self.eta * torch.normal(0, torch.ones_like(bias_.data) * bias_.data.std()).data
                self._online_encoder.convs[i].update_weight2noise(weight_1_noise, bias_1_noise)
        else:
            weight_all = []
            weight_all_mask = []
            for i in range(self._online_encoder.num_layer):
                weight_1, weight1_mask, bias_1, bias_1_mask = self._online_encoder.convs[i].get_weight()
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
            for i in range(self._online_encoder.num_layer):
                self._online_encoder.convs[i].update_weight2prune(threshold)

        # weight_1, weight1_mask, bias_1, bias_1_mask = self._online_encoder.conv1.get_weight()
        # weight_2, weight2_mask, bias_2, bias_2_mask = self._online_encoder.conv2.get_weight()
        #
        # for i in range(self._online_encoder.num_layer):
        #
        #
        # if self.aug_type == 'noise':
        #     weight_1 = torch.from_numpy(weight_1)
        #     weight_2 = torch.from_numpy(weight_2)
        #     bias_1 = torch.from_numpy(bias_1)
        #     bias_2 = torch.from_numpy(bias_2)
        #     weight_1_noise = self.eta * torch.normal(0, torch.ones_like(weight_1.data) * weight_1.data.std())
        #     weight_2_noise = self.eta * torch.normal(0, torch.ones_like(weight_2.data) * weight_2.data.std())
        #     bias_1_noise = self.eta * torch.normal(0, torch.ones_like(bias_1.data) * bias_1.data.std())
        #     bias_2_noise = self.eta * torch.normal(0, torch.ones_like(bias_2.data) * bias_2.data.std())
        #     self._online_encoder.conv1.update_weight2noise(weight_1_noise, bias_1_noise)
        #     self._online_encoder.conv2.update_weight2noise(weight_2_noise, bias_2_noise)
        # else:
        #     weight_vector = np.concatenate([v.flatten() for v in [weight_1, weight_2, bias_1, bias_2]])
        #
        #     # number_of_remaining_weights = np.sum([np.sum(v) for v in [weight1_mask, weight2_mask, bias_1_mask, bias_2_mask]])
        #     number_of_remaining_weights = weight_vector.shape[0]
        #
        #     number_of_weights_to_prune_magnitude = np.ceil(
        #         self.prune_rate * number_of_remaining_weights).astype(
        #         int)
        #     threshold = np.sort(np.abs(weight_vector))[
        #         min(number_of_weights_to_prune_magnitude,
        #             len(weight_vector) - 1)]
        #     self._online_encoder.conv1.update_weight2prune(threshold)
        #     self._online_encoder.conv2.update_weight2prune(threshold)
        #     implement lottery ticket
    def get_weight(self):
        weight_1, weight1_mask, bias_1, bias_1_mask = self._online_encoder.convs[0].get_weight()
        weight_2, weight2_mask, bias_2, bias_2_mask = self._online_encoder.convs[1].get_weight()
        return weight_1, weight1_mask, weight_2, weight2_mask

    def forward(self, data, online_k, target_k, update=0):
        (x1, edge_index1), (x2, edge_index2) = self._augmentor(data=data)
        h1 = self._online_encoder(x1, edge_index1, online_k)
        if update:
            self.get_target_encoder()

        if self.stop_grad:
            h2 = self._online_encoder(x2, edge_index2, target_k, prune=True).detach()
        else:
            h2 = self._online_encoder(x2, edge_index2, target_k, prune=True)

        if self.use_pred:
            h1_pred = self._predictor(h1)
            if self.sym_loss:
                h2_pred = self._predictor(h2)
                return h1, h2, h1_pred, h2_pred
            else:
                return h1, h2, h1_pred
        else:
            return h1, h2

    def testforward(self, data, target_k):
        z = self._online_encoder(data.x, data.edge_index, target_k)
        return z

    def testforward_metric(self, x, edge_index, target_k=1):
        z = self._online_encoder(x, edge_index, target_k)
        z_pos = self._online_encoder(x, edge_index, target_k, prune=True)
        return z, z_pos

    def compute_loss(self, h1, h2):
        if self.loss_type == 'cl':
            loss = loss_cl(h1, h2)
        elif self.loss_type == 'cl2':
            loss = loss_cl2(h1, h2)

        elif self.loss_type == 'br':
            loss = bt_loss(h1, h2)
        else:
            raise ValueError('the loss funs {} should be cl or bt'.format(self.loss_type))
        return loss

    def compute_loss_sym(self, h1, h2, h1_pred, h2_pred):
        if self.loss_type == 'cl':
            loss1 = loss_cl(h1_pred, h2)
            loss2 = loss_cl(h2_pred, h1)
            loss = (loss1 + loss2) * 0.5

        elif self.loss_type == 'cl2':
            loss1 = loss_cl2(h1_pred, h2)
            loss2 = loss_cl2(h2_pred, h1)
            loss = (loss1 + loss2) * 0.5

        elif self.loss_type == 'br':
            loss1 = bt_loss(h1_pred, h2)
            loss2 = bt_loss(h2_pred, h1)
            loss = (loss1 + loss2) * 0.5
        else:
            raise ValueError(
                'the loss funs {} should be cl or bt'.format(self.loss_type))
        return loss

    def predict(self, data, final_k=1):
        with torch.no_grad():
            h1 = self._online_encoder(data.x, data.edge_index, final_k).cpu()
            h2 = self._online_encoder(data.x, data.edge_index, final_k, prune=True).cpu()
        return h1, h2


def loss_cl(x1, x2, eps=1e-15):
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', x1_abs, x2_abs) + eps)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss


def loss_cl2(x1, x2, eps=1e-15):
    T = 0.4
    batch_size, _ = x1.size()
    x1_abs = F.normalize(x1, dim=-1)
    x2_abs = F.normalize(x2, dim=-1)

    sim_matrix = torch.mm(x1_abs, x2_abs.t())
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim.mean()
    return loss


def bt_loss(h1: torch.Tensor, h2: torch.Tensor, lambda_=None, batch_norm=True,
            eps=1e-15, *args, **kwargs):
    batch_size = h1.size(0)
    feature_dim = h1.size(1)

    if lambda_ is None:
        lambda_ = 1. / feature_dim

    if batch_norm:
        z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
        z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
        c = (z1_norm.T @ z2_norm) / batch_size
    else:
        c = h1.T @ h2 / batch_size

    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

    return loss
