import numpy as np
from torch_geometric.data import Batch
import torch
import torch.nn as nn
import torch.nn.functional as F
from method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, \
    UniformSample, RWSample, RandomView, RawView


def aug_init(aug, aug_ratio):
    if aug == "raw" or aug == 0:
        return RawView()
    elif aug == 'dropN' or aug == 1:
         return UniformSample(ratio=aug_ratio)
    elif aug == 'permE' or aug == 2:
        return EdgePerturbation(ratio=aug_ratio)
    elif aug == 'subgraph' or aug == 3:
        return RWSample(ratio=aug_ratio)
    elif aug == 'maskN' or aug == 4:
        return NodeAttrMask(mask_ratio=aug_ratio)
    else:
        raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                'maskN', 'raw'] ")


def aug_view(aug, aug_ratio):
    aug_ratio = (aug_ratio + 1) * 0.1
    if aug == "raw" or aug == 0:
        return RawView()
    elif aug == 'dropN' or aug == 1:
         return UniformSample(ratio=aug_ratio)
    elif aug == 'permE' or aug == 2:
        return EdgePerturbation(ratio=aug_ratio)
    elif aug == 'subgraph' or aug == 3:
        return RWSample(ratio=aug_ratio)
    elif aug == 'maskN' or aug == 4:
        return NodeAttrMask(mask_ratio=aug_ratio)
    else:
        raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                'maskN', 'raw'] ")


class BaseSampler:
    def __init__(self, device, args):
        raise NotImplementedError

    def learn(self, model, valid_index_loader, warmup=None):
        pass


class OriginalSampler(BaseSampler):
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self._init_sampler(args.aug1, args.aug2, args.aug1_ratio, args.aug2_ratio)

    def _init_sampler(self, aug1, aug2, aug1_ratio, aug2_ratio):
        self.aug_1 = aug_init(aug1, aug1_ratio)
        self.aug_2 = aug_init(aug2, aug2_ratio)

    def sample(self, data):
        data_view1 = self.aug_1(data)
        data_view2 = self.aug_2(data)

        return data_view1, data_view2


def generate_view(data, aug_views, aug_ratios):
    dlist = data.to_data_list()
    bsz = aug_views.shape[0]
    data_view1 = []
    data_view2 = []
    for i in range(bsz):
        view1, aug1_ratio = aug_views[i, 0], aug_ratios[i, 0]
        data_view1.append(aug_view(view1, aug1_ratio)(dlist[i]))
        view1, aug1_ratio = aug_views[i, 1], aug_ratios[i, 1]
        data_view2.append(aug_view(view1, aug1_ratio)(dlist[i]))

    return Batch.from_data_list(data_view1, exclude_keys=['mask']), Batch.from_data_list(data_view2, exclude_keys=['mask'])


def generate_view_from_dict(data, aug_views, aug_ratios):
    if type(aug_views) is not np.ndarray:
        aug_views = aug_views.cpu().data.numpy()
        aug_ratios = aug_ratios.cpu().data.numpy()
    data = data.cpu()
    dlist = data.to_data_list()
    bsz = aug_views.shape[0]
    data_view1 = []
    data_view2 = []
    for i in range(bsz):
        view1, aug1_ratio = pair_aug_dict[aug_views[i]][0], pair_aug_ratio_dict[aug_ratios[i]][0]
        data_view1.append(aug_view(view1, aug1_ratio)(dlist[i]))
        view1, aug1_ratio = pair_aug_dict[aug_views[i]][1], pair_aug_ratio_dict[aug_ratios[i]][1]
        data_view2.append(aug_view(view1, aug1_ratio)(dlist[i]))

    return Batch.from_data_list(data_view1, exclude_keys=['mask']), Batch.from_data_list(data_view2, exclude_keys=['mask'])


# class RandomSampler(BaseSampler):
#     def __init__(self, device, args):
#         self.device = device
#         self.aug_list = ['raw', 'dropN', 'permE', 'subgraph', 'maskN']
#         self.augratio_list = list(np.arange(0.1, 1.0, 0.1))
#
#     def sample(self, data):
#
#         # Random
#         bsz = data.batch.max().item() + 1
#         aug_views = torch.randint(len(self.aug_list), (bsz * 2,)).view(bsz, 2).data.numpy()
#         aug_ratios = torch.randint(len(self.augratio_list), (bsz * 2,)).view(bsz, 2).data.numpy()
#
#         data_view1, data_view2 = generate_view(data, aug_views, aug_ratios)
#         return data_view1, data_view2

class RandomSampler(BaseSampler):
    def __init__(self, device, args):
        self.device = device
        self.aug_list = ['raw', 'dropN', 'permE', 'subgraph', 'maskN']
        self.augratio_list = list(np.arange(0.1, 1.0, 0.1))

    def sample(self, data, x=None):

        # Random
        bsz = data.batch.max().item() + 1
        aug_views = torch.randint(len(pair_aug_dict), (bsz,)).data.numpy()
        aug_ratios = torch.randint(len(pair_aug_ratio_dict), (bsz,)).data.numpy()

        data_view1, data_view2 = generate_view_from_dict(data, aug_views, aug_ratios)
        return data_view1, data_view2, None


class SinglePolicyGradientSampler(BaseSampler):
    def __init__(self, device, args):
        # Hyperparameters
        self.learning_rate = 0.01
        self.start_entropy = 0.05 # Larger -> uniform
        self.end_entropy = 0.01 # End entropy
        self.decay = 30 # Linear
        self.num_updates = 5 # Iterations
        self.clip_norm = 30

        self.entropy_coefficient = self.start_entropy

        self.device = device
        self.high_net = SinglePolicyNet(
            args.rl_dim,
            len(pair_aug_dict)
        ).to(device)
        self.optimizer_high = torch.optim.Adam(
            self.high_net.parameters(),
            lr=self.learning_rate,
        )

        self.low_net = SinglePolicyNet(
            args.rl_dim + len(pair_aug_dict),
            # args.rl_dim,
            len(pair_aug_ratio_dict)
        ).to(device)
        self.optimizer_low = torch.optim.Adam(
            self.low_net.parameters(),
            lr=self.learning_rate,
        )

    def high_policy(self, x, training=True):
        logits = self.high_net.forward(x)
        policy_logits = logits["policy"]
        if training:
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1).squeeze()
        else:
            action = torch.argmax(policy_logits, dim=1).squeeze()

        return logits, action

    def low_policy(self, x, training=True):
        logits = self.low_net.forward(x)
        policy_logits = logits["policy"]
        if training:
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1).squeeze()
        else:
            action = torch.argmax(policy_logits, dim=1).squeeze()

        return logits, action

    def sample(self, data, x, training=True, return_logits=False):
        bsz, _ = x.shape
        x = x.detach()

        logits_high, action_high = self.high_policy(x, training=training)

        ones = torch.sparse.torch.eye(len(pair_aug_dict)).to(self.device)
        x_action = ones.index_select(0, action_high)
        x_low = torch.cat((x, x_action), dim=1)
        # x_low = x
        logits_low, action_low = self.low_policy(x_low, training=training)

        data_view1, data_view2 = generate_view_from_dict(data, action_high, action_low)

        if return_logits:
            return data_view1, data_view2, [action_high, action_low], [logits_high, logits_low]
        else:
            return data_view1, data_view2, [action_high, action_low]

    def learn(self, model, valid_index_loader, warmup=None):
        model.eval()

        count = 0
        stats = {
            "policy_loss_high": [],
            "entropy_loss_high": [],
            "total_loss_high": [],
            "policy_loss_low": [],
            "entropy_loss_low": [],
            "total_loss_low": [],
        }
        indicator = False
        while True:
            for batch_data in valid_index_loader:
                out, action, logits = model.compute_pred_and_logits(
                    batch_data,
                    self,
                )  # action=[high_action, low_action] logits=[high_logits, low_logits]
                high_action, low_action = action[0], action[1]
                high_logits, low_logits = logits[0], logits[1]
                # policy_logits = logits["policy"]
                out = out.detach()
                reward = torch.log(out + 1e-15)
                #if warmup is None:
                #    reward_0 = reward[action==0]
                #    reward_1 = reward[action==1]
                #    reward_2 = reward[action==2]
                #    print(len(reward_0), reward_0.mean(), len(reward_1), reward_1.mean(), len(reward_2), reward_2.mean())
                #    input()

                # Normalize rewards
                reward = (reward - reward.mean()) / (reward.std() + 10-8)
                # high-level Policy gradient
                policy_loss = compute_policy_loss(high_logits['policy'], high_action, reward)
                # Entropy
                entropy_loss = compute_entropy_loss(high_logits['policy'])

                if warmup is not None:
                    total_loss = entropy_loss
                else:
                    total_loss = policy_loss + entropy_loss * self.entropy_coefficient

                self.optimizer_high.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.high_net.parameters(), self.clip_norm)
                self.optimizer_high.step()

                stats["policy_loss_high"].append(policy_loss.item())
                stats["entropy_loss_high"].append(entropy_loss.item())
                stats["total_loss_high"].append(total_loss.item())

                # low-level Policy gradient
                policy_loss = compute_policy_loss(low_logits['policy'], low_action, reward)
                # Entropy
                entropy_loss = compute_entropy_loss(low_logits['policy'])

                if warmup is not None:
                    total_loss = entropy_loss
                else:
                    total_loss = policy_loss + entropy_loss * self.entropy_coefficient

                self.optimizer_low.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.low_net.parameters(), self.clip_norm)
                self.optimizer_low.step()

                stats["policy_loss_low"].append(policy_loss.item())
                stats["entropy_loss_low"].append(entropy_loss.item())
                stats["total_loss_low"].append(total_loss.item())

                count += 1
                if warmup is None and count >= self.num_updates:
                    indicator = True
                    break
                elif warmup is not None and count >= warmup:
                    break
            if indicator:
                break

        stats = {key: np.mean(stats[key]) for key in stats}
        print(stats)

        if warmup is None:
            self.entropy_coefficient = max(self.entropy_coefficient - (self.start_entropy - self.end_entropy) / self.decay, self.end_entropy)


pair_aug_dict = {
    0: ['raw', 'raw'],
    1: ['raw', 'dropN'],
    2: ['raw', 'permE'],
    3: ['raw', 'subgraph'],
    4: ['raw', 'maskN'],
    5: ['dropN', 'dropN'],
    6: ['dropN', 'permE'],
    7: ['dropN', 'subgraph'],
    8: ['dropN', 'maskN'],
    9: ['permE', 'permE'],
    10: ['permE', 'subgraph'],
    11: ['permE', 'maskN'],
    12: ['subgraph', 'subgraph'],
    13: ['subgraph', 'maskN'],
    14: ['maskN', 'maskN']
}

search_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ratio_pair = [[search_ratio[i], search_ratio[j]] for i in range(len(search_ratio)) for j in range(i, len(search_ratio))]
pair_aug_ratio_dict = {i: ratio_pair[i] for i in range(len(ratio_pair))}


class SingleActorCriticeSampler(BaseSampler):
    def __init__(self, device, args):
        # Hyperparameters
        self.learning_rate = 0.01
        self.start_entropy = 0.05 # Larger -> uniform
        self.end_entropy = 0.01 # End entropy
        self.decay = 30 # Linear
        self.num_updates = 10 # Iterations
        self.clip_norm = 30
        self.value_coeficient = 0.5

        self.entropy_coefficient = self.start_entropy
        self.candidates = []

        self.device = device
        self.net = SinglePolicyValueNet(
            args.hidden_channels,
            args.num_layers
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
        )

    def sample(self, h_edge_0, h_edge_1, training=True, return_logits=False):
        num_edges, max_hop, _ = h_edge_0.shape
        edge_features = torch.cat((h_edge_0[:,0].detach(), h_edge_1[:,0].detach()), dim=1)
        logits = self.net.forward(edge_features)
        policy_logits = logits["policy"]
        if training:
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1).squeeze()
        else:
            action = torch.argmax(policy_logits, dim=1).squeeze()

        if return_logits:
            return action, logits
        else:
            return action

    def learn(self, model, data, pos_valid_edge, valid_index_loader, warmup=None):
        model.eval()
        with torch.no_grad():
            h = model(data.x, data.edge_index)

        count = 0
        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
        }
        for perm in valid_index_loader:
            valid_edge = pos_valid_edge[perm].t()
            valid_edge_neg = generate_neg_sample(pos_valid_edge, data.num_nodes, data.x.device, perm.shape[0])

            pos_num, neg_num = valid_edge.shape[1], valid_edge_neg.shape[1]
            out, action, logits = model.compute_pred_and_logits(
                h,
                torch.cat((valid_edge, valid_edge_neg), dim=1),
                self,
            )
            policy_logits = logits["policy"]
            value_logits = logits["value"]
            out = out.squeeze()
            pos_out, neg_out = out[:pos_num], out[-neg_num:]
            pos_reward = torch.log(pos_out + 1e-15)
            neg_reward = torch.log(1 - neg_out + 1e-15)
            reward = torch.cat((pos_reward, neg_reward))
            advantage = reward - value_logits.detach().squeeze()

            # Normalize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 10-8)

            # Policy gradient
            policy_loss = compute_policy_loss(policy_logits, action, advantage)

            # Value loss
            value_loss = compute_value_loss(reward - value_logits) * self.value_coeficient

            # Entropy
            entropy_loss = compute_entropy_loss(policy_logits)

            if warmup is not None:
                total_loss = entropy_loss
            else:
                total_loss = policy_loss + value_loss + entropy_loss * self.entropy_coefficient

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy_loss"].append(entropy_loss.item())
            stats["total_loss"].append(total_loss.item())

            count += 1
            if warmup is None and count >= self.num_updates:
                break
            elif warmup is not None and count >= warmup:
                break

        stats = {key: np.mean(stats[key]) for key in stats}
        print(stats)

        if warmup is None:
            self.entropy_coefficient = max(self.entropy_coefficient - (self.start_entropy - self.end_entropy) / self.decay, self.end_entropy)



samplers = {
    "original": OriginalSampler,
    "random": RandomSampler,
    "single_policy_gradient": SinglePolicyGradientSampler,
    # "single_policy_gradientTable": SinglePolicyGradientSamplerTable,
    "single_actor_critic": SingleActorCriticeSampler,
}

def get_sampler(name):
    if name not in samplers:
        return ValueError("Sampler not supported. Choices: {}".format(samplers.keys()))
    return samplers[name]


class SinglePolicyValueNet(nn.Module):
    def __init__(self, dim, num_layers):
        super(SinglePolicyValueNet, self).__init__()

        self.fc1 = nn.Linear(dim*2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, num_layers)

        self.fc1_ = nn.Linear(dim, 64)
        self.fc2_ = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy_head(x)

        value_logits = self.value_head(x)
        return {"policy": policy_logits, "value": value_logits}


class SinglePolicyNet(nn.Module):
    def __init__(self, dim, num_layers):
        super(SinglePolicyNet, self).__init__()

        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_layers)
        # self.fc4 = nn.Linear(32, num_layers)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return {"policy": x}


def compute_policy_loss(policy_logits, action_id, reward):

    cross_entropy = F.nll_loss(
        F.log_softmax(policy_logits, dim=-1),
        target=action_id,
        reduction='none')

    loss = cross_entropy * reward
    loss = torch.mean(loss)

    return loss

def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.mean(policy * log_policy)

def compute_value_loss(advantages):
    return 0.5 * torch.mean(advantages ** 2)


def data_split_train_valid(dataset, ratio=0.2):
    num_class = dataset.num_classes
    y = dataset.data.y
    train_indices = []
    valid_indices = []
    all_index = torch.arange(0, len(dataset), 1)
    for i in range(num_class):
        mask_i = torch.eq(y, i)
        index_i = all_index[mask_i].numpy()
        np.random.shuffle(index_i)
        valid_len = int(len(index_i) * ratio)
        train_indices.append(torch.from_numpy(index_i[:-valid_len]))
        valid_indices.append(torch.from_numpy(index_i[-valid_len:]))

    train_indices = torch.cat(train_indices, dim=0)
    valid_indices = torch.cat(valid_indices, dim=0)
    train_dataset = dataset[train_indices.long()]
    valid_dataset = dataset[valid_indices.long()]
    return train_dataset, valid_dataset