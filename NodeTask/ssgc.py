import torch


class Net(torch.nn.Module):
    def __init__(self, num_features, out):
        super(Net, self).__init__()


        self.conv1 = torch.nn.Linear(num_features, out)


    def forward(self, F1):

        z = self.conv1(F1)


        return z