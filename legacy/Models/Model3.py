from torch_geometric.nn import GINConv, SGConv
import torch
from torch_scatter import scatter_add, scatter_max
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(5,30),torch.nn.ReLU())
        nn2 = torch.nn.Sequential(torch.nn.Linear(30,30),torch.nn.ReLU())
        self.nnconv1 = GINConv(nn1)
        self.sconv1 = SGConv(30,30,K=5)
        self.sconv2 = SGConv(30,30,K=5)
        self.nnconv2 = GINConv(nn2)
        self.nn = torch.nn.Linear(30,3)
        self.m = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.nnconv1(x,edge_index)
        x = self.sconv1(x,edge_index)
        x = self.sconv2(x,edge_index)
        x = self.nnconv2(x,edge_index)

        # x = torch.nn.Dropout()

        x,_ = scatter_max(x,data.batch,dim=0)
        x = self.nn(x)

        return self.m(x)