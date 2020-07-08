from torch_geometric.nn import GINConv, SGConv, SAGPooling, GCNConv, GATConv, DNAConv
import torch
from torch_scatter import scatter_add, scatter_max, scatter_mean
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(5,10)
        self.sagpool1 = SAGPooling(10,0.2,GCNConv)
        self.gatconv = GATConv(10,20,heads=3)
        self.nn = torch.nn.Sequential(torch.nn.Linear(60,30),torch.nn.ReLU(),torch.nn.Linear(30,3))
        self.m = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        x = self.conv1(x,edge_index)
        x, edge_index, _, batch, _, _ = self.sagpool1(x,edge_index,batch=batch)

        x = self.gatconv(x,edge_index)

        x = scatter_add(x,batch,dim=0)

        x = self.nn(x)

        return self.m(x)