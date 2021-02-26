import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(5, 5*6)
        self.conv2 = GCNConv(5*6, 8)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return scatter_mean(x,data.batch,dim=0) #scatter_mean ensures output is independent of num_nodes