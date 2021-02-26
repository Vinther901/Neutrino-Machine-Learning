from torch_geometric.nn import GINConv, SGConv, SAGPooling, GCNConv, GATConv, DNAConv
from torch_scatter import scatter_add, scatter_max, scatter_mean
import torch.nn.functional as F
import torch

# Taken from: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/dna.py

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        in_channels = 5
        hidden_channels = 30
        self.hidden_channels = hidden_channels
        num_layers = 2

        nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels,hidden_channels),torch.nn.ReLU())
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels,hidden_channels),torch.nn.ReLU())
        self.nnconv1 = GINConv(nn1)
        self.sconv1 = SGConv(hidden_channels,hidden_channels,K=5)
        self.sconv2 = SGConv(hidden_channels,hidden_channels,K=5)
        self.nnconv2 = GINConv(nn2)

        self.lin1 = torch.nn.Linear(in_channels,hidden_channels)
        self.DNAConvs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.DNAConvs.append(DNAConv(hidden_channels,heads=3,groups=1,dropout=0))
        self.lin2 = torch.nn.Linear(hidden_channels,2)

    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.nnconv1(x,edge_index)
        # x = F.dropout(x,p=0.5,training.self.training)
        x_all = x.view(-1,1,self.hidden_channels)

        x = self.sconv1(x,edge_index)

        x_all = torch.cat([x_all,x.view(-1,1,self.hidden_channels)],dim=1)

        x = self.sconv2(x,edge_index)

        x_all = torch.cat([x_all,x.view(-1,1,self.hidden_channels)],dim=1)

        x = self.nnconv2(x,edge_index)

        x_all = torch.cat([x_all,x.view(-1,1,self.hidden_channels)],dim=1)

        for conv in self.DNAConvs:
            x = F.relu(conv(x_all,edge_index))
            x = x.view(-1,1,self.hidden_channels)
            x_all = torch.cat([x_all,x],dim=1)
        x = x_all[:,-1]
        x = scatter_add(x,data.batch,dim=0)
        # x = F.dropout(x,p=0.5,training.self.training)
        x = self.lin2(x)
        return torch.log_softmax(x,dim=1)