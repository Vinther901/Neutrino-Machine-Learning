from torch_geometric.nn import GINConv, SGConv, SAGPooling, GCNConv, GATConv, DNAConv, ChebConv
from torch_scatter import scatter_add, scatter_max, scatter_mean
import torch.nn.functional as F
import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hcs = 60 # Hidden channels
        
        # Model1
        self.lin1 = torch.nn.Linear(5,self.hcs)
        self.model1 = torch.nn.ModuleList()
        for i in range(10):
          self.model1.append(GCNConv(self.hcs,self.hcs))
        # self.lin11 = torch.nn.Linear(self.hcs*3,1)
        #add GATConv?
        
        # Model2
        self.lin2 = torch.nn.Linear(5,self.hcs)
        self.model2 = torch.nn.ModuleList()
        for i in range(10):
          self.model2.append(GINConv(torch.nn.Sequential(torch.nn.Linear(self.hcs,self.hcs),torch.nn.Tanhshrink())))
        # self.lin22 = torch.nn.Linear(self.hcs*3,1)

        # Model3
        self.lin3 = torch.nn.Linear(5,self.hcs)
        self.ChebConv = ChebConv(self.hcs,self.hcs,10)
        # self.lin33 = torch.nn.Linear(self.hcs*3,1)

        # Model4 DNAConv
        # self.lin5 = torch.nn.Linear(5,self.hcs)
        # self.model4 = torch.nn.ModuleList()
        # for i in range(5):

        # Final
        self.linFinal = torch.nn.Linear(3*self.hcs*3,1) # input size = Number of models * 1

    def forward(self, data):
        x1 = x2 = x3 = data.x
        edge_index = data.edge_index

        # Model1
        x1 = F.leaky_relu(self.lin1(x1))
        for conv in self.model1:
          x1 = conv(x1,edge_index)
        x1 = torch.cat([scatter_add(x1,data.batch,dim=0),scatter_mean(x1,data.batch,dim=0),scatter_max(x1,data.batch,dim=0)[0]],dim=1)
        # x1 = F.softmax(self.lin11(x1))
        
        # Model2
        x2 = F.leaky_relu(self.lin2(x2))
        for conv in self.model2:
          x2 = conv(x2,edge_index)
        x2 = torch.cat([scatter_add(x2,data.batch,dim=0),scatter_mean(x2,data.batch,dim=0),scatter_max(x2,data.batch,dim=0)[0]],dim=1)
        # x2 = F.softmax(self.lin22(x2))
        
        # Model3
        x3 = F.leaky_relu(self.lin3(x3))
        x3 = self.ChebConv(x3, edge_index)
        x3 = torch.cat([scatter_add(x3,data.batch,dim=0),scatter_mean(x3,data.batch,dim=0),scatter_max(x3,data.batch,dim=0)[0]],dim=1)
        # x3 = F.softmax(self.lin33(x3))

        return F.log_softmax(self.linFinal(torch.cat([x1,x2,x3],dim=1)))