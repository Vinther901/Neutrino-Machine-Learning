import torch
from torch_scatter import scatter_std, scatter_mean

N_edge_feats = 6#3
N_targets = 2
class Net11(torch.nn.Module):
  def __init__(self):
    super(Net11, self).__init__()

    self.act = torch.nn.SiLU() #SiLU is x/(1+exp(-x))
    self.hcs = 32

    N_x_feats = 5 + 2*N_edge_feats
    N_u_feats = 2*N_x_feats

    self.x_encoder = torch.nn.Linear(N_x_feats,self.hcs)
    self.edge_attr_encoder = torch.nn.Linear(N_edge_feats,self.hcs)
    self.u_encoder = torch.nn.Linear(N_u_feats,self.hcs)

    class EdgeModel(torch.nn.Module):
      def __init__(self,hcs,act):
        super(EdgeModel, self).__init__()
        self.hcs = hcs
        self.act = act
        self.lins = torch.nn.ModuleList()
        for i in range(2):
          self.lins.append(torch.nn.Linear(4*self.hcs,4*self.hcs))
        self.decoder = torch.nn.Linear(4*self.hcs,self.hcs)

      def forward(self, src, dest, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # edge_index: [2, E] with max entry N - 1.
        # u: [B, F_u], where B is the number of graphs.
        # batch: [N] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        for lin in self.lins:
          out = self.act(lin(out))
        return self.act(self.decoder(out))

    class NodeModel(torch.nn.Module):
      def __init__(self,hcs,act):
        super(NodeModel, self).__init__()
        self.hcs = hcs
        self.act = act
        self.lins1 = torch.nn.ModuleList()
        for i in range(2):
          self.lins1.append(torch.nn.Linear(2*self.hcs,2*self.hcs))
        self.lins2 = torch.nn.ModuleList()
        for i in range(2):
          self.lins2.append(torch.nn.Linear(4*self.hcs,4*self.hcs))
        self.decoder = torch.nn.Linear(4*self.hcs,self.hcs)

      def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row],edge_attr],dim=1)
        for lin in self.lins1:
          out = self.act(lin(out))
        # out = torch.cat([scatter_mean(out,col,dim=0),torch.square(scatter_std(out,col,dim=0))],dim=1) #6*hcs
        out = scatter_mean(out,col,dim=0) #4*hcs
        out = torch.cat([x,out,u[batch]],dim=1)
        for lin in self.lins2:
          out = self.act(lin(out))
        return self.act(self.decoder(out))

    class GlobalModel(torch.nn.Module):
      def __init__(self,hcs,act):
        super(GlobalModel, self).__init__()
        self.hcs = hcs
        self.act = act
        self.lins1 = torch.nn.ModuleList()
        for i in range(2):
          self.lins1.append(torch.nn.Linear(2*self.hcs,2*self.hcs))
        self.decoder = torch.nn.Linear(2*self.hcs,self.hcs)

      def forward(self, x, edge_index, edge_attr, u, batch):
        # out = torch.cat([u,scatter_mean(x,batch,dim=0),torch.square(scatter_std(x,batch,dim=0))],dim=1) # 3*hcs
        out = torch.cat([u,scatter_mean(x,batch,dim=0)],dim=1) # 2*hcs
        for lin in self.lins1:
          out = self.act(lin(out))
        return self.act(self.decoder(out))
    
    from torch_geometric.nn import MetaLayer
    self.ops = torch.nn.ModuleList()
    for i in range(10):
      self.ops.append(MetaLayer(EdgeModel(self.hcs,self.act), NodeModel(self.hcs,self.act), GlobalModel(self.hcs,self.act)))
    
    # self.rnn = torch.nn.LSTM(self.hcs,self.hcs,num_layers=1,batch_first=True)

    self.decoders = torch.nn.ModuleList()
    # for i in range(8,2,-2):
    #   self.decoders.append(torch.nn.Linear(i*self.hcs,(i-2)*self.hcs))
    self.decoders.append(torch.nn.Linear(10*self.hcs,2*self.hcs))
    
    self.decoders2 = torch.nn.ModuleList()
    # for i in range(4,1,-1):
    #   self.decoders2.append(torch.nn.Linear(i*self.hcs,(i-1)*self.hcs))
    self.decoders2.append(torch.nn.Linear(4*self.hcs,1*self.hcs))
    
    self.decoder = torch.nn.Linear(self.hcs,3)

  def forward(self,data):
    x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
    # times = x[:,1].detach().clone()
    # x = torch.cuda.FloatTensor(self.scaler.transform(x.cpu()))
    x = torch.cat([x,scatter_mean(edge_attr,edge_index[1],dim=0),scatter_std(edge_attr,edge_index[1],dim=0)],dim=1)
    u = torch.cat([scatter_mean(x,batch,dim=0),scatter_std(x,batch,dim=0)],dim=1)


    x = self.act(self.x_encoder(x))
    edge_attr = self.act(self.edge_attr_encoder(edge_attr))
    # u = torch.cuda.FloatTensor(batch.max()+1,self.hcs).fill_(0)
    u = self.act(self.u_encoder(u))
    
    
    for i, op in enumerate(self.ops):
      x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)
      if i == 0:
        out = u
      else:
        out = torch.cat([out,u],dim=1)
    
    # graph_indices, graph_sizes = data.batch.unique(return_counts=True)
    # tmp = torch.zeros((graph_indices.max()+1,graph_sizes.max(),self.hcs)).to(device)
    # sort = torch.sort(graph_sizes,dim=0,descending=True).indices
    # reverse_sort = torch.sort(sort,dim=0).indices

    # for tmp_index, i in enumerate(graph_indices[sort]):
    #   tmp_graph = x[data.batch==i]
    #   tmp_times = times[data.batch==i]
    #   tmp[tmp_index,:graph_sizes[i]] = tmp_graph[torch.sort(tmp_times,dim=0).indices]
    # tmp = torch.nn.utils.rnn.pack_padded_sequence(tmp,graph_sizes[sort].cpu(),batch_first=True,enforce_sorted=True)

    # tmp, (hn, cn) = self.rnn(tmp)
    # #Maybe add TCN?
      
    for lin in self.decoders:
      out = self.act(lin(out))
    
    # out = torch.cat([out,cn[0,reverse_sort],
    #                  scatter_mean(torch.cat([x,scatter_mean(edge_attr,edge_index[1],dim=0)],dim=1),batch,dim=0)],dim=1)

    out = torch.cat([out,scatter_mean(torch.cat([x,scatter_mean(edge_attr,edge_index[1],dim=0)],dim=1),batch,dim=0)],dim=1)

    for lin in self.decoders2:
      out = self.act(lin(out))
    
    return self.decoder(out)
