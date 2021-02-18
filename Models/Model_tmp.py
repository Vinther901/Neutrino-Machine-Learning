def Load_model():
    from typing import Optional

    import torch
    from torch_scatter import scatter_sum, scatter_min, scatter_max
    from torch_scatter.utils import broadcast

    @torch.jit.script
    def scatter_distribution(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                            out: Optional[torch.Tensor] = None,
                            dim_size: Optional[int] = None,
                            unbiased: bool = True) -> torch.Tensor:

        if out is not None:
            dim_size = out.size(dim)

        if dim < 0:
            dim = src.dim() + dim

        count_dim = dim
        if index.dim() <= dim:
            count_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

        index = broadcast(index, src, dim)
        tmp = scatter_sum(src, index, dim, dim_size=dim_size)
        count = broadcast(count, tmp, dim).clamp(1)
        mean = tmp.div(count)

        var = (src - mean.gather(dim, index))
        var = var * var
        var = scatter_sum(var, index, dim, out, dim_size)

        if unbiased:
            count = count.sub(1).clamp_(1)
        var = var.div(count)
        maximum = scatter_max(src, index, dim, out, dim_size)[0]
        minimum = scatter_min(src, index, dim, out, dim_size)[0]

        return torch.cat([mean,var,maximum,minimum],dim=1)
    import torch

    N_edge_feats = 6
    N_dom_feats = 6
    N_scatter_feats = 4
    N_targets = 1
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.act = torch.nn.SiLU()
            self.hcs = 32

            N_x_feats = N_dom_feats + N_scatter_feats*N_edge_feats
            N_u_feats = N_scatter_feats*N_x_feats

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
                        self.lins2.append(torch.nn.Linear((2 + 2*N_scatter_feats)*self.hcs,(2 + 2*N_scatter_feats)*self.hcs))
                    self.decoder = torch.nn.Linear((2 + 2*N_scatter_feats)*self.hcs,self.hcs)

                def forward(self, x, edge_index, edge_attr, u, batch):
                    row, col = edge_index
                    out = torch.cat([x[row],edge_attr],dim=1)
                    for lin in self.lins1:
                        out = self.act(lin(out))
                    out = scatter_distribution(out,col,dim=0) #8*hcs
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
                        self.lins1.append(torch.nn.Linear((1 + N_scatter_feats)*self.hcs,(1 + N_scatter_feats)*self.hcs))
                    self.decoder = torch.nn.Linear((1 + N_scatter_feats)*self.hcs,self.hcs)

                def forward(self, x, edge_index, edge_attr, u, batch):
                    out = torch.cat([u,scatter_distribution(x,batch,dim=0)],dim=1) # 5*hcs
                    for lin in self.lins1:
                        out = self.act(lin(out))
                    return self.act(self.decoder(out))

            from torch_geometric.nn import MetaLayer
            self.ops = torch.nn.ModuleList()
            num_metalayers = 10
            for i in range(num_metalayers):
                self.ops.append(MetaLayer(EdgeModel(self.hcs,self.act), NodeModel(self.hcs,self.act), GlobalModel(self.hcs,self.act)))

            self.decoders = torch.nn.ModuleList()
            self.decoders.append(torch.nn.Linear((1+num_metalayers)*self.hcs + N_scatter_feats*N_x_feats,10*self.hcs))
            self.decoders.append(torch.nn.Linear(10*self.hcs,8*self.hcs))
            self.decoders.append(torch.nn.Linear(8*self.hcs,6*self.hcs))
            self.decoders.append(torch.nn.Linear(6*self.hcs,4*self.hcs))
            self.decoders.append(torch.nn.Linear(4*self.hcs,2*self.hcs))

            self.decoder = torch.nn.Linear(2*self.hcs,2)

            def forward(self,data):
                x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch

                x = torch.cat([x,scatter_distribution(edge_attr,edge_index[1],dim=0)],dim=1)
                u = torch.cat([scatter_distribution(x,batch,dim=0)],dim=1)

                x0 = x.clone()

                x = self.act(self.x_encoder(x))
                edge_attr = self.act(self.edge_attr_encoder(edge_attr))
                u = self.act(self.u_encoder(u))
                out = u.clone()

                for i, op in enumerate(self.ops):
                    x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)
                    out = torch.cat([out,u.clone()],dim=1)

                out = torch.cat([out,scatter_distribution(x0,batch,dim=0)],dim=1)
                for lin in self.decoders:
                    out = self.act(lin(out))

                out = self.decoder(out)
                return out
    return Net()