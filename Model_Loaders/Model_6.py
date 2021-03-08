def Load_model(name, args):
    from FunctionCollection import Loss_Functions, customModule, edge_feature_constructor
    import pytorch_lightning as pl
    
    from typing import Optional

    import torch
    from torch_scatter import scatter_sum, scatter_min, scatter_max
    from torch_scatter.utils import broadcast
#     from torch_geometric.nn import GATConv, FeaStConv

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
    

    N_edge_feats = args['N_edge_feats'] #6
    N_dom_feats = args['N_dom_feats']#6
    N_scatter_feats = 4
#     N_targets = args['N_targets']
    N_outputs = args['N_outputs']
    N_metalayers = args['N_metalayers'] #10
    N_hcs = args['N_hcs'] #32
    #Possibly add (edge/node/global)_layers
    
    crit, y_post_processor, output_post_processor, cal_acc = Loss_Functions(name, args)
    likelihood_fitting = True if name[-4:] == 'NLLH' else False
    
    class Net(customModule):
        def __init__(self):
            super(Net, self).__init__(crit, y_post_processor, output_post_processor, cal_acc, likelihood_fitting, args)

            self.act = torch.nn.SiLU()
            self.hcs = N_hcs

            N_x_feats = N_dom_feats + N_scatter_feats*N_edge_feats
#             N_u_feats = N_scatter_feats*N_x_feats

            self.x_encoder = torch.nn.Linear(N_x_feats,self.hcs)
            self.edge_attr_encoder = torch.nn.Linear(N_edge_feats,self.hcs)
            self.CoC_encoder = torch.nn.Linear(3+N_scatter_feats*N_x_feats,self.hcs)

            class EdgeModel(torch.nn.Module):
                def __init__(self,hcs,act):
                    super(EdgeModel, self).__init__()
                    self.hcs = hcs
                    self.act = act
                    self.lins = torch.nn.ModuleList()
                    for i in range(2):
                        self.lins.append(torch.nn.Linear(4*self.hcs,4*self.hcs))
                    self.decoder = torch.nn.Linear(4*self.hcs,self.hcs)

                def forward(self, src, dest, edge_attr, CoC, batch):
                    # x: [N, F_x], where N is the number of nodes.
                    # src, dest: [E, F_x], where E is the number of edges.
                    # edge_attr: [E, F_e]
                    # edge_index: [2, E] with max entry N - 1.
                    # u: [B, F_u], where B is the number of graphs.
                    # batch: [N] with max entry B - 1.
                    out = torch.cat([src, dest, edge_attr, CoC[batch]], dim=1)
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

                def forward(self, x, edge_index, edge_attr, CoC, batch):
                    row, col = edge_index
                    out = torch.cat([x[row],edge_attr],dim=1)
                    for lin in self.lins1:
                        out = self.act(lin(out))
                    out = scatter_distribution(out,col,dim=0) #8*hcs
                    out = torch.cat([x,out,CoC[batch]],dim=1)
                    for lin in self.lins2:
                        out = self.act(lin(out))
                    return self.act(self.decoder(out))

            class GlobalModel(torch.nn.Module):
                def __init__(self,hcs,act):
                    super(GlobalModel, self).__init__()

                def forward(self, x, edge_index, edge_attr, CoC, batch):
                    return CoC

            from torch_geometric.nn import MetaLayer
            self.ops = torch.nn.ModuleList()
            self.GRUCells = torch.nn.ModuleList()
            self.lins1 = torch.nn.ModuleList()
            self.lins2 = torch.nn.ModuleList()
            self.lins3 = torch.nn.ModuleList()
            for i in range(N_metalayers):
                self.ops.append(MetaLayer(EdgeModel(self.hcs,self.act), NodeModel(self.hcs,self.act), GlobalModel(self.hcs,self.act)))
                self.GRUCells.append( torch.nn.GRUCell(self.hcs*2 + 4 + N_x_feats,self.hcs) )
                self.lins1.append( torch.nn.Linear((1+N_scatter_feats)*self.hcs,(1+N_scatter_feats)*self.hcs) )
                self.lins2.append( torch.nn.Linear((1+N_scatter_feats)*self.hcs,self.hcs) )
                self.lins3.append( torch.nn.Linear(2*self.hcs,self.hcs) )

            self.decoders = torch.nn.ModuleList()
            self.decoders.append(torch.nn.Linear(self.hcs,self.hcs))
            self.decoders.append(torch.nn.Linear(self.hcs,self.hcs))

            self.decoder = torch.nn.Linear(self.hcs,N_outputs)

        def forward(self,data):
            x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
            pos = x[:,-3:]

            x = torch.cat([x,scatter_distribution(edge_attr,edge_index[1],dim=0)],dim=1)

            CoC = scatter_sum( pos*x[:,0].view(-1,1), batch, dim=0) / scatter_sum(x[:,0].view(-1,1), batch, dim=0)
            CoC = torch.cat([CoC,scatter_distribution(x,batch,dim=0)],dim=1)
            
            CoC_edge_index = torch.cat([torch.arange(x.shape[0]).view(1,-1).type_as(batch),batch.view(1,-1)],dim=0)

            cart = pos[CoC_edge_index[0],-3:] - CoC[CoC_edge_index[1],:3]
            del pos

            rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
            rho_mask = rho.squeeze() != 0
            cart[rho_mask] = cart[rho_mask] / rho[rho_mask]

            CoC_edge_attr = torch.cat([cart.type_as(x),rho.type_as(x),x[CoC_edge_index[0]]], dim=1)

            x = self.act(self.x_encoder(x))
            edge_attr = self.act(self.edge_attr_encoder(edge_attr))
            CoC = self.act(self.CoC_encoder(CoC))

#             u = torch.zeros( (batch.max() + 1, self.hcs) ).type_as(x)
            h = torch.zeros( (x.shape[0], self.hcs) ).type_as(x)

            for i, op in enumerate(self.ops):
                x, edge_attr, CoC = op(x, edge_index, edge_attr, CoC, batch)
                h = self.act( self.GRUCells[i]( torch.cat([CoC[CoC_edge_index[1]], x[CoC_edge_index[0]], CoC_edge_attr], dim=1), h ) )
                CoC = self.act( self.lins1[i]( torch.cat([CoC,scatter_distribution(h, batch, dim=0)],dim=1) ) )
                CoC = self.act( self.lins2[i]( CoC ) )
                h = self.act( self.GRUCells[i]( torch.cat([CoC[CoC_edge_index[1]], x[CoC_edge_index[0]], CoC_edge_attr], dim=1), h ) )
                x = self.act( self.lins3[i]( torch.cat([x,h],dim=1) ) )
            
            for lin in self.decoders:
                CoC = self.act(lin(CoC))

            CoC = self.decoder(CoC)
            return CoC
    return Net