def Load_model(name, args):
    from FunctionCollection import Loss_Functions, customModule, edge_feature_constructor
    import pytorch_lightning as pl
    
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
    
    @torch.jit.script
    def x_feature_constructor(x, graph_node_counts):
        tmp = []
        a : List[int] = graph_node_counts.tolist()
        for tmp_x in x.split(a):
            tmp_x = tmp_x.unsqueeze(1) - tmp_x
            
            cart = tmp_x[:,:,-3:]
            
            rho = torch.norm(cart, p=2, dim=-1).unsqueeze(2)
            rho_mask = rho.squeeze() != 0
            if rho_mask.sum() != 0:
                cart[rho_mask] = cart[rho_mask] / rho[rho_mask]
            tmp_x = torch.cat([cart,rho,tmp_x[:,:,:-3]],dim=2)
            
            tmp.append(torch.cat([tmp_x.mean(1),tmp_x.std(1),tmp_x.min(1)[0],tmp_x.max(1)[0]],dim=1))
        return torch.cat(tmp,0)
    
    @torch.jit.script
    def time_edge_indeces(t,
                          batch: torch.Tensor):
        time_sort = torch.argsort(t)
        graph_ids, graph_node_counts = torch.unique(batch,return_counts=True)
        batch_time_sort = torch.cat( [time_sort[batch[time_sort] == i] for i in graph_ids] )
        time_edge_index = torch.cat([batch_time_sort[:-1].view(1,-1),batch_time_sort[1:].view(1,-1)],dim=0)

        tmp_ind = (torch.cumsum(graph_node_counts,0) - 1)[:-1]
        li : List[int] = (tmp_ind + 1).tolist()
        time_edge_index[1,tmp_ind] = time_edge_index[0, [0] + li[:-1]]
        time_edge_index = torch.cat([time_edge_index, 
                                     torch.cat([time_edge_index[1,-1].view(1,1),
                                                time_edge_index[0,(tmp_ind + 1)[-1]].view(1,1)])],dim=1)

        time_edge_index = time_edge_index[:,torch.argsort(time_edge_index[1])]
        return time_edge_index
    
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
            
            class MLP(torch.nn.Module):
                def __init__(self, hcs_list, act = self.act, clean_out = False):
                    super(MLP, self).__init__()
                    mlp = []
                    for i in range(1,len(hcs_list)):
                        mlp.append(torch.nn.Linear(hcs_list[i-1], hcs_list[i]))
                        mlp.append(torch.nn.BatchNorm1d(hcs_list[i]))
                        mlp.append(act)
                    
                    if clean_out:
                        self.mlp = torch.nn.Sequential(*mlp[:-2])
                    else:
                        self.mlp = torch.nn.Sequential(*mlp)
                def forward(self, x):
                    return self.mlp(x)
            
            class GRUConv(torch.nn.Module):
                def __init__(self,hcs = self.hcs, act = self.act):
                    super(GRUConv, self).__init__()
                    self.act = act
                    self.hcs = hcs
                    self.GRU = torch.nn.GRUCell(self.hcs*2,self.hcs)
                    
                    self.lin_CoC_msg = MLP([N_scatter_feats*self.hcs, self.hcs],clean_out = True)
                    self.lin_CoC_self = MLP([self.hcs, self.hcs],clean_out = True)
                    
#                     self.CoC_batch_norm = torch.nn.BatchNorm1d(self.hcs)
                    self.CoC_mlp = MLP([2*self.hcs,self.hcs])
                    
                    self.lin_x_msg = MLP([self.hcs, self.hcs],clean_out = True)
                    self.lin_x_self = MLP([self.hcs, self.hcs],clean_out = True)
                    
#                     self.x_batch_norm = torch.nn.BatchNorm1d(self.hcs)
                    self.x_mlp = MLP([2*self.hcs,self.hcs])
                
                def forward(self, x, CoC, h, batch):
                    h = self.act( self.GRU( torch.cat([CoC[batch], x], dim=1), h) )
                    
                    msg = self.lin_CoC_msg( scatter_distribution(h, batch, dim=0) )
                    CoC = self.lin_CoC_self(CoC)
                    
                    CoC = self.CoC_mlp(torch.cat([self.act(CoC + msg), self.act(CoC - msg)],dim=1))
#                     CoC = self.act( self.CoC_batch_norm(msg+CoC) )
                    
                    h = self.act( self.GRU( torch.cat([x, CoC[batch]], dim=1), h) )
                    
                    msg = self.lin_x_msg(h)
                    x = self.lin_x_self(x)
                    
                    x = self.x_mlp(torch.cat([self.act(x + msg), self.act(x - msg)],dim=1))
#                     x = self.act( self.x_batch_norm(msg+x) )
                    return x, CoC, h

            N_x_feats = 2*N_dom_feats + 4*(N_dom_feats + 1) + N_edge_feats + 4 + 3
            N_CoC_feats = 3+N_scatter_feats*N_x_feats
            
            self.x_encoder = MLP([N_x_feats,4*self.hcs,self.hcs])
            self.CoC_encoder = MLP([N_CoC_feats,4*self.hcs,self.hcs])

            self.convs = torch.nn.ModuleList()
            for i in range(N_metalayers):
                self.convs.append(GRUConv())
                
            self.decoder = MLP([(1+N_scatter_feats)*self.hcs,self.hcs,self.hcs,N_outputs],clean_out=True)
#             self.decoder = MLP([(1+N_scatter_feats)*self.hcs,self.hcs,self.hcs])
            
#             self.decoders = torch.nn.ModuleList()
#             for _ in range(N_outputs):
#                 self.decoders.append(torch.nn.Linear(self.hcs,1))


        def forward(self,data):
            x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
            ###############
            x = x.float()
            ###############
            pos = x[:,-3:]
            
            graph_ids, graph_node_counts = batch.unique(return_counts=True)
            
            time_edge_index = time_edge_indeces(x[:,1],batch)
                                      
            edge_attr = edge_feature_constructor(x, time_edge_index)

            # Define central nodes at Center of Charge:
            CoC = scatter_sum( pos*x[:,0].view(-1,1), batch, dim=0) / scatter_sum(x[:,0].view(-1,1), batch, dim=0)
            
            # Define edge_attr for those edges:
            cart = pos[:,-3:] - CoC[batch,:3]
            del pos
            rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
            rho_mask = rho.squeeze() != 0
            cart[rho_mask] = cart[rho_mask] / rho[rho_mask]
            CoC_edge_attr = torch.cat([cart.type_as(x),rho.type_as(x)], dim=1)
            
            x = torch.cat([x,
                           x_feature_constructor(x,graph_node_counts),
                           edge_attr,
                           x[time_edge_index[0]],
                           CoC_edge_attr,
                           CoC[batch]],dim=1)
            
            CoC = torch.cat([CoC,scatter_distribution(x,batch,dim=0)],dim=1)

            x = self.x_encoder(x)
            CoC = self.CoC_encoder(CoC)

            h = torch.zeros( (x.shape[0], self.hcs) ).type_as(x)

            for i in range(N_metalayers):
                x, CoC, h = self.convs[i](x, CoC, h, batch)
            
            CoC = torch.cat([CoC,scatter_distribution(x, batch, dim=0)],dim=1)

            CoC = self.decoder(CoC)
            
#             out = []
#             for mlp in self.decoders:
#                 out.append(mlp(CoC))
#             CoC = torch.cat(out,dim=1)
            
            return CoC
    return Net