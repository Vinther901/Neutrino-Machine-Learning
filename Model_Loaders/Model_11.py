def Load_model(name, args):
    from FunctionCollection import Loss_Functions, customModule, edge_feature_constructor
    import pytorch_lightning as pl
    
    from typing import Optional

    import torch
    from torch_scatter import scatter_sum, scatter_min, scatter_max
    from torch_scatter.utils import broadcast
    import torch.nn.functional as F

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
    N_outputs = args['N_outputs']
    N_metalayers = args['N_metalayers'] #10
    N_hcs = args['N_hcs'] #32
#     p_dropout = args['dropout']
    
    crit, y_post_processor, output_post_processor, cal_acc = Loss_Functions(name, args)
    likelihood_fitting = True if name[-4:] == 'NLLH' else False
    
    class Net(customModule):
        def __init__(self):
            super(Net, self).__init__(crit, y_post_processor, output_post_processor, cal_acc, likelihood_fitting, args)
            print("This model assumes Charge is at index 0 and position is the last three")

            self.act = torch.nn.SiLU()
            self.hcs = N_hcs
#             self.dropout = torch.nn.Dropout(p=p_dropout)
            
            class MLP(torch.nn.Module):
                def __init__(self, hcs_list, act = self.act):
                    super(MLP, self).__init__()
                    mlp = []
                    for i in range(1,len(hcs_list)):
                        mlp.append(torch.nn.Linear(hcs_list[i-1], hcs_list[i]))
                        mlp.append(torch.nn.BatchNorm1d(hcs_list[i]))
                        mlp.append(act)
                    
                    self.mlp = torch.nn.Sequential(*mlp)
                def forward(self, x):
                    return self.mlp(x)
                
            class AttGNN(torch.nn.Module):
                def __init__(self, hcs_in, hcs_out, act = self.act):
                    super(AttGNN, self).__init__()
                    
                    self.beta = torch.nn.Parameter(torch.ones(1))
                    
                    self.self_mlp = MLP([hcs_in,hcs_in,hcs_out])
                    self.msg_mlp = MLP([hcs_in,hcs_in,hcs_out])

                def forward(self, x, graph_node_counts):
                    li : List[int] = graph_node_counts.tolist()
                    tmp = []
                    for tmp_x, msg in zip(x.split(li), self.msg_mlp(x).split(li)):
                        att = F.normalize(tmp_x,p=2.,dim=1)
                        att = torch.cdist(att,att)
                        tmp.append(torch.matmul(F.softmin(self.beta*att,1),msg))
                    return self.self_mlp(x) + torch.cat(tmp,0)
#                     att = []
#                     for tmp_x in x.split(graph_node_counts.tolist()):
#                         tmp_x = F.normalize(tmp_x,p=2,dim=1)
#                         tmp_x = torch.cdist(tmp_x,tmp_x)
#                         att.append(F.softmax(self.beta*tmp_x,1))
#                     return self.self_mlp(x) + torch.matmul(torch.block_diag(*att), self.msg_mlp(x))
                
            N_x_feats = N_dom_feats + 4*(N_dom_feats + 1) + 4 + 3
            self.x_encoder = MLP([N_x_feats,self.hcs,self.hcs])
            self.CoC_encoder = torch.nn.Linear(3+N_scatter_feats*N_x_feats,self.hcs)
            
            self.convs = torch.nn.ModuleList()
            self.GRUCells = torch.nn.ModuleList()
            
            self.lins_CoC_msg = torch.nn.ModuleList()
            self.lins_CoC_self = torch.nn.ModuleList()
            self.CoC_batch_norm = torch.nn.ModuleList()
            
            self.lins_x_msg = torch.nn.ModuleList()
            self.lins_x_self = torch.nn.ModuleList()
            self.x_batch_norm = torch.nn.ModuleList()
            
            for i in range(N_metalayers):
                self.convs.append(torch.jit.script(AttGNN(self.hcs,self.hcs)))
                self.GRUCells.append(torch.nn.GRUCell(self.hcs*2,self.hcs))
                
                self.lins_CoC_msg.append( torch.nn.Linear(N_scatter_feats*self.hcs, self.hcs) )
                self.lins_CoC_self.append( torch.nn.Linear(self.hcs, self.hcs) )
                self.CoC_batch_norm.append( torch.nn.BatchNorm1d(self.hcs) )
                
                self.lins_x_msg.append( torch.nn.Linear(self.hcs, self.hcs) )
                self.lins_x_self.append( torch.nn.Linear(self.hcs, self.hcs) )
                self.x_batch_norm.append( torch.nn.BatchNorm1d(self.hcs) )
            
            self.decoder = MLP([(1+N_scatter_feats)*self.hcs,N_scatter_feats//2*self.hcs,self.hcs,N_outputs])
 
        def return_CoC_and_edge_attr(self, x, batch):
            pos = x[:,-3:]
            charge = x[:,0].view(-1,1)
            
            # Define central nodes at Center of Charge:
            CoC = scatter_sum( pos*charge, batch, dim=0) / scatter_sum(charge, batch, dim=0)
            
            # Define edge_attr for those edges:
            cart = pos - CoC[batch]
            rho = torch.norm(cart, p=2, dim=1).view(-1, 1)
            rho_mask = rho.squeeze() != 0
            cart[rho_mask] = cart[rho_mask] / rho[rho_mask]
            CoC_edge_attr = torch.cat([cart.type_as(x),rho.type_as(x)], dim=1)
            return CoC, CoC_edge_attr
            

        def forward(self,data):
            x, batch = data.x.float(), data.batch
            
            graph_ids, graph_node_counts = batch.unique(return_counts=True)
            
            CoC, CoC_edge_attr = self.return_CoC_and_edge_attr(x, batch)
            
            x = torch.cat([x,
                           x_feature_constructor(x,graph_node_counts),
                           CoC_edge_attr,
                           CoC[batch]],dim=1)
            
            CoC = torch.cat([CoC,scatter_distribution(x,batch,dim=0)],dim=1)
            
            x = self.x_encoder(x)
            CoC = self.act(self.CoC_encoder(CoC))
            
            h = torch.zeros( (x.shape[0], self.hcs) ).type_as(x)
            for i, conv in enumerate(self.convs):
                x = conv(x,graph_node_counts)
                h = self.act( self.GRUCells[i]( torch.cat([CoC[batch], x], dim=1), h ) )
                CoC = self.act( self.CoC_batch_norm[i]( self.lins_CoC_msg[i](scatter_distribution(h, batch, dim=0)) + self.lins_CoC_self[i](CoC) ) )
                h = self.act( self.GRUCells[i]( torch.cat([CoC[batch], x], dim=1), h ) )
                x = self.act( self.x_batch_norm[i]( self.lins_x_msg[i](h) + self.lins_x_self[i](x) ) )
                
            CoC = torch.cat([CoC,scatter_distribution(x, batch, dim=0)],dim=1)
            
            return self.decoder(CoC)
    return Net