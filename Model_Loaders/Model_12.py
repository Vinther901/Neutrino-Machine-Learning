def Load_model(name, args):
    from FunctionCollection import Loss_Functions, customModule, edge_feature_constructor
    import pytorch_lightning as pl
    
    from typing import Optional

    import torch
    from torch_scatter import scatter_sum, scatter_min, scatter_max, scatter_softmax
    from torch_scatter.utils import broadcast
    import torch.nn.functional as F
#     from torch_geometric.utils import softmax

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
                def __init__(self, hcs_list, act = self.act, no_final_act = False):
                    super(MLP, self).__init__()
                    mlp = []
                    for i in range(1,len(hcs_list)):
                        mlp.append(torch.nn.Linear(hcs_list[i-1], hcs_list[i]))
                        mlp.append(torch.nn.BatchNorm1d(hcs_list[i]))
                        mlp.append(act)
                    if not no_final_act:
                        self.mlp = torch.nn.Sequential(*mlp)
                    else:
                        self.mlp = torch.nn.Sequential(*mlp[:-1])
                def forward(self, x):
                    return self.mlp(x)
                
#             class AttGNN(torch.nn.Module):
#                 def __init__(self, hcs_in, hcs_out, act = self.act):
#                     super(AttGNN, self).__init__()
                    
# #                     self.att_mlp = torch.nn.Sequential(torch.nn.Linear(2*hcs_in,hcs_in),
# #                                                        act,
# #                                                        torch.nn.Linear(hcs_in,1))
#                     self.att_mlp = torch.nn.Linear(2*hcs_in,1)
                    
#                     self.self_mlp = MLP([hcs_in,hcs_in,hcs_out])
#                     self.msg_mlp = MLP([hcs_in,hcs_in,hcs_out])

#                 def forward(self, x, graph_node_counts):
#                     li : List[int] = graph_node_counts.tolist()
#                     tmp = []
#                     for tmp_x, msg in zip(x.split(li), self.msg_mlp(x).split(li)):
#                         tmp_x = torch.cat([tmp_x.unsqueeze(1) - tmp_x, tmp_x.unsqueeze(1) + tmp_x],dim=2)
#                         tmp.append(torch.matmul(self.att_mlp(tmp_x).squeeze(),msg))
#                     return self.self_mlp(x) + torch.cat(tmp,0)
            class AttGNN(torch.nn.Module):
                def __init__(self, hcs_in, hcs_out, act = self.act):
                    super(AttGNN, self).__init__()
                                        
                    self.self_mlp = MLP([hcs_in,hcs_in,hcs_out])
                    self.msg_mlp = torch.nn.Linear(2*hcs_in,hcs_out)#MLP([2*hcs_in,hcs_in,hcs_out])
                    self.msg_mlp2 = MLP([4*hcs_out,2*hcs_out,hcs_out])

                def forward(self, x, graph_node_counts):
#                     print(graph_node_counts.max().item(), graph_node_counts.float().mean().item(), graph_node_counts.float().std().item(), graph_node_counts.min().item())
                    li : List[int] = graph_node_counts.tolist()
                    tmp = []
                    for tmp_x in x.split(li):
                        tmp_x = torch.cat([tmp_x.unsqueeze(1) - tmp_x, tmp_x.unsqueeze(1) + tmp_x],dim=2)
                        tmp_x = self.msg_mlp(tmp_x)
                        tmp.append(self.msg_mlp2(torch.cat([tmp_x.mean(1),tmp_x.std(1),tmp_x.min(1)[0],tmp_x.max(1)[0]],dim=1)))
                    out = self.self_mlp(x) + torch.cat(tmp,0)
                    del tmp, tmp_x
                    return out
                
            N_x_feats = N_dom_feats + 4*(N_dom_feats + 1) + 4 + 3
            
#             self.att = MLP([N_x_feats,self.hcs,1],no_final_act=True)
            
            self.x_encoder = MLP([N_x_feats,self.hcs,self.hcs])
            self.convs = torch.nn.ModuleList()
            for i in range(N_metalayers):
                self.convs.append(AttGNN(self.hcs,self.hcs))
            
            self.decoder = MLP([N_scatter_feats*self.hcs,N_scatter_feats//2*self.hcs,self.hcs,N_outputs],no_final_act=True)
#             self.beta = torch.nn.Parameter(torch.ones(1)*10)
 
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

            CoC, CoC_edge_attr = self.return_CoC_and_edge_attr(x, batch)
            
            graph_ids, graph_node_counts = batch.unique(return_counts=True)
            
            x = torch.cat([x,
                           x_feature_constructor(x,graph_node_counts),
                           CoC_edge_attr,
                           CoC[batch]],dim=1)
            
#             att = scatter_softmax(graph_node_counts[batch].view(-1,1)/100*self.att(x),batch,dim=0)
#             att_d = scatter_distribution(att,batch,dim=0)
#             mask = att.squeeze() > att_d[batch,0] + att_d[batch,1]
# #             mask = att.squeeze() > att_d[batch,2] - 1000*att_d[batch,1]/graph_node_counts[batch]
            
#             x = x[mask]
#             batch = batch[mask]
#             graph_ids, graph_node_counts = batch.unique(return_counts=True)
            
            x = self.x_encoder(x)
            for conv in self.convs:
                x = conv(x,graph_node_counts)

            return self.decoder(scatter_distribution(x,batch,dim=0))
    return Net