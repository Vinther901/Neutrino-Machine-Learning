def Load_model(name, args):
    from FunctionCollection import Loss_Functions, customModule
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
    
    print("if features = x, Charge should be at x[:,-5], time at x[:,-4] and pos at x[:,-3:]")
    assert N_dom_feats == len(args['features'].split(', '))
    
    crit, y_post_processor, output_post_processor, cal_acc = Loss_Functions(name, args)
    likelihood_fitting = True if name[-4:] == 'NLLH' else False
    
    class Net(customModule):
        def __init__(self):
            super(Net, self).__init__(crit, y_post_processor, output_post_processor, cal_acc, likelihood_fitting, args)

            self.act = torch.nn.SiLU()
            self.hcs = N_hcs

            def gaussian(alpha):
                phi = torch.exp(-1*alpha.pow(2))
                return phi
            class RBF(torch.nn.Module):
                def __init__(self, in_features, out_features, basis_func=gaussian):
                    super(RBF, self).__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.centres = torch.nn.Parameter(torch.Tensor(out_features, in_features))
#                     self.log_sigmas = torch.nn.Parameter(torch.Tensor(out_features))
                    self.sigmas = torch.nn.Parameter(torch.Tensor(out_features))
                    self.basis_func = basis_func
                    self.reset_parameters()

                def reset_parameters(self):
                    torch.nn.init.normal_(self.centres, 0, 1)
#                     torch.nn.init.constant_(self.log_sigmas, 2.3)
                    torch.nn.init.constant_(self.sigmas, 10)

                def forward(self, x):
                    size = (x.size(0), self.out_features, self.in_features)
                    x = x.unsqueeze(1).expand(size)
                    c = self.centres.unsqueeze(0).expand(size)
#                     distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
                    distances = (x - c).pow(2).sum(-1) / self.sigmas.unsqueeze(0)
#                     print(distances.mean(), distances.std(), distances.min(), distances.max())
#                     return self.basis_func(distances)
                    return self.basis_func(distances)
            
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
            
            class RBF_scatter(torch.nn.Module):
                def __init__(self, in_features, out_features):
                    super(RBF_scatter, self).__init__() 
                    self.RBF_batch_norm = torch.nn.BatchNorm1d(in_features)
                    self.RBF = RBF(in_features, out_features)
                    self.sum_batch_norm = torch.nn.BatchNorm1d(out_features)
                def forward(self, x, batch):
                    return self.sum_batch_norm(scatter_sum(self.RBF(self.RBF_batch_norm(x)), batch, dim=0))
            
            class GRUConv(torch.nn.Module):
                def __init__(self,hcs = self.hcs, act = self.act):
                    super(GRUConv, self).__init__()
                    self.act = act
                    self.hcs = hcs
                    self.GRU = torch.nn.GRUCell(self.hcs*2,self.hcs)
                    
                    self.lin_CoC_msg = MLP([N_scatter_feats*self.hcs, self.hcs],clean_out = True)
                    self.lin_CoC_self = MLP([self.hcs, self.hcs],clean_out = True)
                    
                    self.RBF_scatter = RBF_scatter(self.hcs, 4*self.hcs)
                    
                    self.CoC_batch_norm = torch.nn.BatchNorm1d(self.hcs)
                    
                    self.lin_x_msg = MLP([4*self.hcs, self.hcs],clean_out = True)
                    self.lin_x_self = MLP([self.hcs, self.hcs],clean_out = True)
                    
                    self.x_batch_norm = torch.nn.BatchNorm1d(self.hcs)
                
                def forward(self, x, CoC, h, batch):
                    h = self.GRU( torch.cat([CoC[batch], x], dim=1), h)
                    
                    msg = self.lin_CoC_msg( self.RBF_scatter(h, batch) )
                    CoC = self.lin_CoC_self(CoC)
                    
                    CoC = self.act( self.CoC_batch_norm(msg+CoC) )
                    
                    h = self.GRU( torch.cat([x, CoC[batch]], dim=1), h)
                    
                    msg = self.lin_x_msg(self.RBF_scatter.RBF(h))
                    x = self.lin_x_self(x)
                    
                    x = self.act( self.x_batch_norm(msg+x) )
                    return x, CoC, h

            N_x_feats = 2*(N_dom_feats + 4*(N_dom_feats + 1)) + N_dom_feats + 4*(N_dom_feats + 1) + 1 + 4 + 3
            self.N_x_to_CoC_feats = N_dom_feats + 4*(N_dom_feats + 1) + 4 + N_dom_feats + 4*(N_dom_feats + 1) + 1
            N_CoC_feats = 3+N_scatter_feats*(self.N_x_to_CoC_feats)
            
            self.x_encoder = MLP([N_x_feats,2*self.hcs,self.hcs])
#             self.att = MLP([N_x_feats,1])
            self.CoC_encoder = MLP([N_CoC_feats,2*self.hcs,self.hcs])

            self.convs = torch.nn.ModuleList()
            for i in range(N_metalayers):
                self.convs.append(GRUConv())
            
#             self.decoderRBF = RBF(self.hcs,4*self.hcs)
#             self.decoder = MLP([self.hcs,self.hcs,self.hcs,N_outputs],clean_out=True)
            self.decoder_RBF_scatter = RBF_scatter(self.hcs,4*self.hcs)
            self.decoder = MLP([(1+N_scatter_feats)*self.hcs,3*self.hcs,N_outputs],clean_out=True)
#             self.decoder = MLP([(1+N_scatter_feats)*self.hcs,self.hcs,self.hcs])
            
#             self.decoders = torch.nn.ModuleList()
#             for _ in range(N_outputs):
#                 self.decoders.append(torch.nn.Linear(self.hcs,1))


        def forward(self,data):
            x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
            ###############
            x = x.float()
            ###############
            
            time_edge_index = time_edge_indeces(x[:,-4],batch)
                                      
            time_edge_attr = self.edge_feature_constructor(x, time_edge_index)

            CoC, CoC_edge_attr = self.return_CoC_and_edge_attr(x, batch)
            
            x = torch.cat([x[time_edge_index[0]],
                           CoC[batch],
                           CoC_edge_attr,
                           time_edge_attr,
                           x],dim=1)
            
            CoC = torch.cat([CoC,scatter_distribution(x[:,-self.N_x_to_CoC_feats:],batch,dim=0)],dim=1)

            x = self.x_encoder(x)
            CoC = self.CoC_encoder(CoC)

            h = torch.zeros( (x.shape[0], self.hcs), device=self.device)

            for i in range(N_metalayers):
                x, CoC, h = self.convs[i](x, CoC, h, batch)
            
            CoC = torch.cat([CoC,self.decoder_RBF_scatter(x, batch)],dim=1)
#             CoC = torch.cat([CoC,scatter_distribution(x, batch, dim=0)],dim=1)

            CoC = self.decoder(CoC)
            
#             out = []
#             for mlp in self.decoders:
#                 out.append(mlp(CoC))
#             CoC = torch.cat(out,dim=1)
            
            return CoC
    
        def return_CoC_and_edge_attr(self, x, batch):
            pos = x[:,-3:]
            charge = x[:,-5].view(-1,1)
            
            # Define central nodes at Center of Charge:
            CoC = scatter_sum( pos*charge, batch, dim=0) / scatter_sum(charge, batch, dim=0)
            
            # Define edge_attr for those edges:
            cart = pos - CoC[batch]
            rho = torch.norm(cart, p=2, dim=1).view(-1, 1)
            rho_mask = rho.squeeze() != 0
            cart[rho_mask] = cart[rho_mask] / rho[rho_mask]
            CoC_edge_attr = torch.cat([cart.type_as(x),rho.type_as(x)], dim=1)
            return CoC, CoC_edge_attr
        
        def edge_feature_constructor(self, x, edge_index):
            (frm, to) = edge_index
            pos = x[:,-3:]
            cart = pos[frm] - pos[to]

            rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
            rho_mask = rho.squeeze() != 0
            cart[rho_mask] = cart[rho_mask] / rho[rho_mask]
            
            diff = x[to,:-3] - x[frm,:-3]

            return torch.cat([cart.type_as(pos),rho,diff], dim=1)
    return Net