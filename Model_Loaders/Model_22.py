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
        summ = tmp.clone()
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

        return torch.cat([summ,mean,var,maximum,minimum],dim=1)
    
    N_edge_feats = args['N_edge_feats'] #6
    N_dom_feats = args['N_dom_feats']#6
    N_scatter_feats = 5
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
            
#             class GRUConv(torch.nn.Module):
#                 def __init__(self,hcs = self.hcs, act = self.act):
#                     super(GRUConv, self).__init__()
#                     self.act = act
#                     self.hcs = hcs
#                     self.GRU = torch.nn.GRUCell(self.hcs*2,self.hcs)
                    
#                     self.scatter_norm = scatter_norm(self.hcs)
#                     self.lin_CoC_msg = MLP([N_scatter_feats*self.hcs, self.hcs],clean_out = True)
#                     self.lin_CoC_self = MLP([self.hcs, self.hcs],clean_out = True)
                    
#                     self.CoC_batch_norm = torch.nn.BatchNorm1d(self.hcs)
                    
#                     self.lin_x_msg = MLP([self.hcs, self.hcs],clean_out = True)
#                     self.lin_x_self = MLP([self.hcs, self.hcs],clean_out = True)
                    
#                     self.x_batch_norm = torch.nn.BatchNorm1d(self.hcs)

#                 def forward(self, x, CoC, h, batch):
#                     h = self.act( self.GRU( torch.cat([CoC[batch], x], dim=1), h) )
                    
#                     msg = self.lin_CoC_msg( self.scatter_norm(h, batch) )
#                     CoC = self.lin_CoC_self(CoC)
                    
#                     CoC = self.act( self.CoC_batch_norm(msg+CoC) )
                    
#                     h = self.act( self.GRU( torch.cat([x, CoC[batch]], dim=1), h) )
                    
#                     msg = self.lin_x_msg(h)
#                     x = self.lin_x_self(x)
                    
#                     x = self.act( self.x_batch_norm(msg+x) )
#                     return x, CoC, h
            
#             class AttConv(torch.nn.Module):
#                 def __init__(self,in_hcs = [self.hcs, self.hcs], out_hcs = self.hcs, heads = 1):
#                     super(AttConv,self).__init__()
                    
#                     self.heads = heads
#                     self.out_hcs = out_hcs
                    
#                     self.lin_key = torch.nn.Linear(in_hcs[0], heads*out_hcs)
#                     self.lin_query = torch.nn.Linear(in_hcs[1], heads*out_hcs)
#                     self.lin_value = torch.nn.Linear(in_hcs[0], heads*out_hcs)
                    
#                     self.sqrt_d = torch.sqrt(out_hcs)
                    
#                     self.reset_parameters()
                    
#                 def reset_parameters(self):
#                     self.lin_key.reset_parameters()
#                     self.lin_query.reset_parameters()
#                     self.lin_value.reset_parameters()
                
#                 def forward(self, x, CoC, batch):
#                     key = self.lin_key(x).view(-1,self.heads,self.out_hcs)
#                     query = self.lin_query(CoC
            
            class scatter_norm(torch.nn.Module):
                def __init__(self, hcs):
                    super(scatter_norm, self).__init__()
                    self.batch_norm = torch.nn.BatchNorm1d(N_scatter_feats*hcs)
                def forward(self, x, batch):
                    return self.batch_norm(scatter_distribution(x,batch,dim=0))


            N_x_feats = N_dom_feats# + 4*(N_dom_feats + 1)
            N_CoC_feats = N_scatter_feats*N_x_feats + 3

            self.scatter_norm = scatter_norm(N_x_feats)
            self.x_encoder = MLP([N_x_feats,self.hcs])
            self.CoC_encoder = MLP([N_CoC_feats,self.hcs])
            
            self.TConv = torch_geometric.nn.TransformerConv(in_channels = [self.hcs,self.hcs],
                                                            out_channels = self.hcs,
                                                            heads = N_metalayers)
            
            self.decoder = MLP([(N_metalayers)*self.hcs,3*self.hcs,self.hcs,N_outputs],clean_out=True)

        def forward(self,data):
            x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
            ###############
            x = x.float()
            ###############

            CoC = scatter_sum( x[:,-3:]*x[:,-5].view(-1,1), batch, dim=0) / scatter_sum(x[:,-5].view(-1,1), batch, dim=0)
            CoC = torch.cat([CoC,self.scatter_norm(x,batch)],dim=1)

            x = self.x_encoder(x)
            CoC = self.CoC_encoder(CoC)
            
            CoC_x = torch.cat([CoC,x],dim=0)
            
            edge_index = self.return_edge_index(batch)
            
            CoC_x = self.TConv(CoC_x, edge_index)

            CoC = self.decoder(CoC_x[batch.unique()])

            return CoC
        
        def return_edge_index(self,batch):
            offset = batch.max() + 1
            frm = torch.arange(offset, offset + batch.shape[0],dtype=torch.long).view(1,-1)
            to = batch.view(1,-1)
            return torch.cat([frm,to],dim=0).contiguous()
    return Net