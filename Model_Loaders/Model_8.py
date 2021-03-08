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

            N_x_feats = 2*N_dom_feats + N_edge_feats
            

            self.x_encoder = torch.nn.Linear(N_x_feats,self.hcs)
            
#             class Conversation(torch.nn.Module):
#                 def __init__(self,hcs,act):
#                     super(Conversation, self).__init__()
#                     self.act = act
#                     self.hcs = hcs
#                     self.GRU = torch.nn.GRUCell(3*self.hcs,self.hcs)
#                     self.lin_msg1 = torch.nn.Linear((1+N_scatter_feats)*self.hcs,self.hcs)
#                     self.lin_msg2 = torch.nn.Linear((1+N_scatter_feats)*self.hcs,self.hcs)
                
#                 def forward(self, x, edge_index, edge_attr, batch, h):
#                     (frm, to) = edge_index
                    
#                     h = self.act( self.GRU( torch.cat([x[to],x[frm],edge_attr],dim=1), h ) )
#                     x = self.act( self.lin_msg1( torch.cat([x,scatter_distribution(h, to, dim=0)],dim=1) ) )
                    
#                     h = self.act( self.GRU( torch.cat([x[frm],x[to],edge_attr],dim=1), h) )
#                     x = self.act( self.lin_msg2( torch.cat([x,scatter_distribution(h, frm, dim=0)],dim=1) ) )
#                     return x
                  
            class GRUConv(torch.nn.Module):
                def __init__(self,hcs,act):
                    super(GRUConv, self).__init__()
                    self.hcs = hcs 
                    self.act = act
                    
                    self.GRU = torch.nn.GRUCell(self.hcs, self.hcs)
                    self.lin = torch.nn.Linear(2*self.hcs, self.hcs)
                    
                def forward(self, x, edge_index, edge_attr, batch, h):
                    (frm, to) = edge_index
                    h = self.act( self.GRU( x, h ) )
                    x = torch.cat([x,h[frm]], dim=1)
                    x = self.act( self.lin( x ) )
                    return x

            self.GRUConvs = torch.nn.ModuleList()
#             self.ConvConvs = torch.nn.ModuleList()
            for i in range(N_metalayers):
                self.GRUConvs.append(GRUConv(self.hcs,self.act))
#                 self.ConvConvs.append(Conversation(self.hcs,self.act)

            self.decoders = torch.nn.ModuleList()
            self.decoders.append(torch.nn.Linear(4*self.hcs,self.hcs))
            self.decoders.append(torch.nn.Linear(self.hcs,self.hcs))

            self.decoder = torch.nn.Linear(self.hcs,N_outputs)


        def forward(self,data):
            x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch

            time_sort = torch.argsort(x[:,1])
            graph_ids, graph_node_counts = batch.unique(return_counts=True)
            batch_time_sort = torch.cat( [time_sort[batch[time_sort] == i] for i in graph_ids] )
            time_edge_index = torch.cat([batch_time_sort[:-1].view(1,-1),batch_time_sort[1:].view(1,-1)],dim=0)

            tmp_ind = (torch.cumsum(graph_node_counts,0) - 1)[:-1]
            time_edge_index[1,tmp_ind] = time_edge_index[0, [0] + (tmp_ind + 1).tolist()[:-1]]
            time_edge_index = torch.cat([time_edge_index, 
                                         torch.cat([time_edge_index[1,-1].view(1,1),
                                                    time_edge_index[0,(tmp_ind + 1)[-1]].view(1,1)])],dim=1)

            time_edge_index = time_edge_index[:,torch.argsort(time_edge_index[1])]
                                      
            edge_attr = edge_feature_constructor(x, time_edge_index)
            
            x = torch.cat([x,edge_attr,x[time_edge_index[0]]],dim=1)
                                      
            x = self.act(self.x_encoder(x))

            h = torch.zeros( (x.shape[0], self.hcs) ).type_as(x)

            for i, conv in enumerate(self.GRUConvs):
                x = conv(x, time_edge_index, edge_attr, batch, h)
            
            out = scatter_distribution(x,batch,dim=0)
            
            for lin in self.decoders:
                out = self.act(lin(out))

            out = self.decoder(out)
            return out
    return Net