def Load_model(name, args):
    from FunctionCollection import Loss_Functions, customModule
    import pytorch_lightning as pl
    
    from torch_geometric_temporal import nn
    
    from typing import Optional

    import torch
    from torch_scatter import scatter_sum, scatter_min, scatter_max
    from torch_scatter.utils import broadcast
    from torch_geometric.nn import GATConv

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

            self.x_encoder = torch.nn.Linear(N_x_feats,self.hcs)
            self.conv = nn.recurrent.temporalgcn.TGCN(self.hcs,self.hcs)
            self.decoder = torch.nn.Linear(self.hcs*4,N_outputs)
            
            self.GATConv = GATConv(self.hcs, self.hcs, 3, add_self_loops = False)
            
        def forward(self,data):
            x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch

            x = torch.cat([x,scatter_distribution(edge_attr,edge_index[1],dim=0)],dim=1)
#             x0 = x.clone()

            time_sort = torch.argsort(x[:,1])
            batch_time_sort = time_sort[torch.argsort(batch[time_sort])]
            time_edge_index = torch.cat([batch_time_sort[:-1].view(1,-1),batch_time_sort[1:].view(1,-1)],dim=0)
            graph_ids, graph_node_counts = batch.unique(return_counts=True)
            tmp_bool = torch.ones(time_edge_index.shape[1],dtype=bool)
            tmp_bool[(torch.cumsum(graph_node_counts,0) - 1)[:-1]] = False
            time_edge_index = time_edge_index[:,tmp_bool]
            time_edge_index = torch.cat([time_edge_index,time_edge_index.flip(0)],dim=1)

            x = self.act(self.x_encoder(x))
            
            x, (e, w) = self.GATConv(x, edge_index, return_attention_weights = True)
            return x, e, w
            print(x, e, w)
            
            h = self.act(self.conv(x, time_edge_index))
            for i in range(N_metalayers):
                h = self.act(self.conv(x,time_edge_index,H=h))
            x = scatter_distribution(h,batch,dim=0)
            x = self.decoder(x)
            
            return x
    return Net