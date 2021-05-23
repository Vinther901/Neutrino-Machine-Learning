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
            
            class FullConv(torch.nn.Module):
                def __init__(self, hcs_in, hcs_out, act = self.act):
                    super(FullConv, self).__init__()
                    self.self_mlp = MLP([hcs_in,hcs_out])
                    self.msg_mlp = torch.nn.Sequential(torch.nn.Linear(2*hcs_in,hcs_out), 
                                                       act)
                    self.msg_mlp2 = MLP([4*hcs_out,hcs_out])
                def forward(self, x, graph_node_counts):
                    li : List[int] = graph_node_counts.tolist()
                    tmp = []
                    for tmp_x in x.split(li):
                        tmp_x = torch.cat([tmp_x.unsqueeze(1) - tmp_x, tmp_x.unsqueeze(1) + tmp_x],dim=2)
                        tmp_x = self.msg_mlp(tmp_x)
                        tmp.append(torch.cat([tmp_x.mean(1),tmp_x.std(1),tmp_x.min(1)[0],tmp_x.max(1)[0]],dim=1))
                    return self.self_mlp(x) + self.msg_mlp2(torch.cat(tmp,0))
            
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
            
            self.convs = torch.nn.ModuleList()
            self.scatter_norms = torch.nn.ModuleList()
            for _ in range(N_metalayers):
                self.convs.append(FullConv(self.hcs,self.hcs))
                self.scatter_norms.append(scatter_norm(self.hcs))

            self.decoder = MLP([(1+N_scatter_feats*N_metalayers)*self.hcs,self.hcs,N_outputs],clean_out=True)

        def forward(self,data):
            x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
            ###############
            x = x.float()
            ###############

            CoC = scatter_sum( x[:,-3:]*x[:,-5].view(-1,1), batch, dim=0) / scatter_sum(x[:,-5].view(-1,1), batch, dim=0)
            CoC = torch.cat([CoC,self.scatter_norm(x,batch)],dim=1)

            x = self.x_encoder(x)
            CoC = self.CoC_encoder(CoC)
            
            graph_ids, graph_node_counts = batch.unique(return_counts=True)
            
            for conv, scatter_norm in zip(self.convs,self.scatter_norms):
                x = conv(x, graph_node_counts)
                CoC = torch.cat([CoC,scatter_norm(x, batch)],dim=1)
            
            CoC = self.decoder(CoC)

            return CoC
    return Net