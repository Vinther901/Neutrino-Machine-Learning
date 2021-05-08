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

            def gaussian(alpha):
                phi = torch.exp(-1*alpha.pow(2))
                return phi
            class RBF(torch.nn.Module):
                def __init__(self, in_features, out_features, basis_func=gaussian):
                    super(RBF, self).__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.centres = torch.nn.Parameter(torch.Tensor(out_features, in_features))
                    self.log_sigmas = torch.nn.Parameter(torch.Tensor(out_features))
#                     self.sigmas = torch.nn.Parameter(torch.Tensor(out_features))
                    self.basis_func = basis_func
                    self.reset_parameters()

                def reset_parameters(self):
                    torch.nn.init.normal_(self.centres, 0, 1)
                    torch.nn.init.constant_(self.log_sigmas, 0)
#                     torch.nn.init.constant_(self.sigmas, 10)

                def forward(self, x):
                    size = (x.size(0), self.out_features, self.in_features)
                    x = x.unsqueeze(1).expand(size)
                    c = self.centres.unsqueeze(0).expand(size)
                    distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
#                     distances = (x - c).pow(2).sum(-1) / self.sigmas.unsqueeze(0)
#                     print(distances.mean(), distances.std(), distances.min(), distances.max())
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
                def __init__(self, in_features, out_features,basis_func=gaussian):
                    super(RBF_scatter, self).__init__() 
                    self.RBF_batch_norm = torch.nn.BatchNorm1d(in_features)
                    self.RBF = RBF(in_features, out_features,basis_func)
                    self.sum_batch_norm = torch.nn.BatchNorm1d(out_features*N_scatter_feats)
                def forward(self, x, batch):
                    return self.sum_batch_norm(scatter_distribution(self.RBF(self.RBF_batch_norm(x)), batch, dim=0))

            N_x_feats = N_dom_feats# + 4*(N_dom_feats + 1)
            
#             self.RBF_scatter = RBF_scatter(N_x_feats,2*self.hcs, basis_func=gaussian)
            self.encoder = MLP([N_dom_feats,2*self.hcs])
            self.batch_norm = torch.nn.BatchNorm1d(N_scatter_feats*2*self.hcs)
            self.decoder = MLP([N_scatter_feats*2*self.hcs,5*self.hcs,self.hcs,N_outputs],clean_out=True)

        def forward(self,data):
            x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
            ###############
            x = x.float()
            ###############
#             x = self.RBF_scatter(x,batch)
            x = self.batch_norm(scatter_distribution(self.encoder(x),batch,dim=0))
            x = self.decoder(x)

            return x
    return Net