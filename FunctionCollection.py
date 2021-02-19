### perhaps 1 target constructor is enough?
### Is this visible?
# def direction_target_constructor(dataset):
#     ##Offsetting with 2 because if I want positive values
#     x_dir = torch.tensor(tfs['truth']['direction_x'].inverse_transform(dataset.data.y.view(-1,10,1)[:,5]),dtype=torch.float)
#     y_dir = torch.tensor(tfs['truth']['direction_y'].inverse_transform(dataset.data.y.view(-1,10,1)[:,6]),dtype=torch.float)
#     z_dir = torch.tensor(tfs['truth']['direction_z'].inverse_transform(dataset.data.y.view(-1,10,1)[:,7]),dtype=torch.float)
#     dataset.data.y = torch.cat([x_dir,y_dir,z_dir],dim=1).flatten()
#     dataset.slices['y'] = np.arange(0,len(dataset.data.y)+1, 3)
#     return dataset

# def theta_target_constructor(dataset):
#     tfs = pd.read_pickle(path+'/train_test_datasets/transformers.pkl')
#     dataset.data.y = torch.tensor(tfs['truth']['zenith'].inverse_transform(dataset.data.y.view(-1,10,1)[:,9]),dtype=torch.float).flatten()
#     dataset.slices['y'] = np.arange(0,len(dataset.data.y)+1,1)
#     return dataset

# def energy_target_constructor(dataset):
#     tfs = pd.read_pickle(path+'/train_test_datasets/transformers.pkl')
#     dataset.data.y = torch.tensor(tfs['truth']['energy_log10'].inverse_transform(dataset.data.y.view(-1,10,1)[:,0]),dtype=torch.float).flatten()
#     dataset.slices['y'] = np.arange(0,len(dataset.data.y) + 1, 1)
#     return dataset

# def E_theta_target_constructor(dataset):
#     tfs = pd.read_pickle(path+'/train_test_datasets/transformers.pkl')
#     energy = torch.tensor(tfs['truth']['energy_log10'].inverse_transform(dataset.data.y.view(-1,10,1)[:,0]),dtype=torch.float)
#     theta = torch.tensor(tfs['truth']['zenith'].inverse_transform(dataset.data.y.view(-1,10,1)[:,9]),dtype=torch.float)
#     dataset.data.y = torch.cat([energy,theta],dim=1).flatten()
#     dataset.slices['y'] = np.arange(0,len(dataset.data.y)+1,2)
#     return dataset

# def periodic_target_constructor(dataset):
#     dataset.data.y = dataset.data.y.view(-1,10)[:,8:10].flatten()

#     tfs = pd.read_pickle(path+'/train_test_datasets/transformers.pkl')
#     az_ze = dataset.data.y.view(-1,2,1)
#     az = torch.tensor(tfs['truth']['azimuth'].inverse_transform(az_ze[:,0]),dtype=torch.float)
#     ze = torch.tensor(tfs['truth']['zenith'].inverse_transform(az_ze[:,1]),dtype=torch.float) #the range seems to be about [0,pi/2]?. Makes sense, Muons come from the atmosphere
    
#     dataset.data.y = torch.cat([az,ze],dim=1).flatten()
#     dataset.slices['y'] = np.arange(0,len(dataset.data.y)+1, 2)
#     return dataset

def custom_feature_constructor(dataset,transformer):
    #proper edge_index
    edge_ind = dataset.data.edge_index.clone()
    for i in range(dataset.__len__()):
        edge_ind[:,dataset.slices['edge_index'][i]:dataset.slices['edge_index'][i+1]] += dataset.slices['x'][i]
    
    (row, col) = edge_ind

    #Spherical
    tfs = transformer
    from math import pi as PI
    import torch
    pos = dataset.data.pos
    cart = pos[row] - pos[col]

    rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)

    # phi = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
    # phi = phi + (phi < 0).type_as(phi) * (2 * PI)

    # theta = torch.acos(cart[..., 2] / rho.view(-1)).view(-1, 1)
    # theta[rho == 0] = torch.zeros((rho == 0).sum())
    rho_mask = rho.squeeze() != 0
    cart[rho_mask] = cart[rho_mask] / rho[rho_mask]

    #"Normalize rho"
    rho = rho / 600 #leads to the interval ~[0,2.25].. atleast for muon_100k_set11_SRT

    #normalize pos
    dataset.data.pos = pos / 300 #leads to absolute sizes of ~1.5-2
    dataset.data.x[:,-3:] = dataset.data.pos

    #Time difference and charge ratio
    T_diff = dataset.data.x[col,1] - dataset.data.x[row,1]
    Q_diff = dataset.data.x[col,0] - dataset.data.x[row,0]
    
    dataset.data.edge_attr = torch.cat([cart.type_as(pos),rho,T_diff.view(-1,1),Q_diff.view(-1,1)], dim=-1)
    dataset.slices['edge_attr'] = dataset.slices['edge_index']

    return dataset

def dataset_preparator(name, path, transformer, tc = None, fc = None, shuffle = True, TrTV_split = (1,0,0), batch_size = 512):
    from torch_geometric.data import DataLoader, InMemoryDataset, DataListLoader
    import torch
    from datetime import datetime
    transformer = transformer
    class LoadDataset(InMemoryDataset):
        def __init__(self, name, path=str(), reload_data = None):
            super(LoadDataset, self).__init__(path)
            if reload_data is not None:
                (self.data, self.slices) = reload_data
            else:
                self.data, self.slices = torch.load(path + '/' + name)  

        @property
        def processed_file_names(self):
            return os.listdir(path)

        def reload(self):
            for data_list in DataListLoader(self,batch_size=self.__len__()):
                pass
            return LoadDataset(name=None, reload_data = self.collate(data_list))
    
    print(f"{datetime.now()}: loading data..")
    dataset = LoadDataset(name,path)
    
    print(f"{datetime.now()}: executing target constructor..")
    if tc is not None: # tc is target constructor, callable
        dataset = tc(dataset,transformer)
    
    print(f"{datetime.now()}: executing feature constructor..")
    if fc is not None: # fc is feature constructor, callable
        dataset = fc(dataset,transformer)
    
    if shuffle:
        print(f"{datetime.now()}: shuffling dataset..")
        dataset = dataset.shuffle()
    
    length = dataset.__len__()
    
    print(f"{datetime.now()}: defining dataloaders..")
    train_loader = DataLoader(dataset[:int(length*TrTV_split[0])], batch_size, shuffle=True) if TrTV_split[0] != 0 else None
    test_loader = DataLoader(dataset[int(length*TrTV_split[0]):int(length*TrTV_split[1])], batch_size, shuffle=False)
    val_loader = DataLoader(dataset[int(length*TrTV_split[1]):int(length*TrTV_split[2])], batch_size, shuffle=False)

    print(f"{datetime.now()}: Done!")
    return dataset, train_loader, test_loader, val_loader

def return_reco_truth(model,loader):
    from tqdm import tqdm
    from torch import no_grad, device
    from torch.cuda import empty_cache
    from numpy import array
    outputs = []
    labels = []
    model.eval()
    with no_grad():
        progress_bar = tqdm(total=loader.__len__(), desc='Batch', position=0)
        for data in loader:
            data = data.to(device('cuda' if next(model.parameters()).device.type == 'cuda' else 'cpu'))
            outputs += model(data).tolist()
            labels += data.y.view(data.num_graphs,-1).tolist()
            del data
            progress_bar.update(1)
        empty_cache()
    return array(outputs), array(labels)

def performance_plot(res, x, bins=10, zero_bounded=False):
    import matplotlib.pyplot as plt
    import numpy as np
    x = x.flatten()
    slices = np.linspace(x.min(),x.max(),bins + 1)
    print(slices,x.min(),x.max())

    quantiles = np.zeros((bins,3))
    xes = np.zeros(bins)

    for i in range(bins):
        mask = (x >= slices[i])&(x <= slices[i+1])
        if zero_bounded:
            quantiles[i] = np.quantile(res[mask],(0,0.5,0.68))
        else:
            quantiles[i] = np.quantile(res[mask],(0.25,0.5,0.75))
        xes[i] = np.mean(x[mask])
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.errorbar(x = xes, y = quantiles[:,1], yerr = abs(quantiles[:,1] - quantiles[:,[0,2]].T),fmt='none')
    ax.plot(xes,quantiles[:,1],'k.')
    ax.set(xlim=(slices[0] - 0.5, slices[-1] + 0.5))
    ax.hlines(0,slices[0],slices[-1])                 
    ax_hist = ax.twinx()
    ax_hist.hist(x,bins=bins,histtype='step',color='grey')
    ax.set_zorder(ax_hist.get_zorder()+1)
    ax.patch.set_visible(False)
    plt.grid()
    fig.show()

    fig2, ax2 = plt.subplots(figsize=(10,7))
    colors = ['r','w','r']
    for i in range(quantiles.shape[1]):
        ax2.plot(xes,quantiles[:,i],f'{colors[i]}--')
    hist2d = ax2.hist2d(x,res,bins=200)
    ax2.set_ylim(np.quantile(res,(0.1,0.9)))
    fig2.colorbar(hist2d[-1],ax=ax2)
    plt.grid()
    fig2.show()

    return quantiles

def lambda_lr(epoch,epoch_warmup=2,maximum=25,a=0.05,q=0.2):
    if epoch < epoch_warmup:
        return a**(1-epoch/epoch_warmup)
    elif epoch < maximum*epoch_warmup:
        return 1/(1 + (1/q-1)*(epoch/epoch_warmup - 1)/(maximum-1))
    else:
        return q

def Loss_Functions(name, args = None):
    '''
    Returns the specified loss function, accompanied by the proper "shape modifiers",
    as defined by y_post_processor and output_post_processor
    '''
    assert name in ['Gaussian_NLLH',
                    'Spherical_NLLH',
                    'Polar_NLLH',
                    'MSE',
                    'Cross_Entropy']
    print("Remember all accuracies are positive and defined to go towards 0 in the optimal case.")
##############################################################################################################    
    if name == 'Gaussian_NLLH':
        from torch import mean, matmul, sub, inverse, logdet
        def Gaussian_NLLH(mean, cov, label):
            loss = mean( matmul( sub(label,mean).unsqueeze(1), matmul( inverse(cov), sub(label,mean).unsqueeze(2) ) ) + logdet(cov) )
            return loss

        assert 'diagonal_cov' in args, "Specify bool of 'diagonal_cov' in the dictionary 'args'."
        assert 'N_targets' in args, "Specify 'N_targets' in the dictionary 'args'."
        
        def y_post_processor(y):
            return y.view(-1, args['N_targets'])
        if args['diagonal_cov']:
            print("This might not be a proper implementation, yet, since the covariances are explicitly given.")
            def output_post_processor(output):
                from torch import tril_indices, zeros, square 
                (row,col) = tril_indices(row=args['N_targets'], col=args['N_targets'], offset=-1)
                
                assert 'device' in args, "Specify 'device' (torch.device('cpu' or 'cuda')) in the dictionary 'args'."
                tmp = zeros( (output.shape[0],args['N_targets'],args['N_targets']) ).to(args['device'])
                tmp[:, row, col] = output[:,2*args['N_targets']:]
                tmp[:, col, row] = output[:,2*args['N_targets']:]
                tmp[:,[i for i in range(args['N_targets'])],[i for i in range(args['N_targets'])]] = square(output[:,args['N_targets']:2*args['N_targets']])
                
                return output[:,:args['N_targets']], tmp
        else:
            def output_post_processor(output):
                from torch import zeros, square 
                assert 'device' in args, "Specify 'device' (torch.device('cpu' or 'cuda')) in the dictionary 'args'."
                tmp = zeros( (output.shape[0],args['N_targets'],args['N_targets']) ).to(args['device'])
                tmp[:,[i for i in range(args['N_targets'])],[i for i in range(args['N_targets'])]] = square(output[:,args['N_targets']:2*args['N_targets']])
                
                return output[:,:args['N_targets']], tmp
            def cal_acc(pred,label):
                return (pred.view(-1) - label.view(-1)).float().abs().mean().item()
            
        return Gaussian_NLLH, y_post_processor, output_post_processor, cal_acc
##############################################################################################################
##############################################################################################################
    elif name == 'Spherical_NLLH':
        from torch import mean, cos, sin, abs, log, exp, square
        def Spherical_NLLH(pred, kappa, label):
            azm = pred[:,0] #Azimuth prediction
            azl = label[:,0] #Azimuth target
            zem = pred[:,1] #Zentih prediction
            zel = label[:,1] #Zentih target
            s1 = torch.sin( zel + azl - azm )
            s2 = torch.sin( zel - azl + azm )
            c1 = torch.cos( zel - zem )
            c2 = torch.cos( zel - zem )
            cos_diff_angle = 0.5*abs(sin(zem))*( s1 + s2 ) + 0.5*(c1 + c2)
            
            nlogC = - log(kappa) + kappa + log( 1 - exp( - 2 * kappa ) )
            
            loss = mean( - kappa*cos_diff_angle + nlogC )
            return loss
        
        assert 'N_targets' in args, "Specify 'N_targets' in the dictionary 'args'."
        def y_post_processor(y):
            return y.view(-1, args['N_targets'])
        def output_post_processor(output):
            return output[:,:2], square(output[:,2])
        def cal_acc(pred,label):
            azm = pred[:,0] #Azimuth prediction
            azl = label[:,0] #Azimuth target
            zem = pred[:,1] #Zentih prediction
            zel = label[:,1] #Zentih target

            s1 = torch.sin( zel + azl - azm )
            s2 = torch.sin( zel - azl + azm )
            c1 = torch.cos( zel - zem )
            c2 = torch.cos( zel - zem )
            cos_diff_angle = 0.5*torch.abs(torch.sin(zem))*( s1 + s2 ) + 0.5*(c1 + c2)

            return 1 - cos_diff_angle.float().mean().item()
        return Spherical_NLLH, y_post_processor, output_post_processor, cal_acc
##############################################################################################################
##############################################################################################################
    elif name == 'Polar_NLLH':
        from torch import mean, cos, multiply, sub, abs, log, exp, square
        def Polar_NLLH(pred, kappa, label):
            lnI0 = kappa + log(1 + exp(-2*kappa)) -0.25*log(1+0.25*square(kappa)) + log(1+0.24273*square(kappa)) - log(1+0.43023*square(kappa))
            loss = mean( - multiply(kappa,cos(sub(label,pred))) + lnI0 )
            return loss
        
        assert 'zenith' in args, "Specify the bool 'zenith' in the dictionary 'args'."
        def y_post_processor(y):
            return y
        if args['zenith']:
            def output_post_processor(output):
                return abs(output[:,0]), square(output[:,1])
        else:
            def output_post_processor(output):
                return output[:,0], square(output[:,1])
        def cal_acc(output,label):
            return 1 - torch.cos(output - label).float().mean().item()
        
        return Spherical_NLLH, y_post_processor, output_post_processor, cal_acc
############################################################################################################## 
##############################################################################################################
    elif name == 'MSE':
        from torch import mean
        def MSE(output,label):
            loss = mean( (output - label)**2 )
            return loss
        
        assert 'N_targets' in args, "Specify 'N_targets' in the dictionary 'args'."
        def y_post_processor(y):
            return y.view(-1, args['N_targets'])
        def output_post_processor(output):
            return output
        def cal_acc(output,label):
            return (output.view(-1) - label.view(-1)).float().abs().mean().item()
        
        return MSE, y_post_processor, output_post_processor, cal_acc
##############################################################################################################
##############################################################################################################
    elif name == 'Cross_Entropy':
        from torch.nn import CrossEntropyLoss
       
        def y_post_processor(y):
            return y
        def output_post_processor(output):
            return output
        def cal_acc(output,label):
            return 1 - output.argmax(dim=1).eq(label).float().mean().item()
        
        return CrossEntropyLoss(), y_post_processor, output_post_processor, cal_acc
##############################################################################################################
    

# def train(model, loss_name, train_loader, val_loader, args):
    
#     crit, y_post_processor, output_post_processor, cal_acc = Loss_Functions(loss_name, args)
    
#     likelihood_fitting = True if loss_name[-4:] == 'NLLH' else False
#     val_len = val_loader.__len__(); best_acc = np.inf
    
#     model.train()
#     for data in train_loader:
#         data = data.to(args['device'])
#         label = y_post_processor(data.y)
#         optimizer.zero_grad()
        
#         if likelihood_fitting:
#             output, cov = output_post_processor( model(data) )
#             loss = crit(output, cov, label)
#         else:
#             output = output_post_processor( model(data) )
#             loss = crit(output, label)
            
#         loss.backward()
#         optimizer.step()
        
#         del data
        
#         acc = cal_acc(output,label)
        
#         if args['wandb_activated']:
#             wandb.log({"Train Loss": loss.item(),
#                        "Train Acc": acc})
    
#         torch.cuda.empty_cache()
#         model.eval()
#         with torch.no_grad()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    