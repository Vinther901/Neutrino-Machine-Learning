### perhaps 1 target constructor is enough?
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

def custom_feature_constructor(dataset):
    #proper edge_index
    edge_ind = dataset.data.edge_index.clone()
    for i in range(dataset.__len__()):
        edge_ind[:,dataset.slices['edge_index'][i]:dataset.slices['edge_index'][i+1]] += dataset.slices['x'][i]
    
    (row, col) = edge_ind

    #Spherical
    tfs = pd.read_pickle(path+'/train_test_datasets/transformers.pkl')
    from math import pi as PI
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

def dataset_preparator(name, path, tc = None, fc = None, shuffle = True, TrTV_split = (1,0,0), batch_size = 512):
    from torch_geometric.data import DataLoader, InMemoryDataset, DataListLoader
    import torch
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
    dataset = LoadDataset(name)

    print(f"{datetime.now()}: executing target constructor..")
    if tc is not None: # tc is target constructor, callable
        dataset = tc(dataset)
    
    print(f"{datetime.now()}: executing feature constructor..")
    if fc is not None: # fc is feature constructor, callable
        dataset = fc(dataset)
    
    if shuffle:
        print(f"{datetime.now()}: shuffling dataset..")
        dataset.shuffle()
    
    length = dataset.__len__()
    
    print(f"{datetime.now()}: defining dataloaders..")
    train_loader = DataLoader(dataset[:int(length*TrTV_split[0])], batch_size, shuffle=True) if TrTV_split[0] != 0 else None
    test_loader = DataLoader(dataset[int(length*TrTV_split[0]):int(length*TrTV_split[1])], batch_size, shuffle=False)
    val_loader = DataLoader(dataset[int(length*TrTV_split[1]):int(length*TrTV_split[2])], batch_size, shuffle=False)

    print(f"{datetime.now()}: Done!")
    return dataset, train_loader, test_loader, val_loader

def return_reco_truth(model,loader):
    from torch import no_grad
    from torch.cuda import empty_cache
    from numpy import array
    outputs = []
    labels = []
    model.eval()
    with no_grad():
        for data in loader:
            labels += data.y.view(-1,N_targets).tolist()
            data = data.to(device)
            outputs += model(data).tolist()
            del data
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