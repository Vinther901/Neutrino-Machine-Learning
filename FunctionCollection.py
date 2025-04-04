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

def edge_feature_constructor(x, edge_index):
    (frm, to) = edge_index
    pos = x[:,-3:]
    cart = pos[frm] - pos[to]
    
    rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
    rho_mask = rho.squeeze() != 0
    cart[rho_mask] = cart[rho_mask] / rho[rho_mask]
    
    #Time difference and charge ratio
    T_diff = x[to,1] - x[frm,1]
    Q_diff = x[to,0] - x[frm,0]
    
    edge_attr = torch.cat([cart.type_as(pos),rho,T_diff.view(-1,1),Q_diff.view(-1,1)], dim=1)
    
    return edge_attr
    

def dataset_feature_constructor(dataset,transformer):
    #proper edge_index
    edge_ind = dataset.data.edge_index.clone()
    for i in range(dataset.__len__()):
        edge_ind[:,dataset.slices['edge_index'][i]:dataset.slices['edge_index'][i+1]] += dataset.slices['x'][i]
    
    dataset.data.x[:,-3:] /= 300
    dataset.data.pos = dataset.data.x[:,-3:]
    
    dataset.data.edge_attr = edge_feature_constructor(dataset.data.x, edge_ind)
    dataset.slices['edge_attr'] = dataset.slices['edge_index']
    return dataset
    
    
#     (row, col) = edge_ind

#     #Spherical
#     tfs = transformer
#     from math import pi as PI
#     import torch
#     pos = dataset.data.pos
#     cart = pos[row] - pos[col]

#     rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)

#     # phi = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
#     # phi = phi + (phi < 0).type_as(phi) * (2 * PI)

#     # theta = torch.acos(cart[..., 2] / rho.view(-1)).view(-1, 1)
#     # theta[rho == 0] = torch.zeros((rho == 0).sum())
#     rho_mask = rho.squeeze() != 0
#     cart[rho_mask] = cart[rho_mask] / rho[rho_mask]

#     #"Normalize rho"
#     rho = rho / 600 #leads to the interval ~[0,2.25].. atleast for muon_100k_set11_SRT

#     #normalize pos
#     dataset.data.pos = pos / 300 #leads to absolute sizes of ~1.5-2
#     dataset.data.x[:,-3:] = dataset.data.pos

#     #Time difference and charge ratio
#     T_diff = dataset.data.x[col,1] - dataset.data.x[row,1]
#     Q_diff = dataset.data.x[col,0] - dataset.data.x[row,0]
    
#     dataset.data.edge_attr = torch.cat([cart.type_as(pos),rho,T_diff.view(-1,1),Q_diff.view(-1,1)], dim=-1)
#     dataset.slices['edge_attr'] = dataset.slices['edge_index']

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
                    'twice_Polar_NLLH',
                    'MSE',
                    'MSE+MAE',
                    'Cross_Entropy']
    print("Remember all accuracies are positive and defined to go towards 0 in the optimal case.")
    
    def cos_diff_angle(azp,zep,azt,zet):
        from torch import sin, cos, abs
        term1 = abs(sin(zep))*cos(azp)*sin(zet)*cos(azt)
        term2 = abs(sin(zep))*sin(azp)*sin(zet)*sin(azt)
        term3 = cos(zep)*cos(zet)
        return term1 + term2 + term3

##############################################################################################################    
    if name == 'Gaussian_NLLH':
        from torch import mean, matmul, sub, inverse, logdet
        def Gaussian_NLLH(pred, cov, label):
            loss = mean( matmul( sub(label,pred).unsqueeze(1), matmul( inverse(cov), sub(label,pred).unsqueeze(2) ) ) + logdet(cov) )
            return loss

        assert 'diagonal_cov' in args, "Specify bool of 'diagonal_cov' in the dictionary 'args'."
        assert 'N_targets' in args, "Specify 'N_targets' in the dictionary 'args'."
        
        def y_post_processor(y):
            return y.view(-1, args['N_targets'])
        if not args['diagonal_cov']:
            print("This might not be a proper implementation, yet, since the covariances are explicitly given.")
            def output_post_processor(output):
                from torch import tril_indices, zeros, square 
                (row,col) = tril_indices(row=args['N_targets'], col=args['N_targets'], offset=-1)

                tmp = zeros( (output.shape[0],args['N_targets'],args['N_targets']) ).type_as(output)
                tmp[:, row, col] = output[:,2*args['N_targets']:]
                tmp[:, col, row] = output[:,2*args['N_targets']:]
                tmp[:,[i for i in range(args['N_targets'])],[i for i in range(args['N_targets'])]] = square(output[:,args['N_targets']:2*args['N_targets']])
                
                return output[:,:args['N_targets']], tmp
        else:
            assert 'eps' in args, "Specify 'eps' in the dictionary 'args'."
            def output_post_processor(output):
                from torch import zeros, square
                
                tmp = zeros( (output.shape[0],args['N_targets'],args['N_targets']) ).type_as(output)
                tmp[:,[i for i in range(args['N_targets'])],[i for i in range(args['N_targets'])]] = square(output[:,args['N_targets']:2*args['N_targets']])
                
                return output[:,:args['N_targets']], tmp + args['eps']
        def cal_acc(pred,label):
            return (pred.view(-1) - label.view(-1)).float().abs().mean()
            
        return Gaussian_NLLH, y_post_processor, output_post_processor, cal_acc
##############################################################################################################
##############################################################################################################
    elif name == 'Spherical_NLLH':
        from torch import mean, cos, sin, abs, log, exp, square
        def Spherical_NLLH(pred, kappa, label, weight):
            azp = pred[:,0] #Azimuth prediction
            azt = label[:,0] #Azimuth target
            zep = pred[:,1] #Zenith prediction
            zet = label[:,1] #Zenith target
            s1 = sin( zet + azt - azp )
            s2 = sin( zet - azt + azp )
            c1 = cos( zet - zep )
            c2 = cos( zet + zep )
            cos_angle = 0.5*abs(sin(zep))*( s1 + s2 ) + 0.5*(c1 + c2)
#             cos_angle = cos_diff_angle(azp,zep,azt,zet)
            
            nlogC = - log(kappa) + kappa + log( 1 - exp( - 2 * kappa ) )
            
            loss = mean( - kappa*cos_angle + nlogC )
            return loss
        
#         assert 'N_targets' in args, "Specify 'N_targets' in the dictionary 'args'."
        def y_post_processor(y):
            return y.view(-1, 2)
        assert 'eps' in args, "Specify 'eps' in the dictionary 'args'."
        def output_post_processor(output):
            return output[:,:2] + torch.tensor(args['output_offset']).type_as(output), square(output[:,2]) + args['eps']#(square(output[:,2]) + args['eps'])**(-1)
        def cal_acc(pred,label):
            azp = pred[:,0] #Azimuth prediction
            azt = label[:,0] #Azimuth target
            zep = pred[:,1] #Zenith prediction
            zet = label[:,1] #Zenith target

            s1 = sin( zet + azt - azp )
            s2 = sin( zet - azt + azp )
            c1 = cos( zet - zep )
            c2 = cos( zet + zep )
            cos_angle = 0.5*abs(sin(zep))*( s1 + s2 ) + 0.5*(c1 + c2)
#             cos_angle = cos_diff_angle(azp,zep,azt,zet)

            return (1 - cos_angle.float()).mean()
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
            return (1 - cos(output - label).float()).mean()
        
        return Polar_NLLH, y_post_processor, output_post_processor, cal_acc
##############################################################################################################
##############################################################################################################
    elif name == 'twice_Polar_NLLH':
        from torch.nn.functional import relu
        from torch import mean, cos, sin, multiply, sub, abs, log, exp, square
        def twice_Polar_NLLH(pred, kappa, label, weight):
            lnI0_az = kappa[:,0] + log(1 + exp(-2*kappa[:,0])) -0.25*log(1+0.25*square(kappa[:,0])) + log(1+0.24273*square(kappa[:,0])) - log(1+0.43023*square(kappa[:,0]))
            lnI0_ze = kappa[:,1] + log(1 + exp(-2*kappa[:,1])) -0.25*log(1+0.25*square(kappa[:,1])) + log(1+0.24273*square(kappa[:,1])) - log(1+0.43023*square(kappa[:,1]))
            az_correction = 0#100*relu(abs(pred[:,0] - label[:,0]) - 3.14)
            
            if torch.is_tensor(weight):
                loss = mean(weight*( - multiply(kappa[:,0],cos(sub(label[:,0],pred[:,0]))) - multiply(kappa[:,1],cos(sub(label[:,1],abs(pred[:,1])))) + lnI0_az + lnI0_ze + az_correction))/weight.sum()
            else:
                loss = mean( - multiply(kappa[:,0],cos(sub(label[:,0],pred[:,0]))) - multiply(kappa[:,1],cos(sub(label[:,1],abs(pred[:,1])))) + lnI0_az + lnI0_ze + az_correction)
            return loss
        
        def y_post_processor(y):
            return y.view(-1, 2)
        def output_post_processor(output):
            return output[:,:2] + torch.tensor(args['output_offset']).type_as(output), square(output[:,2:]) + args['eps']#torch.cat([square(output[:,2]).view(-1,1), square(output[:,3]).view(-1,1)],dim=1) + args['eps']
        def cal_acc(output,label):
            azp = output[:,0] #Azimuth prediction
            azt = label[:,0] #Azimuth target
            zep = output[:,1] #Zenith prediction
            zet = label[:,1] #Zenith target

            s1 = sin( zet + azt - azp )
            s2 = sin( zet - azt + azp )
            c1 = cos( zet - zep )
            c2 = cos( zet + zep )
            cos_angle = 0.5*abs(sin(zep))*( s1 + s2 ) + 0.5*(c1 + c2)
#             cos_angle = cos_diff_angle(azp,zep,azt,zet)

            return (1 - cos_angle.float()).mean()
#             return ((2 - cos(output[:,0] - label[:,0]).float() - cos(abs(output[:,1]) - label[:,1]).float())*0.5).mean().item()
        
        return twice_Polar_NLLH, y_post_processor, output_post_processor, cal_acc
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
            return (output.view(-1) - label.view(-1)).float().abs().mean()
        
        return MSE, y_post_processor, output_post_processor, cal_acc
##############################################################################################################
##############################################################################################################
    elif name == 'MSE+MAE':
        from torch import mean, abs
        def MSE_MAE(output,label):
            loss = mean( (output - label)**2 + abs(output - label) )
            return loss
        
        assert 'N_targets' in args, "Specify 'N_targets' in the dictionary 'args'."
        def y_post_processor(y):
            return y.view(-1, args['N_targets'])
        def output_post_processor(output):
            return output
        def cal_acc(output,label):
            return (output.view(-1) - label.view(-1)).float().abs().mean()
        
        return MSE_MAE, y_post_processor, output_post_processor, cal_acc
##############################################################################################################
##############################################################################################################
    elif name == 'Cross_Entropy':
        from torch.nn import CrossEntropyLoss
       
        def y_post_processor(y):
            return y
        def output_post_processor(output):
            return output
        def cal_acc(output,label):
            return 1 - output.argmax(dim=1).eq(label).float().mean()
        
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
        
        
from pytorch_lightning import LightningModule
import torch
class customModule(LightningModule):
    def __init__(self, crit, y_post_processor, output_post_processor, cal_acc, likelihood_fitting, args):
        super(customModule, self).__init__()
        self.crit = crit
        self.y_post_processor = y_post_processor
        self.output_post_processor = output_post_processor
        self.cal_acc = cal_acc
        self.likelihood_fitting = likelihood_fitting
        self.lr = args['lr']

    def forward(self,data):
        print("This should not print, then 'forward' in your model is not defined")
        return

    def configure_optimizers(self):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import LambdaLR
        from FunctionCollection import lambda_lr
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lambda_lr)
        return [optimizer], [scheduler]

    def training_step(self, data, batch_idx):
        try:
            weight = data.weight
        except:
            weight = None
        label = self.y_post_processor(data.y)
        if self.likelihood_fitting:
            output, cov = self.output_post_processor( self(data) )
            loss = self.crit(output, cov, label, weight)
        else:
            output = self.output_post_processor( self(data) )
            loss = self.crit(output, label, weight)

        acc = self.cal_acc(output, label)
        self.log("Train Loss", loss, on_step = True)
        self.log("Train Acc", acc, on_step = True)
        return {'loss': loss}

    def validation_step(self, data, batch_idx):
        label = self.y_post_processor(data.y)
        if self.likelihood_fitting:
            output, cov = self.output_post_processor( self(data) )
        else:
            output = self.output_post_processor( self(data) )

        acc = self.cal_acc(output, label)
#         self.log("val_batch_acc", acc, on_step = True)
#         self.log("Val Acc2", acc, on_epoch=True)
        return {'val_batch_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack([x['val_batch_acc'] for x in outputs]).mean()
        self.log("Val Acc", avg_acc)
#         self.log("lr", self.lr)
        return #{'Val Acc': avg_acc}  #It said it should not return anything \_(^^)_/

    def test_step(self, data, batch_idx):
        label = self.y_post_processor(data.y)
        if self.likelihood_fitting:
            output, cov = self.output_post_processor( self(data) )
        else:
            output = self.output_post_processor( self(data) )
        acc = self.cal_acc(output, label)
        self.log("Test Acc", acc, on_step = True)
        return {'Test Acc': acc}

#     def test_epoch_end(self, outputs):
#         avg_acc = torch.stack([x['test_batch_acc'] for x in outputs]).mean()
#         self.log("Test Acc", avg_acc)
#         return #{'Test Acc': avg_acc}
        
def edge_creators(iteration):
    if iteration == 1:
        from torch_geometric.transforms import KNNGraph
        return KNNGraph(loop=True)
    
    if iteration == 2:
        from torch_geometric.transforms import KNNGraph, ToUndirected, AddSelfLoops
        def edge_creator(dat):
            KNNGraph(k=5, loop=False, force_undirected = False)(dat)
            dat.adj_t = None
            ToUndirected()(dat)
            AddSelfLoops()(dat)
            dat.edge_index = dat.edge_index.flip(dims=[0])
            return dat
        return edge_creator
    
#     if iteration == 3:
##############################################################################################################     
        
import sqlite3
import os
import torch
import numpy as np
from pandas import read_sql
from torch_geometric.data import Data, Batch

import torch.utils.data


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class custom_db_dataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 filepath, 
                 filename, 
                 features, 
                 targets, 
                 TrTV, 
                 event_nos = None, 
                 x_transform = lambda x: torch.tensor(x.values), 
                 y_transform = lambda y: torch.tensor(y.values),
                 batch_transform = lambda tmp_x, events: tmp_x,
                 Data_constructor = lambda tmp_x, tmp_y: Data(x=tmp_x,y=tmp_y),
                 shuffle = False,
                 SRT_clean = False,
                 reweighter = None):

        self.filepath = filepath
        self.filename = filename
        self.features = features #Should be string of features, eg: "charge_log10, time, pulse_width, SRTInIcePulses, dom_x, dom_y, dom_z"
        self.targets = targets #Should be string of targets, eg: "azimuth, zenith, energy_log10"
        self.TrTV = TrTV #Should be cumulative sum of percentages for "Tr(ain)T(est)V(alidation)"" sets.
        
#         self.con = sqlite3.connect('file:'+os.path.join(self.filepath,self.filename+'?mode=ro'),uri=True)
        self.con_path = 'file:'+os.path.join(self.filepath,self.filename+'?mode=ro')
        self.x_transform = x_transform #should transform df to tensor
        self.y_transform = y_transform
        self.batch_transform = batch_transform
        self.Data_constructor = Data_constructor
        self.shuffle = shuffle
        self.SRT_clean = SRT_clean
        self.reweighter = reweighter
        
        if isinstance(event_nos,type(None)):
            with sqlite3.connect(self.con_path,uri=True) as con: 
                self.event_nos = np.asarray(read_sql("SELECT event_no FROM truth",con)).reshape(-1)
        else:
            self.event_nos = event_nos
        
        if self.shuffle:
            np.random.shuffle(self.event_nos)
        
        
    def __len__(self):
        """length method, number of events"""
        return len(self.event_nos)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_single(index)
        if isinstance(index, list):
            return self.get_list(index)
    
    def get_single(self,index):
        with sqlite3.connect(self.con_path,uri=True) as con:
            query = f"SELECT {self.features} FROM features WHERE event_no = {self.event_nos[index]}"
            if self.SRT_clean:
                query += " AND SRTInIcePulses = 1" 
            x = self.x_transform(read_sql(query,con))

            query = f"SELECT {self.targets} FROM truth WHERE event_no = {self.event_nos[index]}"
            y = self.y_transform(read_sql(query,con))
        return Data(x=x, y=y)
    
    def get_list(self,index):
        with sqlite3.connect(self.con_path,uri=True) as con:
            query = f"SELECT event_no, {self.features} FROM features WHERE event_no IN {tuple(self.event_nos[index])}"
            if self.SRT_clean:
                query += " AND SRTInIcePulses = 1" 
            events = read_sql(query, con)
            x = self.x_transform(events.iloc[:,1:])

            query = f"SELECT {self.targets} FROM truth WHERE event_no IN {tuple(self.event_nos[index])}"
            y = self.y_transform(read_sql(query,con))
        
        data_list = []
        _, events = np.unique(events.event_no.values.flatten(), return_counts = True)
        events = events.tolist()
        for tmp_x, tmp_y in zip(torch.split(x, events), y):
            tmp_x = self.batch_transform(tmp_x,events)
            data_list.append(self.Data_constructor(tmp_x,tmp_y))
#         return self.collate(data_list)
        return data_list
    
    def return_self(self,event_nos, extra_targets = ''):
        return custom_db_dataset(self.filepath,
                                 self.filename,
                                 self.features,
                                 self.targets + extra_targets,
                                 self.TrTV,
                                 event_nos,
                                 self.x_transform,
                                 self.y_transform,
                                 self.batch_transform,
                                 self.Data_constructor,
                                 self.shuffle,
                                 self.SRT_clean,
                                 self.reweighter)
    
    def train(self):
        return self.return_self(self.event_nos[:int(self.TrTV[0]*self.__len__())])

    def test(self, extra_targets = ', energy_log10'):
        return self.return_self(self.event_nos[int(self.TrTV[0]*self.__len__()):int(self.TrTV[1]*self.__len__())], extra_targets)

    def val(self):
        return self.return_self(self.event_nos[int(self.TrTV[1]*self.__len__()):int(self.TrTV[2]*self.__len__())])
    
    def collate(self,batch):
        return Batch.from_data_list(batch)
    
    def return_dataloader(self, batch_size, shuffle = False):
        from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, RandomSampler
        
        def collate(batch):
            return Batch.from_data_list(batch[0])
        
        if shuffle:
            sampler = BatchSampler(RandomSampler(self.train()),batch_size=batch_size,drop_last=False)
        else:
            sampler = BatchSampler(SequentialSampler(self.test(extra_targets = '')),batch_size=batch_size,drop_last=False)
        
        return DataLoader(dataset = self, collate_fn = collate, sampler = sampler)
    
    def return_dataloaders(self, batch_size, num_workers = 0): #Perhaps rewrite this using return_dataloader method
        from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, RandomSampler
        if self.reweighter:
            N_targets = len(self.targets.split(', '))
            def collate(batch):
                batch = Batch.from_data_list(batch[0])
                batch.weight = self.reweighter(batch)#torch.tensor(self.reweighter.predict_weights(batch.y.view(-1,N_targets))).view(-1,1)
                return batch
        else:
            def collate(batch):
                return Batch.from_data_list(batch[0])
            

        train_loader = DataLoader(dataset = self.train(),
                                      collate_fn = collate,
                                      num_workers = num_workers,
#                                       persistent_workers=True,
                                      pin_memory = True,
                                      sampler = BatchSampler(RandomSampler(self.train()),
                                                             batch_size=batch_size,
                                                             drop_last=False))
        
        test_loader = DataLoader(dataset = self.test(),
                                     collate_fn = collate,
                                     num_workers = num_workers,
#                                      persistent_workers=True,
                                     pin_memory = True,
                                     sampler = BatchSampler(SequentialSampler(self.test()),
                                                            batch_size=batch_size,
                                                            drop_last=False))
        
        val_loader = DataLoader(dataset = self.val(),
                                    collate_fn = collate,
                                    num_workers = num_workers,
#                                     persistent_workers=True,
                                    pin_memory = True,
                                    sampler = BatchSampler(RandomSampler(self.val()),
                                                           batch_size=batch_size,
                                                           drop_last=False))
        return train_loader, test_loader, val_loader

##############################################################################################################  
import pytorch_lightning as pl

def return_trainer(path, run_name, args, ckpt = None, patience = 7, max_epochs=50, log_every_n_steps=50):
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor='Val Acc', 
                                                                    min_delta=0.00, 
                                                                    patience=patience, 
                                                                    verbose=False, 
                                                                    mode='min')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath = path + '/checkpoints/' + run_name + '_' + args['id'],
                                                       filename = '{epoch}-{Val Acc:.3f}',
                                                       save_top_k = 1,
                                                       verbose = True,
                                                       monitor = 'Val Acc',
                                                       mode = 'min',
                                                       prefix = run_name)

    lr_logger = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval = 'epoch')

    from pytorch_lightning.loggers import WandbLogger
    if ckpt != None:
        wandb_logger = WandbLogger(name = run_name,
                                   project = 'Neutrino-Machine-Learning',
                                   version = run_name + '_' + args['id'],   # id and version can be interchanged, depending on whether you want to initialize or resume
                                   save_dir = path,
                                   sync_step = False) #Specify version as id if you want to resume a run.
        # wandb_logger.experiment.config.update(args)
        trainer =  pl.Trainer(gpus=-1, #-1 for all gpus
                              min_epochs=1,
                              max_epochs=max_epochs,
                              auto_lr_find = False,
                              auto_select_gpus = True,
                              log_every_n_steps = log_every_n_steps,
                              terminate_on_nan = True,
                              num_sanity_val_steps = 0,
                              callbacks=[early_stop_callback, checkpoint_callback, lr_logger] if args['wandb_activated'] else [early_stop_callback, checkpoint_callback], 
                              resume_from_checkpoint = path + '/checkpoints/' + run_name + '_' + args['id'] + '/' + ckpt,
                              logger = wandb_logger if args['wandb_activated'] else False,
                              default_root_dir = path)
    else:
        wandb_logger = WandbLogger(name = run_name,
                                   project = 'Neutrino-Machine-Learning',
                                   id = run_name + '_' + args['id'],   # id and version can be interchanged, depending on whether you want to initialize or resume
                                   save_dir = path) #Specify version as id if you want to resume a run.
        trainer =  pl.Trainer(gpus=-1, #-1 for all gpus
                              min_epochs=1,
                              max_epochs=max_epochs,
                              auto_lr_find = False,
                              auto_select_gpus = True,
                              log_every_n_steps = log_every_n_steps,
                              terminate_on_nan = True,
                              num_sanity_val_steps = 0,
                              callbacks=[early_stop_callback, checkpoint_callback, lr_logger] if args['wandb_activated'] else [early_stop_callback, checkpoint_callback],
                              logger = wandb_logger if args['wandb_activated'] else False,
                              default_root_dir = path)
    return trainer, wandb_logger  

def Print(statement):
    from time import localtime, strftime
    print("{} - {}".format(strftime("%H:%M:%S", localtime()),statement))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    