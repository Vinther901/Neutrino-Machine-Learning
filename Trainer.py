# import json

# exp_path = r"C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\args.json"

# with open(exp_path) as file:
#     args = json.load(file)

from argparse import ArgumentParser


import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import time
import wandb
import pytorch_lightning as pl
import sqlite3

import FunctionCollection as fc
import importlib
fc = importlib.reload(fc)
import os

path = r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning'

run_name = 'TEST_OscNext_AngleO_m17'

args = {'N_edge_feats': 6,
        'N_dom_feats': 7,
        'N_targets': 2,
        'N_outputs': 3,
        'N_metalayers': 2,
        'N_hcs': 64,
        'diagonal_cov': True,
        'wandb_activated': True,
        'type': 'Spherical_NLLH',
        'zenith': True,
        'id': wandb.util.generate_id()[:4],
        'eps': 1e-5,
        'output_offset': [3.14,1.57],
        'lr': 3e-3,
        'batch_size': 512,
        'filename': 'Nu_lvl7_1Mio_unscaled_SRT.db',#dev_level7_mu_e_tau_oscweight_000.db #rasmus_classification_muon_3neutrino_3mio.db #dev_level7_oscNext_IC86_003.db
        'features': 'width, rqe, charge_log10, dom_time, dom_x, dom_y, dom_z',#'charge_log10, dom_time, width, rqe, dom_x, dom_y, dom_z', # SRTInIcePulses,
        'targets': 'azimuth, zenith',
        'TrTV': (0.9,0.999,1),#(0.025,0.995,1)
        'SRT_clean': False
       }

centers = pd.DataFrame({'charge_log10': [-0.033858],
                        'dom_time': [10700.0],
                        'dom_x': [0],
                        'dom_y': [0],
                        'dom_z': [0],
                        'width': [4.5],
                        'rqe': [1.175]})
scalers = pd.DataFrame({'charge_log10': [0.274158],
                        'dom_time': [2699.0],
                        'dom_x': [300],
                        'dom_y': [300],
                        'dom_z': [300],
                        'width': [3.5],
                        'rqe': [0.175]})
centers = centers[args['features'].split(', ')].values
scalers = scalers[args['features'].split(', ')].values

def x_transform(df):
    df = (df - centers)/scalers
    return torch.tensor(df.values)

def y_transform(df):
    return torch.tensor(df.values)

from typing import List
@torch.jit.script
def batch_transform(x,events: List[int]):
    tmp_x = x.unsqueeze(1) - x
    cart = tmp_x[:,:,-3:]

    rho = torch.norm(cart, p=2, dim=-1).unsqueeze(2)
    rho_mask = rho.squeeze() != 0
    if rho_mask.sum() != 0:
        cart[rho_mask] = cart[rho_mask] / rho[rho_mask]
    tmp_x = torch.cat([cart,rho,tmp_x[:,:,:-3]],dim=2)
    return torch.cat([tmp_x.mean(1),tmp_x.std(1),tmp_x.min(1)[0],tmp_x.max(1)[0],x],dim=1)

filepath = os.path.join(path,'raw_data')


def return_trainer(ckpt = None):
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor='Val Acc', 
                                                                    min_delta=0.00, 
                                                                    patience=20, 
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
    wandb_logger = WandbLogger(name = run_name,
                               project = 'Neutrino-Machine-Learning',
                               id = run_name + '_' + args['id'],   # id and version can be interchanged, depending on whether you want to initialize or resume
                               save_dir = path) #Specify version as id if you want to resume a run.
    trainer =  pl.Trainer(gpus=hparams.gpus, #-1 for all gpus
                          min_epochs=1,
                          max_epochs=50,
                          auto_lr_find = False,
#                           accelerator = 'ddp',
                          auto_select_gpus = True,
                          log_every_n_steps = 50,
                          terminate_on_nan = True,
                          num_sanity_val_steps = 0,
                          callbacks=[early_stop_callback, checkpoint_callback, lr_logger],
                          logger = wandb_logger if args['wandb_activated'] else False,
                          default_root_dir = path)
    return trainer, wandb_logger
def main(hparams):
    import Model_Loaders.Model_17 as M
    M = importlib.reload(M)

    Net = M.Load_model(args['type'],args)

    model = Net()

    trainer, wandb_logger = return_trainer()
    
    dataset = fc.custom_db_dataset(filepath = filepath,
                               filename = args['filename'],
                               features = args['features'],
                               targets = args['targets'],
                               TrTV = args['TrTV'],
#                                event_nos = event_nos,
                               x_transform = x_transform,
                               y_transform = y_transform,
                               batch_transform = batch_transform,
                               shuffle = True,
                               SRT_clean = args['SRT_clean'],
                              #  reweighter = ze_reweighter
                               )

    train_loader, test_loader, val_loader = dataset.return_dataloaders(batch_size=args['batch_size'],num_workers = 0)

#     lr_finder = trainer.tuner.lr_find(model,train_loader,val_loader,min_lr=1e-6,max_lr=5e-2,num_training=100,mode='exponential',early_stop_threshold=4)

#     fig = lr_finder.plot(True,True)
#     print(lr_finder.suggestion())
#     fig.show()
    device = torch.device('cuda')
    model.to(device)
    for dat in train_loader:
        dat.to(device)
        model.training_step(dat,None)
        break
    print(model.log)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    hparams = parser.parse_args()
    
    main(hparams)