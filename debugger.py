import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import time
import wandb
import pytorch_lightning as pl

import FunctionCollection as fc
import importlib
fc = importlib.reload(fc)
import os

path = r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning'

run_name = 'test_angle_m10'

args = {'N_edge_feats': 6,
        'N_dom_feats': 7,
        'N_targets': 2,
        'N_outputs': 4,
        'N_metalayers': 1,
        'N_hcs': 64,
        'diagonal_cov': True,
        'wandb_activated': False,
        'type': 'twice_Polar_NLLH',
        'zenith': True,
        'id': wandb.util.generate_id()[:4],
        'eps': 0,
        'lr': 8e-2,
        'filename': 'rasmus_classification_muon_3neutrino_3mio.db',#dev_level7_mu_e_tau_oscweight_000.db #rasmus_classification_muon_3neutrino_3mio.db #dev_level7_oscNext_IC86_003.db
        'features': 'charge_log10, time, pulse_width, SRTInIcePulses, dom_x, dom_y, dom_z',
        'targets': 'azimuth, zenith',
        'TrTV': (0.1,0.99,1)#(0.025,0.995,1)
       }

# filepath = os.path.join(path,'raw_data/dev_level7_mu_e_tau_oscweight_000/data')
filepath = os.path.join(path,'raw_data')


tf = pd.read_pickle(r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\datasets\transformers.pkl')
# tf = pd.read_pickle(r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\raw_data\dev_level7_mu_e_tau_oscweight_000\data\meta\transformers.pkl')
event_nos = pd.read_pickle(r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\datasets\event_nos_500k_muon_set1.pkl').values.reshape(-1)

def x_transform(df):
    pos = ['dom_x','dom_y','dom_z']
    for col in pos:
        df[col] = tf['features'][col].inverse_transform(df[[col]])
    df[pos] /= 300
    return torch.tensor(df.values)
def y_transform(df):
    for col in df.columns:
        df[col] = tf['truth'][col].inverse_transform(df[[col]])
    return torch.tensor(df.values)

# #@torch.jit.script
# def x_transform(df):
#     df['charge_log10'] = (df['charge_log10'] - charge_center)/charge_scale
#     df['dom_time'] = (df['dom_time'] - time_center)/time_scale
#     df[['dom_x','dom_y','dom_z']] /= 300
#     return torch.tensor(df.values)
# #@torch.jit.script
# def y_transform(df):
#     return torch.tensor(df.values)



dataset = fc.custom_db_dataset(filepath = filepath,
                               filename = args['filename'],
                               features = args['features'],
                               targets = args['targets'],
                               TrTV = args['TrTV'],
#                                event_nos = event_nos,
                               x_transform = x_transform,
                               y_transform = y_transform,
                               shuffle = True)


train_loader, test_loader, val_loader = dataset.return_dataloaders(batch_size=512) #~0.6sec loading time pr. batch

i = 0
start = time.time()
for dat in train_loader:
    print(time.time() - start)
    start = time.time()
    i += 1
    if i > 100:
        break