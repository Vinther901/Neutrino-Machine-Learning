import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import torch_geometric.transforms as T
import torch

from datetime import datetime
import sqlite3

# filename = "rasmus_classification_muon_3neutrino_3mio.db"
# db_path = "C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/raw_data/{}".format(filename)
# destination = "C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/datasets"

# particle = 'muon' #'muonNeutrino' #'3_neutrinos' #'muon'
# save_filename = 'muon_set2' #'muonNeutrino_set1' #'neutrinos_set1' #'muon_set1'

# event_nos = None
# event_nos = pd.read_pickle(destination + '/event_nos_500k_muon_set1.pkl').iloc[:100000]

#Select event_no only for muons:

print('{}: Collecting event numbers..'.format(datetime.now()))

if event_nos is None:
    subdivides = 100000
    N = 2*subdivides
    
    if particle == 'muon':
        query = "SELECT event_no FROM truth WHERE pid = 13"
        with sqlite3.connect(str(db_path)) as con:
            event_nos = pd.read_sql(query,con)
        event_nos = event_nos.sample(N)
        
    elif particle == '3_neutrinos':
        query = "SELECT event_no FROM truth WHERE pid = 12"
        with sqlite3.connect(str(db_path)) as con:
            event_nos = pd.read_sql(query,con)
        event_nos_electron = event_nos.sample(N//3)
        
        query = "SELECT event_no FROM truth WHERE pid = 14"
        event_nos = pd.read_sql(query,con)
        event_nos_muon = event_nos.sample(N//3)
        
        query = "SELECT event_no FROM truth WHERE pid = 16"
        event_nos = pd.read_sql(query,con)
        event_nos_tau = event_nos.sample(N//3)
        
        event_nos = pd.concat([event_nos_electron,event_nos_muon,event_nos_tau],dim=0)
    
    elif particle == '2_neutrinos':
        query = "SELECT event_no FROM truth WHERE pid = 12"
        with sqlite3.connect(str(db_path)) as con:
            event_nos = pd.read_sql(query,con)
        event_nos_electron = event_nos.sample(N//2)
        
        query = "SELECT event_no FROM truth WHERE pid = 14"
        event_nos = pd.read_sql(query,con)
        event_nos_muon = event_nos.sample(N//2)
        
        event_nos = pd.concat([event_nos_electron,event_nos_muon],dim=0)
    
    elif particle == 'muonNeutrino':
        query = "SELECT event_no FROM truth WHERE pid = 14"
        with sqlite3.connect(str(db_path)) as con:
            event_nos = pd.read_sql(query,con)
        event_nos = event_nos.sample(N)
        
    event_nos.to_pickle(destination + '/' + 'event_nos_{}k_{}.pkl'.format(N//1000,save_filename))

    print('{}: Saved relevant event numbers.. \nBeginning Graph creation..'.format(datetime.now()))
else:
    with sqlite3.connect(str(db_path)) as con:
        subdivides = event_nos.shape[0]
        N = subdivides

tfs = pd.read_pickle("C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/datasets/transformers.pkl")
print("Remember, transformer is currently activated to inverse_transform position variables..")
subset = 1
data_list = []
for i, event_no in enumerate(event_nos.values.flatten()):
    query = "SELECT charge_log10, time, dom_x, dom_y, dom_z FROM features WHERE event_no = {} AND SRTInIcePulses = 1".format(event_no)
    
    tmp_event = pd.read_sql(query,con)

    tmp_event = tmp_event.loc[tmp_event.SRTInIcePulses == 1]
    
    if tfs is not None:
        x_pos = torch.tensor(tfs['features']['dom_x'].inverse_transform(tmp_event[['dom_x']]),dtype=torch.float)
        y_pos = torch.tensor(tfs['features']['dom_y'].inverse_transform(tmp_event[['dom_y']]),dtype=torch.float)
        z_pos = torch.tensor(tfs['features']['dom_z'].inverse_transform(tmp_event[['dom_z']]),dtype=torch.float)
        x = torch.cat([torch.tensor(tmp_event[['charge_log10','time']].values,dtype=torch.float),x_pos,y_pos,z_pos],dim=1)
        pos = torch.cat([x_pos,y_pos,z_pos],dim=1)
    else:
        x = torch.tensor(tmp_event[['charge_log10','time','dom_x','dom_y','dom_z']].values,dtype=torch.float) #Features
        pos = torch.tensor(tmp_event[['dom_x','dom_y','dom_z']].values,dtype=torch.float) #Position

    query = "SELECT energy_log10, time, position_x, position_y, position_z, direction_x, direction_y, direction_z, azimuth, zenith FROM truth WHERE event_no = {}".format(event_no)
    y = pd.read_sql(query,con)

    y = torch.tensor(y.values,dtype=torch.float) #Target

    dat = Data(x=x,edge_index=None,edge_attr=None,y=y,pos=pos) 
    
#     T.KNNGraph(loop=True)(dat) #defining edges by k-NN with k=6 !!! Make sure .pos is not scaled!!! ie. x,y,z  -!-> ax,by,cz
    
    T.KNNGraph(k=6, loop=False, force_undirected = False)(dat)
    dat.adj_t = None
    T.ToUndirected()(dat)
    T.AddSelfLoops()(dat)
    (row, col) = dat.edge_index
    dat.edge_index = torch.stack([col,row],dim=0)
    
    data_list.append(dat)

    if (i+1) % subdivides == 0:
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data,slices), destination + '/{}k_{}{}.pt'.format(subdivides//1000,save_filename,subset))
        subset += 1
        data_list = [] #Does this free up the memory?
    
    if i % 500 == 0:
        print("{}: Completed {}/{}".format(datetime.now(),i,N))

if data_list != []:
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data,slices), destination + '/{}k_{}{}.pt'.format(subdivides//1000,save_filename,subset))
