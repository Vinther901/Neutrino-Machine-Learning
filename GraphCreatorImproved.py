import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import torch_geometric.transforms as T
import torch
from tqdm import tqdm

from datetime import datetime
import sqlite3

filename = "rasmus_classification_muon_3neutrino_3mio.db"
db_path = "C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/raw_data/{}".format(filename)
destination = "C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/datasets"

particle = '2_neutrinos' #'muonNeutrino' #'3_neutrinos' #'muon'
save_filename = 'EMu_neutrinos_set2' #'muonNeutrino_set2' #'muon_set2' #'muonNeutrino_set1' #'neutrinos_set1' #'muon_set1'

event_nos = None
# event_nos = pd.read_pickle(destination + '/event_nos_500k_muon_set1.pkl').iloc[:100000]
# event_nos = pd.read_pickle(destination + '/event_nos_0k_test.pkl')

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
        
        event_nos = pd.concat([event_nos_electron,event_nos_muon],axis=0).sample(N)
        print(event_nos)
        
    
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

for subset in range(N//subdivides):
    event_no_subset = event_nos.iloc[subset*subdivides:(subset+1)*subdivides]

    query = "SELECT charge_log10, time, pulse_width, dom_x, dom_y, dom_z, event_no FROM features WHERE event_no IN {} and SRTInIcePulses = 1".format(tuple(event_no_subset.event_no))
    events = pd.read_sql(query,con)
    
    print('{}: Events features extracted..'.format(datetime.now()))
    
    if tfs is not None:
        print('{}: Inverse transforming..'.format(datetime.now()))
        x_pos = torch.tensor(tfs['features']['dom_x'].inverse_transform(events[['dom_x']]),dtype=torch.float)
        y_pos = torch.tensor(tfs['features']['dom_y'].inverse_transform(events[['dom_y']]),dtype=torch.float)
        z_pos = torch.tensor(tfs['features']['dom_z'].inverse_transform(events[['dom_z']]),dtype=torch.float)
        x = torch.cat([torch.tensor(events[['charge_log10','time','pulse_width']].values,dtype=torch.float),x_pos,y_pos,z_pos],dim=1)
#         pos = torch.cat([x_pos,y_pos,z_pos],dim=1)
    else:
        x = torch.tensor(events[['charge_log10','time','pulse_width','dom_x','dom_y','dom_z']].values,dtype=torch.float) #Features
#         pos = torch.tensor(events[['dom_x','dom_y','dom_z']].values,dtype=torch.float) #Position
        
    _, events = np.unique(events.event_no.values.flatten(), return_counts = True)
    
    query = "SELECT energy_log10, time, position_x, position_y, position_z, direction_x, direction_y, direction_z, azimuth, zenith, pid FROM truth WHERE event_no IN {}".format(tuple(event_no_subset.event_no))
    y = pd.read_sql(query,con)
    
    print('{}: Events truths extracted..'.format(datetime.now()))

    y = torch.tensor(y.values,dtype=torch.float) #Target
    
    data_list = []
    for tmp_x, tmp_y in tqdm(zip(torch.split(x, events.tolist()), y), total = subdivides):
        dat = Data(x=tmp_x,edge_index=None,edge_attr=None,y=tmp_y,pos=tmp_x[:,-3:]) 

    #     T.KNNGraph(loop=True)(dat) #defining edges by k-NN with k=6 !!! Make sure .pos is not scaled!!! ie. x,y,z  -!-> ax,by,cz

        T.KNNGraph(k=5, loop=False, force_undirected = False)(dat)
        dat.adj_t = None
        T.ToUndirected()(dat)
        T.AddSelfLoops()(dat)
        (row, col) = dat.edge_index
        dat.edge_index = torch.stack([col,row],dim=0)

        data_list.append(dat)

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data,slices), destination + '/{}k_{}{}.pt'.format(subdivides//1000,save_filename,subset))
    print('{}: File saved..'.format(datetime.now()))
    subset += 1
    data_list = []
#     if (i+1) % subdivides == 0:
#         data, slices = InMemoryDataset.collate(data_list)
#         torch.save((data,slices), destination + '/{}k_{}{}.pt'.format(subdivides//1000,save_filename,subset))
#         subset += 1
#         data_list = [] #Does this free up the memory?

#     if i % 500 == 0:
#         print("{}: Completed {}/{}".format(datetime.now(),i,N))

if data_list != []:
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data,slices), destination + '/{}k_{}{}.pt'.format(subdivides//1000,save_filename,subset))
