import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import torch_geometric.transforms as T
import torch

from datetime import datetime
import sqlite3

filename = "rasmus_classification_muon_3neutrino_3mio.db"
db_path = "C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/raw_data/{}".format(filename)
destination = "C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/datasets"

# event_nos = None
event_nos = pd.read_pickle(destination + '/event_nos_500k_muon_set1.pkl').iloc[200000:]

#Select event_no only for muons:

print('{}: Collecting event numbers..'.format(datetime.now()))

if event_nos is None:
    query = "SELECT event_no FROM truth WHERE pid = 13"
    with sqlite3.connect(str(db_path)) as con:
        event_nos = pd.read_sql(query,con)

    subdivides = 100000
    N = 5*subdivides

    event_nos = event_nos.sample(N)
    event_nos.to_pickle(destination + '/' + 'event_nos_{}k_muon_set1.pkl'.format(N//1000))

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
    query = "SELECT charge_log10, time, dom_x, dom_y, dom_z, SRTInIcePulses FROM features WHERE event_no = {}".format(event_no)
    
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
    T.KNNGraph(loop=True)(dat) #defining edges by k-NN with k=6 !!! Make sure .pos is not scaled!!! ie. x,y,z  -!-> ax,by,cz
    data_list.append(dat)

    if (i+1) % subdivides == 0:
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data,slices), destination + '/muon_{}k_set1{}.pt'.format(subdivides//1000,subset))
        subset += 1
        data_list = [] #Does this free up the memory?
    
    if i % 500 == 0:
        print("{}: Completed {}/{}".format(datetime.now(),i,N))

if data_list != []:
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data,slices), destination + '/muon_{}k_set1{}.pt'.format(subdivides//1000,subset))


# def KNN_graph_dataset(path, seq, scalar, destination, pre_transform = None):
#     print('Loading data..')
#     seq = pd.read_csv( path + '/' + seq ) # input
#     scalar = pd.read_csv( path + '/' + scalar) # target
#     print('Data loaded!')

#     data_list = []
#     i = 0

#     print('Starting processing..')
#     t0 = time()

#     for index, sca in scalar.iterrows():
#         tmp_event = seq.loc[seq['event_no'] == sca['event_no']]
#         x = torch.tensor(tmp_event[['dom_charge','dom_time','dom_x','dom_y','dom_z']].values,dtype=torch.float) #Features
#         pos = torch.tensor(tmp_event[['dom_x','dom_y','dom_z']].values,dtype=torch.float) #Position
#         y = torch.tensor(sca[sca.keys()[2:]].values,dtype=torch.float) #Target
#         dat = Data(x=x,edge_index=None,edge_attr=None,y=y,pos=pos) 
#         T.KNNGraph(loop=True)(dat) #defining edges by k-NN with k=6
#         data_list.append(dat)
#         print(i)
#         if i >= 5:
#             break
#         i += 1
    
#     print('Processing done! \nTime passed:',time()-t0)


#     # if pre_filter is not None:
#     #     data_list = [data for data in data_list if pre_filter(data)]

#     if pre_transform is not None:
#         data_list = [pre_transform(data) for data in data_list]

#     # print(data_list)
#     data, slices = InMemoryDataset.collate(None,data_list)
#     torch.save((data,slices), destination)