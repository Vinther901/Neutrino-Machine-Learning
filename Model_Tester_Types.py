import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.data import DataLoader, InMemoryDataset
import time
import importlib
from sklearn.dummy import DummyClassifier

# #### For loading the dataset as a torch_geometric InMemoryDataset   ####
# #The @properties should be unimportant for now, including process since the data is processed.
# class MakeDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(MakeDataset, self).__init__(root, transform, pre_transform)
#         # print(self.processed_paths)
#         self.data, self.slices = torch.load(self.processed_paths[-1]) #Loads PROCESSED.file
#                                                                       #perhaps check print(self.processed_paths)
#     @property
#     def raw_file_names(self):
#         return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/raw_data')

#     @property
#     def processed_file_names(self):
#         return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

#     def process(self):
#         pass

# print('Loads data')
# dataset = MakeDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset')
# dataset_background = MakeDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/dataset_background')
# ####                                                                #####

# #### Changing target variables to one hot encoded (or not) neutrino type ####
# #### or changing them to neutri8730no and have muons as background
# #### It is important to remember to change the slicing as well as y#

# # types = torch.tensor([np.zeros(100000),np.ones(100000),np.ones(100000)*2],dtype=torch.int64).reshape((1,-1))
# types = torch.tensor(np.zeros(300000),dtype=torch.int64)
# dataset.data.y = types
# dataset.slices['y'] = torch.tensor(np.arange(300000+1))

# dataset_background.data.y = torch.tensor(np.ones(dataset_background.len()),dtype=torch.int64)
# dataset_background.slices['y'] = torch.tensor(np.arange(dataset_background.len() + 1))
# ####                                    ####

# ####Look at subset  ####
# subsize = 30000

# nu_e_ind = np.random.choice(np.arange(100000),subsize,replace=False).tolist()
# nu_t_ind = np.random.choice(np.arange(100000,200000),subsize,replace=False).tolist()
# nu_m_ind = np.random.choice(np.arange(200000,300000),subsize,replace=False).tolist()
# muon_ind = np.random.choice(np.arange(dataset_background.len()),3*subsize,replace=False).tolist()

# train_dataset = dataset[nu_e_ind] + dataset[nu_t_ind] + dataset[nu_m_ind] + dataset_background[muon_ind]

# test_ind_e = np.arange(100000)[pd.Series(np.arange(100000)).isin(nu_e_ind).apply(lambda x: not(x))].tolist()
# test_ind_t = np.arange(100000,200000)[pd.Series(np.arange(100000,200000)).isin(nu_e_ind).apply(lambda x: not(x))].tolist()
# test_ind_m = np.arange(200000,300000)[pd.Series(np.arange(200000,300000)).isin(nu_e_ind).apply(lambda x: not(x))].tolist()
# test_ind_muon = np.arange(dataset_background.len())[pd.Series(np.arange(dataset_background.len())).isin(nu_e_ind).apply(lambda x: not(x))].tolist()

# test_dataset = dataset[test_ind_e] + dataset[test_ind_t] + dataset[test_ind_m] + dataset_background[test_ind_muon]

# # train_dataset = dataset[:100000] + dataset[100000:200000] + dataset[200000:] + dataset_background

# # dataset = dataset.shuffle()


# # train_dataset = dataset[:200000]
# # val_dataset = dataset[200000:250000]
# # test_dataset = dataset[250000:]
# ####                ####

class MakeDataset(InMemoryDataset):
    def __init__(self, root, dataset):
        super(MakeDataset, self).__init__(root)
        # print(self.processed_paths)
        self.data, self.slices = torch.load(root+'/' + dataset)
    @property
    def processed_file_names(self):
        return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

    def process(self):
        pass

train_dataset = MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets','train_class')
train_dataset = train_dataset.shuffle()

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=1024)

#### predicting baseline classification
def dummy_prediction(stratified_num = 20):
    dummy_clf1 = DummyClassifier(strategy="most_frequent")
    dummy_clf2 = DummyClassifier(strategy='stratified')
    target = dataset.data.y[train_dataset.indices()]
    empty_array = np.empty(train_dataset.__len__())
    dummy_clf1.fit(empty_array,target)
    dummy_clf2.fit(empty_array,target)
    pred1 = dummy_clf1.score(empty_array,target)
    pred2 = np.asarray([dummy_clf2.score(empty_array,target) for i in range(stratified_num)])
    print(f'Dummy predicter: most frequent: {pred1}, stratified: {pred2.mean()} +- {pred2.std()}')
# dummy_prediction()
####

print('Loads model')
#Define model:
#The syntax is for model i: from Models.Model{i} import Net
import Models.Model7 as Model
Model = importlib.reload(Model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

crit = torch.nn.NLLLoss()   #Loss function

# # #For loading existing model and optimizer parameters.
# print('Loading existing model and optimizer states')
# state = torch.load('Trained_Models/Model5_Class.pt')
# model.load_state_dict(state['model_state_dict'])
# optimizer.load_state_dict(state['optimizer_state_dict'])

batch_loss, batch_acc = [], []
def train():
    model.train()
    correct = 0
    for data in train_loader:
        data = data.to(device)
        label = data.y
        optimizer.zero_grad()
        output = model(data)

        del data

        loss = crit(output, label)
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())
        acc = output.argmax(dim=1).eq(label).sum()
        batch_acc.append(acc.item()/batch_size)

        correct += acc
    torch.cuda.empty_cache()

    return loss.item(), (correct.float()/len(train_dataset)).item()

print('ready for training')
loss_list, acc_list = [], []
def epochs(i):
    print('Begins training')
    t = time.time()
    for epoch in range(i):
        print(f'Epoch: {epoch}')
        curr_loss,ratio = train()
        print(curr_loss,ratio)
        loss_list.append(curr_loss)
        acc_list.append(ratio)
        print(f'time since beginning: {time.time() - t}')
    print('Done')

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            label = data.y
            output = model(data)

            del data

            correct += output.argmax(dim=1).eq(label).sum()
        torch.cuda.empty_cache()
    return (correct.float()/len(test_dataset)).item()

def test_all():
    model.eval()
    score_list = []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            label = data.y
            output = model(data)

            del data

            score_list.append(output.tolist())

        torch.cuda.empty_cache()
    return score_list