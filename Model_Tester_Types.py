import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch_geometric.data import DataLoader, InMemoryDataset
import time
import importlib
from sklearn.dummy import DummyClassifier

#### For loading the dataset as a torch_geometric InMemoryDataset   ####
#The @properties should be unimportant for now, including process since the data is processed.
class MakeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MakeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[-1]) #Loads PROCESSED.file
                                                                      #perhaps check print(self.processed_paths)
    @property
    def raw_file_names(self):
        return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/raw_data')

    @property
    def processed_file_names(self):
        return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

    def process(self):
        pass

print('Loads data')
dataset = MakeDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset')
####                                                                #####

#### Changing target variables to one hot encoded (or not) neutrino type ####
#### It is important to remember to change the slicing as well as y#
types = torch.tensor([np.zeros(100000),np.ones(100000),np.ones(100000)*2],dtype=torch.int64).reshape((1,-1))

dataset.data.y = types[0]
dataset.slices['y'] = torch.tensor(np.arange(300000+1))
####                                    ####

####Look at subset  ####
# dataset = dataset[:100000] + dataset[200000:]
# train_dataset = dataset

dataset = dataset.shuffle()

train_dataset = dataset[:50000]
val_dataset = dataset[50000:75000]
test_dataset = dataset[75000:100000]

# train_dataset = dataset[:200000]
# val_dataset = dataset[200000:250000]
# test_dataset = dataset[250000:]
####                ####

batch_size= 64
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
dummy_prediction()
####

print('Loads model')
#Define model:
#The syntax is for model i: from Models.Model{i} import Net
import Models.Model5 as Model
Model = importlib.reload(Model)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

crit = torch.nn.NLLLoss()   #Loss function

def train():
    model.train()
    correct = 0
    for data in train_loader:
        label = data.y.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        del data

        loss = crit(output, label)
        loss.backward()
        optimizer.step()

        correct += output.argmax(dim=1).eq(label).sum()
    torch.cuda.empty_cache()
    return loss, correct.float()/len(train_dataset)

# def test():
#     model.eval()
#     true_prob = 0
#     for data in test_loader:
#         data = data.to(device)
#         label = data.y
#         output = model(data)
#         for i in range(len(label)):
#             true_prob += torch.exp(output[i][label[i]])
#     return true_prob/len(test_dataset)

print('ready for training')
loss_list, ratio_list = [], []
def epochs(i):
    print('Begins training')
    t = time.time()
    for epoch in range(i):
        print(f'Epoch: {epoch}')
        curr_loss,ratio = train()
        loss_list.append(curr_loss.item())
        ratio_list.append(ratio.item())
        print(curr_loss.item(),ratio.item())
        print(f'time since beginning: {time.time() - t}')
    print('Done')

