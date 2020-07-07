import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch_geometric.data import DataLoader, InMemoryDataset
import time

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
# dataset.data.y = dataset.data.y.reshape((300000,8))
####                                                                #####

#### Changing target variables to one hot encoded neutrino type ####
#### It is important to remember to change the slicing as well as y#
# types = torch.nn.functional.one_hot(torch.tensor([np.zeros(100000),np.ones(100000),np.ones(100000)*2],dtype=torch.int64).reshape((1,-1)))
types = torch.tensor([np.zeros(100000),np.ones(100000),np.ones(100000)*2],dtype=torch.int64).reshape((1,-1))

dataset.data.y = types[0]
dataset.slices['y'] = torch.tensor(np.arange(300000))
####                                    ####

####Look at subset  ####
# dataset = dataset[100000:]
dataset = dataset.shuffle()

train_dataset = dataset[:50000]
val_dataset = dataset[50000:75000]
test_dataset = dataset[75000:100000]

# train_dataset = dataset[:200000]
# val_dataset = dataset[200000:250000]
# test_dataset = dataset[250000:]
####                ####

batch_size= 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print('Loads model')
#Define model:
# from Models.Model1 import Net   #The syntax is for model i: from Models.Model{i} import Net
from Models.Model3 import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
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
        loss = crit(output, label)
        loss.backward()
        optimizer.step()

        guess = torch.argmax(output,dim=1)
        correct += sum(guess == label)
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

print('Begins training')
t = time.time()

for epoch in range(10):
    print(f'Epoch: {epoch}')
    curr_loss,ratio = train()
    print(curr_loss,ratio)
    print(f'time since beginning: {time.time() - t}')

print('Done')
