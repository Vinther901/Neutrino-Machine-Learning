import matplotlib.pyplot as plt
import os
import torch
from torch_geometric.data import DataLoader, InMemoryDataset

#For loading the dataset as a torch_geometric InMemoryDataset
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

dataset = dataset.shuffle()
# train_dataset = dataset[:200000]
# val_dataset = dataset[200000:250000]
# test_dataset = dataset[250000:]

#Look at subset
train_dataset = dataset[:20000]
val_dataset = dataset[20000:25000]
test_dataset = dataset[25000:30000]

batch_size= 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print('Loads model')
#Define model:
from Models.Model1 import Net   #The syntax is for model i: from Models.Model{i} import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

crit = torch.nn.MSELoss()   #Loss function

def train():
    model.train()
    loss_all = 0
    for data in train_loader:
        label = torch.reshape(data.y,(-1,8)).to(device)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    
    model.eval()
    test_loss = 0
    for data in test_loader:
        label = torch.reshape(data.y,(-1,8)).to(device)
        data = data.to(device)
        output = model(data)
        loss = crit(output, label)
        test_loss += data.num_graphs * loss.item()

    return loss_all / len(train_dataset), test_loss / len(train_dataset)

print('Begins training')

TrainLoss, TestLoss = [], []
for epoch in range(5):
    print(f'Epoch: {epoch}')
    trainloss, testloss = train()
    TrainLoss.append(trainloss)
    TestLoss.append(testloss)

print('Done')



