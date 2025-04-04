import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch_geometric.data import DataLoader, InMemoryDataset
import time
import importlib

#### For loading the dataset as a torch_geometric InMemoryDataset   ####
# The @properties should be unimportant for now, including process since the data is processed.
def load_dataset(path='C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset'):
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
    var = MakeDataset(root = path)
    return var

dataset = load_dataset()
####                                                                #####

#### Changing target variables from 8 to Energy alone ####
dataset.data.y = dataset.data.y[::8]
dataset.slices['y'] = torch.tensor(np.arange(300000+1))
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

train_batch_size = 128
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=train_batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print('Loads model')
#Define model:
# from Models.Model1 import Net   #The syntax is for model i: from Models.Model{i} import Net
import Models.Model2 as Model
Model = importlib.reload(Model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.MSELoss()   #Loss function

# #For loading existing model and optimizer parameters.
print('Loading existing model and optimizer states')
state = torch.load('Trained_Models/Model2_Energy.pt')
model.load_state_dict(state['model_state_dict'])
optimizer.load_state_dict(state['optimizer_state_dict'])

def save_model(path):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},path)
    print('Model saved')

train_size = train_dataset.__len__()
def train():
    model.train()
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        label = data.y
        optimizer.zero_grad()
        output = model(data)

        del data

        loss = crit(output,label)
        loss.backward()
        optimizer.step()
        train_loss += loss

    torch.cuda.empty_cache()
    return train_loss/train_size

test_size = test_dataset.__len__()
def test():
    model.eval()
    acc = 0
    with torch.no_grad():
        for data in test_loader:
            label = data.y
            data = data.to(device)
            output = model(data)

            del data

            acc += output.cpu().view(-1).dist(label)
        torch.cuda.empty_cache()
    return acc / test_size

def epochs(i):
    t = time.time()
    for epoch in range(i):
        print(f'Epoch: {epoch}')
        err = train()
        acc = test()
        print(f'training error: {err.item()} testing accuracy: {acc.item()}')
        print(f'time since beginning: {time.time() - t}')

print('Ready for training')

# def train():
#     model.train()
#     train_score = 0
#     for data in train_loader:
#         label = data.y.to(device)
#         data = data.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         print(label.shape,output.shape)
#         train_score += (output.view(-1) - label)/label
#         loss = crit(output, label)
#         loss.backward()
#         optimizer.step()
#     train_score /= train_loader.__len__()
    
#     model.eval()
#     test_score = 0
#     for data in test_loader:
#         label = data.y.to(device)
#         data = data.to(device)
#         output = model(data)
#         test_score += (output.view(-1) - label)/label
#     test_score /= test_loader.__len__()

#     return torch.mean(train_score,0).data.cpu().numpy(), torch.mean(test_score,0).data.cpu().numpy()

# print('Begins training')
# t = time.time()
# train_scores, test_scores = [], []
# for epoch in range(5):
#     print(f'Epoch: {epoch}')
#     train_score, test_score = train()
#     train_scores.append(train_score)
#     test_scores.append(test_score)
#     print(f'time since beginning: {time.time() - t}')

# print('Done')

# #### plotting   ####

# labels = ['Energy','Time','x','y','z','dir_x','dir_y','dir_z']

# train_scores = np.array(train_scores)
# test_scores = np.array(test_scores)

# # fig, ax = plt.subplots(2,1)

# # for feature,label in zip(range(8),labels):
# #     ax[0].plot(train_scores[:,feature],label=label)
# #     ax[1].plot(test_scores[:,feature],label=label)

# # ax[0].set_title('Train Scores')
# # ax[1].set_title('Test Scores')
# # ax[0].legend(loc = 2,ncol = 4)
# # # ax[1].legend()

# fig, ax = plt.subplots(figsize = (16,8),nrows=4,ncols=2)
# ax = ax.flatten()

# for feature,label in zip(range(8),labels):
#     ax[feature].plot(train_scores[:,feature],c='k',label = label)
#     # ax[feature].plot(test_scores[:,feature],ls='--',c=ax[feature].get_lines()[0].get_color(),label = label)
#     ax[feature].plot(test_scores[:,feature],ls='--',c='r',label = label)
#     z_train = train_scores[-1,feature]
#     z_test = test_scores[-1,feature]
#     ax[feature].set_title(label+f' Final scores: Train = {z_train} Test = {z_test}')

# print('plotting')

# def zoom(axes,ZOOM):
#     for f,ax in enumerate(axes):
#         mini = min(train_scores[-1,f],test_scores[-1,f])
#         maxi = max(train_scores[-1,f],test_scores[-1,f])
#         ax.set_ylim(mini - ZOOM*(maxi-mini), maxi + ZOOM*(maxi-mini))
#     fig.canvas.draw()
#     return

# fig.tight_layout()
# fig.show()
# ####            ####