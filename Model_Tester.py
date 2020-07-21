import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch_geometric.data import DataLoader, InMemoryDataset
import time
import importlib
from sklearn.metrics import roc_curve

classifiers = ['energy','type','class']
classifying = classifiers[2]

class LoadDataset(InMemoryDataset):
    def __init__(self, name, root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets'):
        super(LoadDataset, self).__init__(root)
        self.data, self.slices = torch.load(root + '/' + name)
    
    @property
    def processed_file_names(self):
        return os.listdir(root)

print(f'Loading datasets for {classifying} prediction')
train_dataset = LoadDataset(f'train_{classifying}')
test_dataset = LoadDataset(f'test_{classifying}')

train_loader = DataLoader(train_dataset, batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

print('Loads model')
#Define model:
#The syntax is for model i: from Models.Model{i} import Net
import Models.Model2 as Model
Model = importlib.reload(Model)

print(f'remember to double check that model is suitable for {classifying} prediction')
if not torch.cuda.is_available(): print('CUDA not available') 

print(f'Memory before .to(device) {torch.cuda.memory_allocated()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.Net().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print(f'Memory after .to(device) {torch.cuda.memory_allocated()}')

# #For loading existing model and optimizer parameters.
# print('Loading existing model and optimizer states')
# state = torch.load('Trained_Models/Model7_Class.pt')
# model.load_state_dict(state['model_state_dict'])
# optimizer.load_state_dict(state['optimizer_state_dict'])

def save_model(name):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},'Trained_Models/'+name)
    print('Model saved')

if classifying == classifiers[0]: #Energy prediction
    crit = torch.nn.MSELoss()

    def cal_acc(output,label):
        return (output.view(-1) - label).float().mean().item()

elif classifying == classifiers[1]: #Type prediction
    crit = torch.nn.NLLLoss()

    def cal_acc(output,label):
        return output.argmax(dim=1).eq(label).float().mean().item()

elif classifying == classifiers[2]: #Class prediction
    crit = torch.nn.NLLLoss()

    def cal_acc(output,label):
        return output.argmax(dim=1).eq(label).float().mean().item()


batch_loss, batch_acc = [], []
def train():
    model.train()

    for data in train_loader:
        data = data.to(device)
        label = data.y
        optimizer.zero_grad()
        output = model(data)
        del data
        loss = crit(output,label)
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())
        batch_acc.append(cal_acc(output,label))

    torch.cuda.empty_cache()
    return

def test():
    acc = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            label = data.y
            output = model(data)

            del data

            acc += cal_acc(output,label)
        torch.cuda.empty_cache()
    return acc/test_loader.__len__()

def ROC():
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for data in test_loader:
            label = data.y
            data = data.to(device)
            output = model(data)

            del data

            scores += output.cpu()[:,0].tolist()

            del output
            # print(torch.cuda.memory_allocated())
            labels += label.cpu().tolist()
        torch.cuda.empty_cache()
    return scores, labels


def epochs(i,mean_length=500):
    print('Begins training')
    t0 = time.time()
    for epoch in range(i):
        print(f'Epoch: {epoch}')
        train()
        mean_loss = np.mean(batch_loss[-mean_length:])
        mean_acc = np.mean(batch_acc[-mean_length:])
        std_acc = np.std(batch_acc[-mean_length:])
        print(f'Mean of last {mean_length} batches; loss: {mean_loss}, acc: {mean_acc} +- {std_acc}')
        print(f'time since beginning: {time.time() - t0}')
        print('Done')

def plot():
    fig, ax = plt.subplots(nrows=2,sharex=True)
    ax[0].plot(batch_loss)
    ax[1].plot(batch_acc)
    fig.show()
