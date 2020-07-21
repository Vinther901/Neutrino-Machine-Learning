from torch_geometric.data import DataListLoader, InMemoryDataset
import numpy as np
import pandas as pd
import os
import torch
def create_class_dataset():
    class LoadDataset(InMemoryDataset):
        def __init__(self, root):
            super(LoadDataset, self).__init__(root)
            self.data, self.slices = torch.load(root+'/processed/processed')

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

        def process(self):
            pass

    print('Loads data')
    dataset = LoadDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset')
    dataset_background = LoadDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/dataset_background')


    dataset.data.y = torch.tensor(np.zeros(300000),dtype=torch.int64)
    dataset.slices['y'] = torch.tensor(np.arange(300000+1))

    dataset_background.data.y = torch.tensor(np.ones(dataset_background.len()),dtype=torch.int64)
    dataset_background.slices['y'] = torch.tensor(np.arange(dataset_background.len() + 1))

    subsize = 30000

    nu_e_ind = np.random.choice(np.arange(100000),subsize,replace=False).tolist()
    nu_t_ind = np.random.choice(np.arange(100000,200000),subsize,replace=False).tolist()
    nu_m_ind = np.random.choice(np.arange(200000,300000),subsize,replace=False).tolist()
    muon_ind = np.random.choice(np.arange(dataset_background.len()),3*subsize,replace=False).tolist()

    train_dataset = dataset[nu_e_ind] + dataset[nu_t_ind] + dataset[nu_m_ind] + dataset_background[muon_ind]

    test_ind_e = np.arange(100000)[pd.Series(np.arange(100000)).isin(nu_e_ind).apply(lambda x: not(x))].tolist()
    test_ind_t = np.arange(100000,200000)[pd.Series(np.arange(100000,200000)).isin(nu_t_ind).apply(lambda x: not(x))].tolist()
    test_ind_m = np.arange(200000,300000)[pd.Series(np.arange(200000,300000)).isin(nu_m_ind).apply(lambda x: not(x))].tolist()
    test_ind_muon = np.arange(dataset_background.len())[pd.Series(np.arange(dataset_background.len())).isin(muon_ind).apply(lambda x: not(x))].tolist()

    test_dataset = dataset[test_ind_e] + dataset[test_ind_t] + dataset[test_ind_m] + dataset_background[test_ind_muon]

    for train_list in DataListLoader(train_dataset,batch_size=len(train_dataset)):
        pass

    for test_list in DataListLoader(test_dataset,batch_size=len(test_dataset)):
        pass


    class MakeDataset(InMemoryDataset):
        def __init__(self, root,data_list,name):
            super(MakeDataset, self).__init__(root)
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data,self.slices),root+'/'+name)

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

        def process(self):
            pass

    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',train_list,'train_class')
    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',test_list,'test_class')

def create_type_dataset():
    class LoadDataset(InMemoryDataset):
        def __init__(self, root):
            super(LoadDataset, self).__init__(root)
            self.data, self.slices = torch.load(root+'/processed/processed')

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

        def process(self):
            pass

    print('Loads data')
    dataset = LoadDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset')

    dataset.data.y = torch.tensor([np.zeros(100000),np.ones(100000),np.ones(100000)*2],dtype=torch.int64).view(-1)
    dataset.slices['y'] = torch.tensor(np.arange(300000+1))

    dataset = dataset.shuffle()
    
    train_dataset = dataset[:200000]
    print(len(train_dataset))
    test_dataset = dataset[200000:]

    for train_list in DataListLoader(train_dataset,batch_size=len(train_dataset)):
        pass

    for test_list in DataListLoader(test_dataset,batch_size=len(test_dataset)):
        pass

    class MakeDataset(InMemoryDataset):
        def __init__(self, root,data_list,name):
            super(MakeDataset, self).__init__(root)
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data,self.slices),root+'/'+name)

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets')

        def process(self):
            pass
    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',train_list,'train_type')
    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',test_list,'test_type')

def create_energy_dataset():
    class LoadDataset(InMemoryDataset):
        def __init__(self, root):
            super(LoadDataset, self).__init__(root)
            self.data, self.slices = torch.load(root+'/processed/processed')

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

        def process(self):
            pass

    print('Loads data')
    dataset = LoadDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset')

    dataset.data.y = dataset.data.y[::8]
    dataset.slices['y'] = torch.tensor(np.arange(300000+1))

    dataset = dataset.shuffle()
    
    train_dataset = dataset[:200000]
    print(len(train_dataset))
    test_dataset = dataset[200000:]

    for train_list in DataListLoader(train_dataset,batch_size=len(train_dataset)):
        pass

    for test_list in DataListLoader(test_dataset,batch_size=len(test_dataset)):
        pass

    class MakeDataset(InMemoryDataset):
        def __init__(self, root,data_list,name):
            super(MakeDataset, self).__init__(root)
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data,self.slices),root+'/'+name)

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets')

        def process(self):
            pass
    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',train_list,'train_energy')
    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',test_list,'test_energy')

def create_background_dataset():
    class LoadDataset(InMemoryDataset):
        def __init__(self, root):
            super(LoadDataset, self).__init__(root)
            self.data, self.slices = torch.load(root+'/processed/processed')

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

        def process(self):
            pass

    print('Loads data')
    dataset = LoadDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset')
    dataset_background = LoadDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/dataset_background')


    # dataset.data.y = torch.tensor(np.zeros(300000),dtype=torch.int64)
    # dataset.slices['y'] = torch.tensor(np.arange(300000+1))

    # dataset_background.data.y = torch.tensor(np.ones(dataset_background.len()),dtype=torch.int64)
    # dataset_background.slices['y'] = torch.tensor(np.arange(dataset_background.len() + 1))

    subsize = 30000

    nu_e_ind = np.random.choice(np.arange(100000),subsize,replace=False).tolist()
    nu_t_ind = np.random.choice(np.arange(100000,200000),subsize,replace=False).tolist()
    nu_m_ind = np.random.choice(np.arange(200000,300000),subsize,replace=False).tolist()
    muon_ind = np.random.choice(np.arange(dataset_background.len()),3*subsize,replace=False).tolist()

    train_dataset = dataset[nu_t_ind] + dataset[nu_e_ind] + dataset[nu_m_ind] + dataset_background[muon_ind]

    test_ind_e = np.arange(100000)[pd.Series(np.arange(100000)).isin(nu_e_ind).apply(lambda x: not(x))].tolist()
    test_ind_t = np.arange(100000,200000)[pd.Series(np.arange(100000,200000)).isin(nu_t_ind).apply(lambda x: not(x))].tolist()
    test_ind_m = np.arange(200000,300000)[pd.Series(np.arange(200000,300000)).isin(nu_m_ind).apply(lambda x: not(x))].tolist()
    test_ind_muon = np.arange(dataset_background.len())[pd.Series(np.arange(dataset_background.len())).isin(muon_ind).apply(lambda x: not(x))].tolist()

    test_dataset = dataset[test_ind_t] + dataset[test_ind_e] + dataset[test_ind_m] + dataset_background[test_ind_muon]

    for train_list in DataListLoader(train_dataset,batch_size=len(train_dataset)):
        pass

    for test_list in DataListLoader(test_dataset,batch_size=len(test_dataset)):
        pass


    class MakeDataset(InMemoryDataset):
        def __init__(self, root,data_list,name):
            super(MakeDataset, self).__init__(root)
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data,self.slices),root+'/'+name)

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

        def process(self):
            pass

    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',train_list,'t30k_e30k_m30k_muon90k')
    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',test_list,'t70k_e70k_m70k_muon38730')

def create_neutrino_dataset():
    class LoadDataset(InMemoryDataset):
        def __init__(self, root):
            super(LoadDataset, self).__init__(root)
            self.data, self.slices = torch.load(root+'/processed/processed')

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset/processed')

        def process(self):
            pass

    print('Loads data')
    dataset = LoadDataset(root = 'C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/copy_dataset')
    
    subsize = 70000

    nu_e_ind = np.random.choice(np.arange(100000),subsize,replace=False).tolist()
    nu_t_ind = np.random.choice(np.arange(100000,200000),subsize,replace=False).tolist()
    nu_m_ind = np.random.choice(np.arange(200000,300000),subsize,replace=False).tolist()

    train_dataset = dataset[nu_t_ind] + dataset[nu_e_ind] + dataset[nu_m_ind]

    test_ind_e = np.arange(100000)[pd.Series(np.arange(100000)).isin(nu_e_ind).apply(lambda x: not(x))].tolist()
    test_ind_t = np.arange(100000,200000)[pd.Series(np.arange(100000,200000)).isin(nu_t_ind).apply(lambda x: not(x))].tolist()
    test_ind_m = np.arange(200000,300000)[pd.Series(np.arange(200000,300000)).isin(nu_m_ind).apply(lambda x: not(x))].tolist()
    # test_ind_muon = np.arange(dataset_background.len())[pd.Series(np.arange(dataset_background.len())).isin(muon_ind).apply(lambda x: not(x))].tolist()

    test_dataset = dataset[test_ind_t] + dataset[test_ind_e] + dataset[test_ind_m]


    for train_list in DataListLoader(train_dataset,batch_size=len(train_dataset)):
        pass

    for test_list in DataListLoader(test_dataset,batch_size=len(test_dataset)):
        pass

    class MakeDataset(InMemoryDataset):
        def __init__(self, root,data_list,name):
            super(MakeDataset, self).__init__(root)
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data,self.slices),root+'/'+name)

        @property
        def processed_file_names(self):
            return os.listdir('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets')

        def process(self):
            pass
    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',train_list,'t70k_e70k_m70k')
    MakeDataset('C:/Users/jv97/Desktop/github/Neutrino-Machine-Learning/train_test_datasets',test_list,'t30k_e30k_m30k')
