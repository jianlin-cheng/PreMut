import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

import pickle
# from PDB import *
from tqdm import tqdm
from PDB_all_atom import *
from sklearn.model_selection import train_test_split
import random
class My_Dataset(Dataset):
    def __init__(self, csv_file,cluster_dict,training=True,test_ratio = 0.1, transform=None, target_transform=None):
        self.df = pd.read_csv(csv_file)
        self.cluster_dict = cluster_dict
        self.test_ratio = test_ratio
        self.training = training
    def __len__(self):
        return len(self.read_cluster_dict()[-1])

    def read_cluster_dict(self):
        with open(self.cluster_dict,'rb') as f:
            cluster = pickle.load(f)
        key_list = []

        for key,value in cluster.items():
            key_list.append(key)
        test_size = int(len(key_list) * self.test_ratio)
        train_key_list, test_key_list = train_test_split(key_list,test_size=test_size,random_state=42)
        # print(len(train_key_list),len(test_key_list))
        train_cluster_dict = {}
        test_cluster_dict = {}
        for key in train_key_list:
            train_cluster_dict[key] = cluster[key]
        for key in test_key_list:
            test_cluster_dict[key] = cluster[key]
        
        if self.training == True:
            return train_cluster_dict, train_key_list
        else:
            return test_cluster_dict, test_key_list

        
        
        

    def __getitem__(self, idx):
        # print(idx)
        cluster, keys= self.read_cluster_dict()
        pick_a_mutant_cluster = cluster[keys[idx]]
        # random_number = random.randint(0,len(pick_a_mutant_cluster)-1)
        # mutant = pick_a_mutant_cluster[random_number]
        # mutant_pdb = mutant.split('_')[0]
        # mutant_chain = mutant.split('_')[1]
        # possible_wilds = self.df.loc[self.df['Mutated_PDB']==mutant][['Mutation INFO','Possible Wilds']]
        # # print('Length Possible Wilds: {0}'.format(len(possible_wilds)))
        # random_number_for_wilds = random.randint(0,len(possible_wilds)-1)
        # mutation_info = possible_wilds['Mutation INFO'].iloc[random_number_for_wilds]
        # wild = possible_wilds['Possible Wilds'].iloc[random_number_for_wilds]
        # # print(mutant,mutation_info,wild)
        # pdb_reader = PDBReader_All_Atom(pdb_dir=PDB_REINDEXED_DIR,mutant_pdb=mutant,wild_pdb=wild,mutation_info=mutation_info)
        # graph, label_coords = pdb_reader.pdb_to_graph()
        # return graph, label_coords
        return pick_a_mutant_cluster
        

class My_Dataset_2023(Dataset):
    def __init__(self, csv_file,cluster_dict,training=True,test_ratio = 0.1, transform=None, target_transform=None):
        self.df = pd.read_csv(csv_file)
        self.cluster_dict = cluster_dict
        self.test_ratio = test_ratio
        self.training = training
    def __len__(self):
        return len(self.read_cluster_dict()[-1])

    def read_cluster_dict(self):
        with open(self.cluster_dict,'rb') as f:
            cluster = pickle.load(f)
        key_list = []

        for key,value in cluster.items():
            key_list.append(key)
        return cluster, key_list
        test_size = int(len(key_list) * self.test_ratio)
        train_key_list, test_key_list = train_test_split(key_list,test_size=test_size,random_state=42)
        # print(len(train_key_list),len(test_key_list))
        train_cluster_dict = {}
        test_cluster_dict = {}
        for key in train_key_list:
            train_cluster_dict[key] = cluster[key]
        for key in test_key_list:
            test_cluster_dict[key] = cluster[key]
        
        if self.training == True:
            return train_cluster_dict, train_key_list
        else:
            return test_cluster_dict, test_key_list

        
        
        

    def __getitem__(self, idx):
        # print(idx)
        cluster, keys= self.read_cluster_dict()
        pick_a_mutant_cluster = cluster[keys[idx]]
        # random_number = random.randint(0,len(pick_a_mutant_cluster)-1)
        # mutant = pick_a_mutant_cluster[random_number]
        # mutant_pdb = mutant.split('_')[0]
        # mutant_chain = mutant.split('_')[1]
        # possible_wilds = self.df.loc[self.df['Mutated_PDB']==mutant][['Mutation INFO','Possible Wilds']]
        # # print('Length Possible Wilds: {0}'.format(len(possible_wilds)))
        # random_number_for_wilds = random.randint(0,len(possible_wilds)-1)
        # mutation_info = possible_wilds['Mutation INFO'].iloc[random_number_for_wilds]
        # wild = possible_wilds['Possible Wilds'].iloc[random_number_for_wilds]
        # # print(mutant,mutation_info,wild)
        # pdb_reader = PDBReader_All_Atom(pdb_dir=PDB_REINDEXED_DIR,mutant_pdb=mutant,wild_pdb=wild,mutation_info=mutation_info)
        # graph, label_coords = pdb_reader.pdb_to_graph()
        # return graph, label_coords
        return pick_a_mutant_cluster



dataset = My_Dataset(csv_file='MutData2022.csv',cluster_dict='MutData2022_cluster_dict',training=True)
test_dataset = My_Dataset(csv_file='MutData2022.csv',cluster_dict='MutData2022_cluster_dict',training=False)
test_dataset_2023 = My_Dataset_2023(csv_file='MutData2023.csv',cluster_dict='MutData2023_cluster_dict')
train_set_size = int(len(dataset) * 0.9)
valid_set_size = len(dataset) - train_set_size
train_ds, validate_ds = random_split(dataset, [train_set_size, valid_set_size])

# print(train_ds)
# print(validate_ds)
train_dataloader = DataLoader(dataset=train_ds,batch_size=1)
validation_dataloader = DataLoader(dataset=validate_ds,batch_size=1)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)

# for epoch in range(10):
# for idx, batch in tqdm(enumerate(train_dataloader)):
#     print(idx)
#     print(batch)
#     break

# dataset_lst = []
# for i in range(dataset.__len__()):
#     dataset_lst += dataset.__getitem__(i)
# print(len(dataset_lst))

# test_dataset_lst = []
# for i in range(test_dataset.__len__()):
#     test_dataset_lst += test_dataset.__getitem__(i)
# count = 0
# for item in test_dataset_lst:
#     if item in dataset_lst:
#         count += 1

# df = pd.read_csv('MutData2022.csv')
# wild_dataset_lst = []
# wild_test_dataset_lst = []
# for item in dataset_lst:
#     wild = df.loc[df['Mutated_PDB']== item]['Possible Wilds'].tolist()
#     wild_dataset_lst += wild

# for item in test_dataset_lst:
#     wild = df.loc[df['Mutated_PDB']== item]['Possible Wilds'].tolist()
#     wild_test_dataset_lst += wild
# count = 0
# for item in wild_test_dataset_lst:
#     if item in wild_dataset_lst:
#         print(item)
#         count += 1

# print(count)


        

# print(train_ds.__getitem__(idx=0))