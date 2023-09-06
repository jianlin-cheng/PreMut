import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# from dataset import loader
from rmsd import rmsd
import pandas as pd
from biopandas.pdb import PandasPdb

import pytorch_lightning as pl
from constants import *
import random
from PDB_all_atom import PDBReader_All_Atom
from sklearn.metrics import mean_squared_error,f1_score
from csv import writer, DictWriter
class LitGat(pl.LightningModule):
    def __init__(self, MODEL,csv_file= 'MutData2022.csv',effect_prediction=EFFECT_PREDICTION,loss_weight= 1,mask=MASK):
        super().__init__()
        self.model = MODEL
        self.df = pd.read_csv(csv_file)
        self.loss_weight = loss_weight
        self.mask = mask
        # self.effect_prediction = effect_prediction
        self.save_hyperparameters()
        
    
    def output_to_pdb(self,coords,node_one_hot_sequence_atoms,node_one_hot_sequence_residues,name='temp.pdb'):
        node_one_hot_sequence_atoms = node_one_hot_sequence_atoms.tolist()
        node_one_hot_sequence_residues = node_one_hot_sequence_residues.tolist()
        # print(len(node_one_hot_sequence_residues))
        residue_num = 1
        record_name = []
        atom_number = []
        blank_1 = []
        atom_name = []
        alt_loc = []
        residue_name = []
        blank_2 = []
        chain_id = []
        residue_number = []
        insertion = []
        blank_3 = []
        x_coord = []
        y_coord = []
        z_coord = []
        occupancy = []
        b_factor = []
        blank_4 = []
        segment_id = []
        element_symbol = []
        charge = []
        line_idx = []

        for idx, xyz in enumerate(coords):
            atom_type = ATOMS[node_one_hot_sequence_atoms[idx].index(1)]
            # residue_type = node_one_hot_sequence_residues[idx].index(1)
            try: 
                residue_type = BASE_AMINO_ACIDS_3_LETTER[node_one_hot_sequence_residues[idx].index(1)]
            except:
                residue_type = 'X'
            if idx == 0:
                residue_num = 1
            
            else:
                try:
                    previous_residue_type = BASE_AMINO_ACIDS_3_LETTER[node_one_hot_sequence_residues[idx-1].index(1)]
                except:
                    previous_residue_type = 'X'
                
                # if residue_type != previous_residue_type:
                #     residue_num += 1
                if atom_type == 'N':
                    residue_num += 1
                
                
            # print('ATOM {0}  {4}  {5}  {6} {7}   {1} {2} {3} {8} {9}   {10}  {11}'.format(idx+1,xyz[0],xyz[1],xyz[2],atom_type,residue_type,'A',residue_num,'1.00','0.00',atom_type[0],idx+1))
            # with open('temp.pdb','a') as f:
            #     f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20}\n')
            record_name.append('ATOM')
            atom_number.append(idx+1)
            blank_1.append('')
            atom_name.append(atom_type)
            alt_loc.append('')
            residue_name.append(residue_type)
            blank_2.append('')
            chain_id.append('A')
            residue_number.append(residue_num)
            insertion.append('')
            blank_3.append('')
            x_coord.append(xyz[0])
            y_coord.append(xyz[1])
            z_coord.append(xyz[2])
            occupancy.append(1.00)
            b_factor.append(0.00)
            blank_4.append('')
            segment_id.append('')
            element_symbol.append(atom_type[0])
            charge.append(0.00)
            line_idx.append(idx+1)
        dict = {'record_name': record_name,'atom_number': atom_number,'blank_1': blank_1,'atom_name': atom_name, 'alt_loc': alt_loc, 'residue_name': residue_name,'blank_2': blank_2,'chain_id': chain_id
        ,'residue_number': residue_number, 'insertion': insertion, 
        'blank_3': blank_3, 'x_coord': x_coord, 'y_coord': y_coord, 'z_coord': z_coord, 'occupancy': occupancy, 'b_factor': b_factor, 'blank_4': blank_4, 'segment_id': segment_id,
        'element_symbol': element_symbol, 'charge': charge, 'line_idx': line_idx}
        df = pd.DataFrame(dict)
        ppdb = PandasPdb()
        ppdb.df['ATOM'] = df
        # print(ppdb)
        ppdb.to_pdb(path=name,records=None,gz=False,append_newline=True)

    def prepare_graph(self,batch,state='train'):
        pick_a_mutant_cluster = batch
        random_number = random.randint(0,len(pick_a_mutant_cluster)-1)
        mutant = pick_a_mutant_cluster[random_number][0]
        # while mutant == '2XWK_A' or mutant == '7KGT_A' or mutant == '2BWX_A':
        #     random_number = random.randint(0,len(pick_a_mutant_cluster)-1)
        #     mutant = pick_a_mutant_cluster[random_number][0]
        
        mutant_pdb = mutant.split('_')[0]
        mutant_chain = mutant.split('_')[1]
        possible_wilds = self.df.loc[self.df['Mutated_PDB']==mutant][['Mutation INFO','Possible Wilds']]
        # print('Length Possible Wilds: {0}'.format(len(possible_wilds)))
        if len(possible_wilds) == 0:
            print(mutant)
        random_number_for_wilds = random.randint(0,len(possible_wilds)-1)
        mutation_info = possible_wilds['Mutation INFO'].iloc[random_number_for_wilds]
        wild = possible_wilds['Possible Wilds'].iloc[random_number_for_wilds]
        # print(mutant,mutation_info,wild)
        pdb_reader = PDBReader_All_Atom(pdb_dir=PDB_REINDEXED_DIR,mutant_pdb=mutant,wild_pdb=wild,mutation_info=mutation_info,state=state)
        graph, label_coords = pdb_reader.pdb_to_graph()
        return graph, label_coords, [wild, mutant, mutation_info]
    
    def inverse_min_max_scale(self,tensor_scaled,min_val=-1,max_val=1,min_tensor=MIN,max_tensor=MAX):
        # Calculate the minimum and maximum values of the tensor
        min_tensor = min_tensor
        max_tensor = max_tensor

        # tensor_scaled = (tensor_scaled - min_val)/(max_val - min_val)
        # tensor = tensor_scaled * (max_tensor-min_tensor) + min_tensor

        tensor = (((tensor_scaled-min_val)*(max_tensor-min_tensor))/(max_val-min_val)) + min_tensor


        return tensor
    
    def calculate_TM_Score_and_RMSD(self, ground_truth,prediciton):
        # ground_truth_dir = '/bml/sajid/Mutation_Data_Preparation/PDB'
        
        command = './TMscore {0} {1} -outfmt 2'.format(ground_truth,prediciton)
        command_lst = command.split()

        result = subprocess.check_output(command, shell=True)
        result = result.decode()
        # print(result)
        try:
            tmscore = float(result.split('\n')[1].split('\t')[3])
            rmsd = float(result.split('\n')[1].split('\t')[4])
            return tmscore,rmsd
        except:
            print(result)
            tmscore = float(result.split('\n')[-1].split('\t')[3])
            rmsd = float(result.split('\n')[-1].split('\t')[4])
            return tmscore,rmsd
    
    def save_results_to_csv(self,column_names, lst):
        field_names = column_names
 
        # Dictionary that we want to add as a new row
        dict = {column_names[0]: lst[0], column_names[1]: lst[1],
                column_names[2]: lst[2], column_names[3]: lst[3],column_names[4]: lst[4],column_names[5]: lst[5],column_names[6]: lst[6]}
        
        # Open CSV file in append mode
        # Create a file object for this file
        with open('esm_only.csv', 'a') as f_object:
        
            # Pass the file object and a list
            # of column names to DictWriter()
            # You will get a object of DictWriter
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(dict)
        
            # Close the file object
            f_object.close()
    def create_mask(self, node_one_hot_sequence_residues):
        mask = torch.zeros(len(node_one_hot_sequence_residues),3)
        for i in range(len(node_one_hot_sequence_residues)):
            if -1 in node_one_hot_sequence_residues[i]:
                mask[i,:] = 1
        return mask
    def step(self, batch, batch_idx,state):
        # training_step defines the train loop.
        
        graph, label, info = self.prepare_graph(batch=batch,state=state)
        graph = graph.cuda()
        label = label.cuda()
        
        
        # graph, ddg, effect = batch
        x = (graph.node_coords, graph.node_one_hot_sequence,graph.node_one_hot_sequence_residues)
        # x = (graph.node_coords, graph.node_one_hot_sequence,graph.node_one_hot_sequence_residues,graph.node_esm_features)
        mask_tensor = self.create_mask(node_one_hot_sequence_residues=graph.node_one_hot_sequence_residues).cuda()
        # x = torch.cat((graph.node_coords,graph.node_one_hot_sequence,graph.node_one_hot_sequence_residues),dim=1)
        if state == 'test' or state =='validation':
            self.model.eval()
        else:
            self.model.train()
        out = self.model(node_feats=x, edge_index=graph.edge_index, edge_attr=graph.edge_distances,batch=graph.batch)
        if self.mask == True:
            out = out * mask_tensor
        # if self.effect_prediction == True:
        #     loss = F.mse_loss(out_ddg, ddg) + F.cross_entropy(out_effect,effect)
        # else:
        #     loss = F.mse_loss(out_ddg,ddg)
        out = out + graph.node_coords

        loss = self.loss_weight * F.mse_loss(out,label)

        self.log('{0}/loss'.format(state), loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size = 1)
        if state == 'test' or state=='validation':
            out_np = out.detach().clone().cpu().numpy()
            label_np = label.detach().clone().cpu().numpy()
            node_coords_np = graph.node_coords.detach().clone().cpu().numpy()
            node_one_hot_sequence_np = graph.node_one_hot_sequence.detach().clone().cpu().numpy()
            node_one_hot_sequence_residues_np = graph.node_one_hot_sequence_residues.detach().clone().cpu().numpy()
            self.output_to_pdb(coords=out_np,node_one_hot_sequence_atoms=node_one_hot_sequence_np,node_one_hot_sequence_residues=node_one_hot_sequence_residues_np)
            self.output_to_pdb(coords=label_np,node_one_hot_sequence_atoms=node_one_hot_sequence_np,node_one_hot_sequence_residues=node_one_hot_sequence_residues_np,name='temp_mutant.pdb')
            if state == 'test':
                self.output_to_pdb(coords=node_coords_np,node_one_hot_sequence_atoms=node_one_hot_sequence_np,node_one_hot_sequence_residues=node_one_hot_sequence_residues_np,name='ESM_Features_Results/{0}_{1}_{2}_input.pdb'.format(info[0],info[2],info[1]))
                self.output_to_pdb(coords=out_np,node_one_hot_sequence_atoms=node_one_hot_sequence_np,node_one_hot_sequence_residues=node_one_hot_sequence_residues_np,name='ESM_Features_Results/{0}_{1}_{2}_predicted.pdb'.format(info[0],info[2],info[1]))
                self.output_to_pdb(coords=label_np,node_one_hot_sequence_atoms=node_one_hot_sequence_np,node_one_hot_sequence_residues=node_one_hot_sequence_residues_np,name='ESM_Features_Results/{0}_{1}_{2}_groundtruth.pdb'.format(info[0],info[2],info[1]))
            # out_np = self.inverse_min_max_scale(tensor_scaled=out_np)
            # label_np = self.inverse_min_max_scale(tensor_scaled= label_np)
            # error = rmsd(out_np,label_np)
            # error_rmse = mean_squared_error(label_np,out_np,squared=False)
            error_tmscore, error_rmsd = self.calculate_TM_Score_and_RMSD(ground_truth='temp_mutant.pdb',prediciton='temp.pdb')
            if state == 'test':
                error_tmscore_input, error_rmsd_input = self.calculate_TM_Score_and_RMSD(ground_truth='temp_mutant.pdb',prediciton='ESM_Features_Results/{0}_{1}_{2}_input.pdb'.format(info[0],info[2],info[1]))
            if state == 'test':
                column_names = ['wild_pdb','mutation_info','mutant_pdb','tmscore','rmsd','tmscore_with_input','rmsd_with_input']
                results = [info[0],info[2],info[1],error_tmscore,error_rmsd,error_tmscore_input,error_rmsd_input]
                self.save_results_to_csv(column_names=column_names,lst=results)

            self.log('{0}/rmsd'.format(state),error_rmsd,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size = 1)
            self.log('{0}/tmscore'.format(state),error_tmscore,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size = 1)
            
            
            


        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,state='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,state='validation')

    def test_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,state='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



