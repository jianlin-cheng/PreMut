import os
import pandas as pd
from constants import *
from biopandas.pdb import PandasPdb
from tqdm import tqdm
import numpy as np
import subprocess
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
import pickle
import torch
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class PDBReader_All_Atom():
    def __init__(self,pdb_dir,mutant_pdb,wild_pdb,mutation_info,state='train') -> None:
        self.pdb_dir = pdb_dir
        self.mutant_pdb = mutant_pdb.split('_')[0].lower()
        self.wild_pdb = wild_pdb.split('_')[0].lower()
        self.mutant_chain = mutant_pdb.split('_')[-1]
        self.wild_chain = wild_pdb.split('_')[-1]
        self.mutation_info = mutation_info
        self.initial_residue = self.mutation_info.split('_')[0]
        self.pos = int(self.mutation_info.split('_')[1])
        self.mutated_residue = self.mutation_info.split('_')[2]
        self.state = state
        self.name = wild_pdb+'_'+mutation_info + '_' + mutant_pdb
        
    
    def get_pdb_df(self,pdb_file,chain_id):
        pdb_path = os.path.join(self.pdb_dir,pdb_file+'.pdb')
        pdb = PandasPdb().read_pdb(pdb_path)
        pdb_df = pdb.df['ATOM'][(pdb.df['ATOM']['chain_id'] == chain_id) & (pdb.df['ATOM']['atom_name'].isin(ATOMS)) & (pdb.df['ATOM']['alt_loc'].isin(['A','']))]
        return pdb_df
    
    def get_ppdb(self,pdb_file,chain_id):
        pdb_path = os.path.join(self.pdb_dir,pdb_file+'.pdb')
        pdb = PandasPdb().read_pdb(pdb_path)
        pdb.df['ATOM'] = pdb.df['ATOM'][(pdb.df['ATOM']['chain_id'] == chain_id) & (pdb.df['ATOM']['atom_name'].isin(ATOMS)) & (pdb.df['ATOM']['alt_loc'].isin(['A','']))]

        return pdb
    
    def min_max_scale(self,tensor, min_val=-1, max_val=1,min_tensor=MIN,max_tensor=MAX):
        """
        Perform min-max scaling on a 2D tensor.

        Parameters
        ----------
        tensor : numpy.ndarray
            The 2D tensor to be scaled.
        min_val : float, optional
            The minimum value to scale the tensor to, by default 0
        max_val : float, optional
            The maximum value to scale the tensor to, by default 1

        Returns
        -------
        numpy.ndarray
            The scaled 2D tensor.
        """
        # Calculate the minimum and maximum values of the tensor
        min_tensor = min_tensor
        max_tensor = max_tensor

        # Scale the tensor
        tensor_scaled = (tensor - min_tensor) / (max_tensor - min_tensor)
        tensor_scaled = tensor_scaled * (max_val - min_val) + min_val

        return tensor_scaled
    
    def inverse_min_max_scale(self,tensor_scaled,min_val=-1,max_val=1,min_tensor=MIN,max_tensor=MAX):
        # Calculate the minimum and maximum values of the tensor
        min_tensor = min_tensor
        max_tensor = max_tensor

        tensor_scaled = (tensor_scaled - min_val)/(max_val - min_val)
        tensor = tensor_scaled * (max_tensor-min_tensor) + min_tensor


        return tensor


    

    def generate_random_point_on_sphere(self, center, radius: float):
        """
        Generates a random point on the surface of a sphere with a given radius from a center point in 3D space.

        Arguments:
        - center: A tuple of three values representing the center coordinates (x, y, z).
        - radius: The desired radius.

        Returns:
        - A tuple of three values representing the random point coordinates (x, y, z).
        """
        import random
        import math
        from typing import Tuple
        x, y, z = center

        # Generate random angles in spherical coordinates
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)

        # Calculate the random point coordinates on the surface of the sphere
        x_surface = x + radius * math.sin(phi) * math.cos(theta)
        y_surface = y + radius * math.sin(phi) * math.sin(theta)
        z_surface = z + radius * math.cos(phi)

        random_point = [x_surface, y_surface, z_surface]

        return random_point


    def prepare_wild_xyz_for_input(self,wild_xyz):
        
        number_of_atoms = len(wild_xyz['residue_number'].to_list())
        # length_wild = wild_xyz['residue_number'].to_list()[number_of_atoms-1]
        length_wild = int(wild_xyz['residue_number'].iloc[-1]) - int(wild_xyz['residue_number'].iloc[0]) + 1
        # print(number_of_atoms,length_wild)
        flag_length = int(wild_xyz['residue_number'].iloc[0])-1
        # print(length_wild,flag_length,number_of_atoms)
        # print(mutant_xyz.loc[mutant_xyz['residue_number']==pos+1])
        # print(number_of_atoms)
        wild_xyz_for_input = []
        for idx in range(1+flag_length,length_wild+flag_length+1):
            if idx != self.pos+flag_length+1:
                xyz_for_that_residue = wild_xyz.loc[wild_xyz['residue_number']==idx][['atom_name','x_coord','y_coord','z_coord']]
                
                residue_name = wild_xyz.loc[wild_xyz['residue_number']==idx]['residue_name'].iloc[0]
                
                atoms_for_this_particular_residue = AMINO_ACID_ATOM[residue_name]
                if idx == length_wild + flag_length:
                    atoms_for_this_particular_residue = atoms_for_this_particular_residue + ['OXT']
                atoms_available_in_pdb_for_this_particular_residue = wild_xyz.loc[wild_xyz['residue_number']==idx]['atom_name'].to_list()
                # if len(atoms_available_in_pdb_for_this_particular_residue)!=len(atoms_for_this_particular_residue):
                #     print(residue_name)
                #     print(atoms_for_this_particular_residue)
                #     print(atoms_available_in_pdb_for_this_particular_residue)
                
                # print(atoms_available_in_pdb_for_this_particular_residue)
                # print(atoms_for_this_particular_residue)
                for atoms in atoms_for_this_particular_residue:
                    if atoms in atoms_available_in_pdb_for_this_particular_residue:
                        coords = xyz_for_that_residue.loc[xyz_for_that_residue['atom_name']==atoms][['x_coord','y_coord','z_coord']]
                        for j in range(len(coords)):
                            # print(coords.iloc[j].to_list())
                            wild_xyz_for_input.append(coords.iloc[j].to_list())
                    else:

                        wild_xyz_for_input.append([0,0,0])

                # for j in range(len(xyz_for_that_residue)):
                #     # print(xyz_for_that_residue.iloc[j].to_list())
                #     wild_xyz_for_input.append(xyz_for_that_residue.iloc[j].to_list())
            else:
                xyz_for_that_residue = wild_xyz.loc[wild_xyz['residue_number']==idx][['residue_name','atom_name','x_coord','y_coord','z_coord']]
                # print(xyz_for_that_residue)
                # print(AMINO_ACIDS_1_TO_3[wild_residue])
                # print(AMINO_ACIDS_1_TO_3[mutated_residue])
                # atoms_for_this_particular_residue = AMINO_ACID_ATOM[residue_name]
                # atoms_available_in_pdb_for_this_particular_residue = wild_xyz.loc[wild_xyz['residue_number']==idx]['atom_name'].to_list()
                # if len(atoms_available_in_pdb_for_this_particular_residue)!=len(atoms_for_this_particular_residue):
                #     print(residue_name)
                #     print(atoms_for_this_particular_residue)
                #     print(atoms_available_in_pdb_for_this_particular_residue)
                atoms_for_wild_residue = xyz_for_that_residue['atom_name'].to_list()
                # print(atoms_for_wild_residue)
                atoms_for_mutated_residue = AMINO_ACID_ATOM[AMINO_ACIDS_1_TO_3[self.mutated_residue]]
                # if idx == length_wild + flag_length:
                #     atoms_for_mutated_residue.append('OXT')
                for atom in atoms_for_mutated_residue:
                    if atom in atoms_for_wild_residue:
                        coords = xyz_for_that_residue.loc[xyz_for_that_residue['atom_name']==atom][['x_coord','y_coord','z_coord']]
                        for j in range(len(coords)):
                            # print(coords.iloc[j].to_list())
                            wild_xyz_for_input.append(coords.iloc[j].to_list())
                    else:
                        # print(atom)
                        coords_ca = xyz_for_that_residue.loc[xyz_for_that_residue['atom_name']=='CA'][['x_coord','y_coord','z_coord']]
                        for j in range(len(coords)):
                            # print(coords.iloc[j].to_list())
                            coords_ca_lst = coords_ca.iloc[j].to_list()
                        # print(self.generate_random_point_on_sphere(center=coords_ca_lst,radius=1.54))
                        # wild_xyz_for_input.append([0,0,0])
                        wild_xyz_for_input.append(self.generate_random_point_on_sphere(center=coords_ca_lst,radius=1.54))
        return np.asarray(wild_xyz_for_input)
    
    def prepare_mutant_xyz_for_ground_truth(self,mutant_xyz):
        length_mutant = int(mutant_xyz['residue_number'].iloc[-1]) - int(mutant_xyz['residue_number'].iloc[0]) + 1
        # print(length_mutant)
        flag_length = int(mutant_xyz['residue_number'].iloc[0])-1

        # print(mutant_xyz.loc[mutant_xyz['residue_number']==pos+1])
        wild_xyz_for_input = []
        for idx in range(1+flag_length,length_mutant+flag_length+1):
            
            xyz_for_that_residue = mutant_xyz.loc[mutant_xyz['residue_number']==idx][['residue_name','atom_name','x_coord','y_coord','z_coord']]
            
            residue_name = mutant_xyz.loc[mutant_xyz['residue_number']==idx]['residue_name'].iloc[0]
            
            atoms_for_this_particular_residue = AMINO_ACID_ATOM[residue_name]
            atoms_available_in_pdb_for_this_particular_residue = mutant_xyz.loc[mutant_xyz['residue_number']==idx]['atom_name'].to_list()
            if idx == length_mutant + flag_length:
                atoms_for_this_particular_residue = atoms_for_this_particular_residue + ['OXT']
            # if len(atoms_available_in_pdb_for_this_particular_residue)!=len(atoms_for_this_particular_residue):
                # print(residue_name)
                # print(atoms_for_this_particular_residue)
                # print(atoms_available_in_pdb_for_this_particular_residue)

            for atoms in atoms_for_this_particular_residue:
                if atoms in atoms_available_in_pdb_for_this_particular_residue:
                    coords = xyz_for_that_residue.loc[xyz_for_that_residue['atom_name']==atoms][['x_coord','y_coord','z_coord']]
                    for j in range(len(coords)):
                        # print(coords.iloc[j].to_list())
                        wild_xyz_for_input.append(coords.iloc[j].to_list())
                else:
                    wild_xyz_for_input.append([0,0,0])
        return np.asarray(wild_xyz_for_input)
    def do_TMalign(self,wild_ppdb,mutant_ppdb):
        if not os.path.exists('TMP'):
            os.mkdir('TMP')
        wild_ppdb.to_pdb(path='TMP/{0}.pdb'.format(self.wild_pdb), 
            records=['ATOM'], 
            gz=False, 
            append_newline=True)
        
        mutant_ppdb.to_pdb(path='TMP/{0}.pdb'.format(self.mutant_pdb), 
            records=['ATOM'], 
            gz=False, 
            append_newline=True)
        
        # os.system('./TMalign TMP/{0}.pdb TMP/{1}.pdb -o TMP/{0}_{1}.sup'.format(self.wild_pdb,self.mutant_pdb))
        # os.system('mv TMP/{0}_{1}.sup_all_atm TMP/{0}_{1}.sup_all_atm.pdb'.format(self.wild_pdb,self.mutant_pdb))
        subprocess.run('./TMalign TMP/{0}.pdb TMP/{1}.pdb -o TMP/{0}_{1}.sup'.format(self.wild_pdb,self.mutant_pdb),shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run('mv TMP/{0}_{1}.sup_all_atm TMP/{0}_{1}.sup_all_atm.pdb'.format(self.wild_pdb,self.mutant_pdb),shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ###Chain A because after alignment, the first PDB gets automatically assigned chain A, and the second one B. So here wild pdb is chain A and mutant is chain B

        
        #For wild_df
        pdb = PandasPdb().read_pdb('TMP/{0}_{1}.sup_all_atm.pdb'.format(self.wild_pdb,self.mutant_pdb))
        wild_df = pdb.df['ATOM'][(pdb.df['ATOM']['chain_id'] == 'A') & (pdb.df['ATOM']['atom_name'].isin(ATOMS)) & (pdb.df['ATOM']['alt_loc'].isin(['A','']))]
        
        #for mutant_df

        mutant_df = pdb.df['ATOM'][(pdb.df['ATOM']['chain_id'] == 'B') & (pdb.df['ATOM']['atom_name'].isin(ATOMS)) & (pdb.df['ATOM']['alt_loc'].isin(['A','']))]
        subprocess.run('cd TMP; rm *',shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wild_df,mutant_df
    def get_xyz(self):
        # wild_df = self.get_pdb_df(pdb_file=self.wild_pdb,chain_id=self.wild_chain)
        # mutant_df = self.get_pdb_df(pdb_file=self.mutant_pdb,chain_id=self.mutant_chain)

        wild_ppdb = self.get_ppdb(pdb_file=self.wild_pdb,chain_id=self.wild_chain)
        mutant_ppdb = self.get_ppdb(pdb_file=self.mutant_pdb,chain_id=self.mutant_chain)
        if self.state == 'train':
            wild_df, mutant_df = self.do_TMalign(wild_ppdb=wild_ppdb,mutant_ppdb=mutant_ppdb)
        else:
            wild_df = self.get_pdb_df(pdb_file=self.wild_pdb,chain_id=self.wild_chain)
            mutant_df = self.get_pdb_df(pdb_file=self.mutant_pdb,chain_id=self.mutant_chain)
        

        ###Let's do TMAlign processing
        wild_xyz = wild_df[['residue_number','residue_name','atom_name','x_coord','y_coord','z_coord']]

        wild_xyz_centered_coords = wild_xyz[['x_coord','y_coord','z_coord']].apply(lambda x: x-x.mean())
        wild_xyz_centered = pd.concat([wild_xyz[['residue_number','residue_name','atom_name']],wild_xyz_centered_coords],axis=1)

        # print(wild_xyz_centered.head())

        wild_xyz_for_input = self.prepare_wild_xyz_for_input(wild_xyz=wild_xyz_centered)
        # wild_xyz_for_input = wild_xyz[['x_coord','y_coord','z_coord']].to_numpy()

        mutant_df = self.get_pdb_df(pdb_file=self.mutant_pdb,chain_id=self.mutant_chain)
        mutant_xyz = mutant_df[['residue_number','residue_name','atom_name','x_coord','y_coord','z_coord']]
        mutant_xyz_centered_coords = mutant_xyz[['x_coord','y_coord','z_coord']].apply(lambda x: x-x.mean())
        mutant_xyz_centered = pd.concat([mutant_xyz[['residue_number','residue_name','atom_name']],mutant_xyz_centered_coords],axis=1)


        mutant_xyz_for_ground_truth = self.prepare_mutant_xyz_for_ground_truth(mutant_xyz=mutant_xyz_centered)
        # mutant_xyz_for_ground_truth = mutant_xyz[['x_coord','y_coord','z_coord']].to_numpy()

        return wild_xyz_for_input,mutant_xyz_for_ground_truth
    
    def get_atoms(self):
        wild_df = self.get_pdb_df(pdb_file=self.wild_pdb,chain_id=self.wild_chain)
        wild_atom = wild_df[['residue_number','residue_name','atom_name']]
        length_wild = int(wild_atom['residue_number'].iloc[-1]) - int(wild_atom['residue_number'].iloc[0]) + 1
        flag_length = int(wild_atom['residue_number'].iloc[0])-1
        wild_atom_for_input = []
        enc = OneHotEncoder(handle_unknown='ignore',categories=ATOMS)
        one_hot_wild_atoms = []
        one_hot_residues = []
        for idx in range(1+flag_length,length_wild+flag_length+1):
            if idx != self.pos+flag_length+1:
                atom_for_that_residue = wild_atom.loc[wild_atom['residue_number']==idx][['atom_name']]
                
                residue_name = wild_atom.loc[wild_atom['residue_number']==idx]['residue_name'].iloc[0]
                atoms_for_this_particular_residue = AMINO_ACID_ATOM[residue_name]
                atoms_available_in_pdb_for_this_particular_residue = wild_atom.loc[wild_atom['residue_number']==idx]['atom_name'].to_list()
                if idx == length_wild + flag_length:
                    atoms_for_this_particular_residue = atoms_for_this_particular_residue + ['OXT']
                for atom in atoms_for_this_particular_residue:
                    # print(atom)
                    one_hot_for_this_atom = [0] * len(ATOMS)
                    one_hot_for_this_atom_residue = [0] * (len(BASE_AMINO_ACIDS_3_LETTER)+ 1)
                    if residue_name in BASE_AMINO_ACIDS_3_LETTER:
                        one_hot_for_this_atom_residue[BASE_AMINO_ACIDS_3_LETTER.index(residue_name)] += 1
                    else:
                        one_hot_for_this_atom_residue[-1] += 1

                    one_hot_for_this_atom[ATOMS.index(atom)] += 1
                    one_hot_wild_atoms.append(one_hot_for_this_atom)
                    one_hot_residues.append(one_hot_for_this_atom_residue)
                # print(atoms_for_this_particular_residue)
                # print(atoms_available_in_pdb_for_this_particular_residue)
            else:
                atom_for_that_residue = wild_atom.loc[wild_atom['residue_number']==idx][['atom_name']]
                wild_residue_name = wild_atom.loc[wild_atom['residue_number']==idx]['residue_name'].iloc[0]

                mutated_residue_name = AMINO_ACIDS_1_TO_3[self.mutated_residue]
                atoms_for_wild_residue = atom_for_that_residue['atom_name'].to_list()
                # print(atoms_for_wild_residue)
                atoms_for_mutated_residue = AMINO_ACID_ATOM[AMINO_ACIDS_1_TO_3[self.mutated_residue]]
                # print(atoms_for_wild_residue)
                # print(atoms_for_mutated_residue)
                if idx == length_wild + flag_length:
                    atoms_for_mutated_residue + ['OXT']
                for atom in atoms_for_mutated_residue:
                    one_hot_for_this_atom = [0] * len(ATOMS)
                    one_hot_for_this_atom[ATOMS.index(atom)] += 1
                    one_hot_wild_atoms.append(one_hot_for_this_atom)

                    one_hot_for_this_atom_residue = [0] * (len(BASE_AMINO_ACIDS_3_LETTER)+ 1)
                    if mutated_residue_name in BASE_AMINO_ACIDS_3_LETTER:
                        one_hot_for_this_atom_residue[BASE_AMINO_ACIDS_3_LETTER.index(mutated_residue_name)] += 1
                        if wild_residue_name in BASE_AMINO_ACIDS_3_LETTER:
                            one_hot_for_this_atom_residue[BASE_AMINO_ACIDS_3_LETTER.index(wild_residue_name)] = -1
                        else:
                            one_hot_for_this_atom_residue[-1] = -1
                    else:
                        one_hot_for_this_atom_residue[-1] += 1
                        one_hot_for_this_atom_residue[BASE_AMINO_ACIDS_3_LETTER.index(wild_residue_name)] = -1
                    one_hot_residues.append(one_hot_for_this_atom_residue)

                




        mutated_atoms_index_list = []
        for idx,item in enumerate(one_hot_residues):
            if -1 in item:
                # print(idx)
                mutated_atoms_index_list.append(idx)

        return torch.from_numpy(np.asarray(one_hot_wild_atoms)), torch.from_numpy(np.asarray(one_hot_residues)), mutated_atoms_index_list
    
    def get_sequence(self,pdb_file,chain_id):
        pdb = PandasPdb().read_pdb(path=pdb_file)
        seq3_amino = pdb.df['ATOM'][
            (pdb.df['ATOM']['chain_id'] == chain_id) & (pdb.df['ATOM']['atom_name'] == 'CA') & (
                pdb.df['ATOM']['alt_loc'].isin(['A', '']))]['residue_name'].to_numpy()
        # print(len(seq3_amino))
        sequence = ''
        for amino_acid in seq3_amino:
            sequence += RESI_THREE_TO_1[amino_acid]
        return sequence
    def get_nearest_neighbors(self):
        # Read the PDB file using the pandaspdb library
        # pdb = PandasPdb().read_pdb(self.pdb_file)
        # Select the columns containing the XYZ coordinates
        X,y = self.get_xyz()
        # Initialize the KNN classifier
        knn = NearestNeighbors(n_neighbors=30)
        # Fit the classifier to the data
        knn.fit(X)
        # Get the 20 nearest neighbors for each atom
        neighbors = knn.kneighbors(X)
        # print(np.shape(neighbors))
        neighbors_distance = torch.from_numpy(neighbors[0])
        neighbors_index = torch.from_numpy(neighbors[1])
        # print(neighbors_distance)
        # print(neighbors_index)
        # return neighbors_distance, neighbors_index
        return self.compute_edge_attr_from_neighbours_distance(neighbors_distance),self.compute_edge_indexes_from_nearest_neighbours(neighbors_index)

    def compute_edge_indexes_from_nearest_neighbours(self,neighbours_index):
        edge_index = []
        neighbours_index = neighbours_index.numpy()
        for i, neighbours in enumerate(neighbours_index):
            # print(i)
            for neighbour in neighbours:
                edge_index.append([i, neighbour])
        edge_index = torch.from_numpy(np.array(edge_index))

        return edge_index.t().contiguous()

    def compute_edge_attr_from_neighbours_distance(self,neighbours_distance):
        edge_attr = []
        for source_node, neighbours in enumerate(neighbours_distance):
            for neighbour_distance in neighbours:
                # print(source_node,neighbour_distance.item())
                edge_attr.append(neighbour_distance.item())
        edge_attr = np.expand_dims(edge_attr, axis=1)
        return torch.from_numpy(edge_attr)
    # def centralize(self,coords):
    #     node_sum = 0
    #     count = 0
    #     for atom in coords:
    #         for coord in atom:
                
    #             node_sum += coord
    #             count += 1
    #     node_mean = node_sum / count
    #     length = coords.shape[0]
    #     coords_centralized = np.zeros(shape=(length,3))
    #     return node_mean
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
    def make_esm_features_atom_level(self):
        seq = self.get_sequence(pdb_file=os.path.join(self.pdb_dir,self.mutant_pdb+'.pdb'),chain_id=self.mutant_chain)
        with open(self.esm_feature_dict,'rb') as f:
            esm_dict = pickle.load(f)
        residue_level_features = esm_dict[seq]
        # print(residue_level_features.size())
        sum = 0
        # print(len(seq))
        atom_level_feature_list = []
        for i,residue in enumerate(seq):
            residue_3 = AMINO_ACIDS_1_TO_3[residue]
            
            number_of_atoms = len(AMINO_ACID_ATOM[residue_3])
            ones = torch.ones(size=(1,number_of_atoms,1280))
            # print(ones.size())
            # print(residue_level_features[0][i].size())
            atom_level_feature = ones * residue_level_features[0][i]
            # print(atom_level_feature.size())
            atom_level_feature_list.append(atom_level_feature)
        oxt_features = torch.unsqueeze(residue_level_features[:,-1,:],dim=1)

        atom_level_feature_list.append(oxt_features)
        atom_level_features = torch.cat(atom_level_feature_list,dim=1)
            
        # print(atom_level_features.size())
        return atom_level_features

    def pdb_to_graph(self):
        node_coords, label_coords = self.get_xyz()
        # if MIN_MAX_SCALE:
        #     node_coords = self.min_max_scale(node_coords)
        #     label_coords = self.min_max_scale(label_coords)

        node_one_hot_sequence_atoms, node_one_hot_sequence_residues, mutated_atoms_index_list = self.get_atoms()
        # self.output_to_pdb(coords=node_coords,node_one_hot_sequence_atoms=node_one_hot_sequence_atoms,node_one_hot_sequence_residues=node_one_hot_sequence_residues,name=self.name + '_input.pdb')
        # self.output_to_pdb(coords=label_coords,node_one_hot_sequence_atoms=node_one_hot_sequence_atoms,node_one_hot_sequence_residues=node_one_hot_sequence_residues,name='temp_mutant.pdb')

        edge_distances , edge_index = self.get_nearest_neighbors()
        
        # for idx,atom in enumerate(zip(node_coords,label_coords)):
        #     if idx in mutated_atoms_index_list:
        #         print(idx,atom)

            
                

        

        node_coords = torch.from_numpy(node_coords)
        label_coords = torch.from_numpy(label_coords)
        
        graph = Data(node_coords = node_coords.float(), node_one_hot_sequence = node_one_hot_sequence_atoms.float(),node_one_hot_sequence_residues=node_one_hot_sequence_residues.float(),edge_index=edge_index,edge_distances=edge_distances.float())
        return graph, label_coords.float()
    
   
# pdbreader = PDBReader_All_Atom(pdb_dir=PDB_REINDEXED_DIR,mutant_pdb='7MGR_A',wild_pdb='8b0s_A',mutation_info='C_144_A')

# x,y = pdbreader.pdb_to_graph()
# print(x['node_one_hot_sequence'][1118])
# print(x['node_one_hot_sequence_residues'][1118])
# print(x['node_one_hot_sequence_residues'].size())
# node_one_hot_sequence_residues = x['node_one_hot_sequence_residues']
# print(pdbreader.make_esm_features_atom_level())



        

# print(y.size())
# subprocess.run('./TMscore temp.pdb temp_mutant.pdb -outfmt 2',shell=True)
# # x,y = pdbreader.get_xyz()
# x = pdbreader.get_ppdb(pdb_file=pdbreader.wild_pdb,chain_id=pdbreader.wild_chain)
# y = pdbreader.get_ppdb(pdb_file=pdbreader.mutant_pdb,chain_id=pdbreader.mutant_chain)

# x, y = pdbreader.do_TMalign(x,y)
# print(x['residue_number'].head())
# print(y['residue_number'].head())
# exit()
# print(x.shape,y.shape)
# x = pdbreader.get_sequence(pdb_file=os.path.join(pdbreader.pdb_dir,pdbreader.wild_pdb+'.pdb'),chain_id=pdbreader.wild_chain)
# y = pdbreader.get_sequence(pdb_file=os.path.join(pdbreader.pdb_dir,pdbreader.mutant_pdb+'.pdb'),chain_id=pdbreader.mutant_chain)
# print(len(x),len(y))

# for x_char, y_char in zip(x,y):
#     if x_char != y_char:
#         print(AMINO_ACIDS_1_TO_3[x_char],AMINO_ACIDS_1_TO_3[y_char])
# exit()
# df = pd.read_csv('MutData2022.csv')
# count = 0
# problematic_pairs = []

# for i in tqdm(range(len(df))):
#     mutant_pdb = df['Mutated_PDB'].iloc[i]
#     mutation_info = df['Mutation INFO'].iloc[i]
#     wild_pdb = df['Possible Wilds'].iloc[i]


#     pdbreader = PDBReader_All_Atom(pdb_dir=PDB_REINDEXED_DIR,mutant_pdb=mutant_pdb,wild_pdb=wild_pdb,mutation_info=mutation_info)
#     pos = pdbreader.pos
#     wild_residue = pdbreader.initial_residue
#     mutated_residue = pdbreader.mutated_residue

#     wild_xyz , mutant_xyz = pdbreader.get_xyz()

#     wild_atom_len = wild_xyz.shape[0]
#     mutant_atom_len = mutant_xyz.shape[0]
#     if wild_atom_len != mutant_atom_len:
#         count += 1

#         print(count,wild_pdb,mutation_info,mutant_pdb)
#         print(wild_atom_len,mutant_atom_len)

#         problematic_pairs.append([mutant_pdb,mutation_info,wild_pdb])

# with open('problematic_pairs.pkl', 'wb') as f:
#     pickle.dump(problematic_pairs, f)
        
    

# print(count)

    

# length_wild = int(wild_xyz['residue_number'].iloc[-1])
# # print(mutant_xyz.loc[mutant_xyz['residue_number']==pos+1])
# wild_xyz_for_input = []
# for idx in range(1,length_wild+1):
#     if idx != pos+1:
#         xyz_for_that_residue = wild_xyz.loc[wild_xyz['residue_number']==idx][['x_coord','y_coord','z_coord']]
#         for j in range(len(xyz_for_that_residue)):
#             # print(xyz_for_that_residue.iloc[j].to_list())
#             wild_xyz_for_input.append(xyz_for_that_residue.iloc[j].to_list())
#     else:
#         xyz_for_that_residue = wild_xyz.loc[wild_xyz['residue_number']==idx][['residue_name','atom_name','x_coord','y_coord','z_coord']]
#         print(xyz_for_that_residue)
#         print(AMINO_ACIDS_1_TO_3[wild_residue])
#         print(AMINO_ACIDS_1_TO_3[mutated_residue])
#         atoms_for_wild_residue = xyz_for_that_residue['atom_name'].to_list()
#         print(atoms_for_wild_residue)
#         atoms_for_mutated_residue = AMINO_ACID_ATOM[AMINO_ACIDS_1_TO_3[mutated_residue]]
#         for atom in atoms_for_mutated_residue:
#             if atom in atoms_for_wild_residue:
#                 coords = xyz_for_that_residue.loc[xyz_for_that_residue['atom_name']==atom][['x_coord','y_coord','z_coord']]
#                 for j in range(len(coords)):
#                     print(coords.iloc[j].to_list())
#                     wild_xyz_for_input.append(coords.iloc[j].to_list())
#             else:
#                 wild_xyz_for_input.append([0,0,0])
            
        






