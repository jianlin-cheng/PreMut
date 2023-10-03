import os
from typing import Dict, List
import numpy as np
import subprocess
BASE_AMINO_ACIDS: List[str] = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


# BASE_AMINO_ACIDS_3_LETTER = ['ALA','ARG','ASN','ASP','ASX','CYS','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

BASE_AMINO_ACIDS_3_LETTER = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

# print(len(BASE_AMINO_ACIDS_3_LETTER))
BACKBONE_ATOMS = ['N','CA','C','O']
# AMINO_ACID_ATOM ={'ALA': BACKBONE_ATOMS + ['CB'],
# 'ARG': BACKBONE_ATOMS + ['CB','CG','CD','NE','CZ','NH1','NH2'], 
# 'ASN': BACKBONE_ATOMS+['CB','CG','OD1','ND2'],
# 'ASP': BACKBONE_ATOMS + ['CB','CG','OD1','OD2'],
# 'CYS': BACKBONE_ATOMS + ['CB','SG'],
# 'GLU': BACKBONE_ATOMS + ['CB','CG','CD','OE1','OE2'],
# 'GLN': BACKBONE_ATOMS + ['CB','CG','CD','OE1','NE1'],
# 'GLY': BACKBONE_ATOMS,
# 'HIS': BACKBONE_ATOMS + ['CB','CG','ND1','CD2','CE1','NE2'],
# 'ILE': BACKBONE_ATOMS + ['CB','CG1','CG2','CD1'],
# 'LEU': BACKBONE_ATOMS + ['CB','CG','CD1','CD2'],
# 'LYS': BACKBONE_ATOMS + ['CB','CG','CD','CE','NZ'],
# 'MET': BACKBONE_ATOMS + ['CB','CG','SD','CE'],
# 'PHE': BACKBONE_ATOMS + ['CB','CG','CD1','CD2','CE1','CE2','CZ'],
# 'PRO': BACKBONE_ATOMS + ['CB','CG','CD'],
# 'SER': BACKBONE_ATOMS + ['CB','OG'],
# 'THR': BACKBONE_ATOMS + ['CB','OG1','CG2'],
# 'TRP': BACKBONE_ATOMS + ['CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
# 'TYR': BACKBONE_ATOMS + ['CB','CG','CD1','CD2','CE1','CE2','CZ','OH'],
# 'VAL': BACKBONE_ATOMS + ['CB','CG1','CG2']}


AMINO_ACID_ATOM ={'ALA': BACKBONE_ATOMS + ['CB'],
'ARG': BACKBONE_ATOMS + ['CB','CG','CD','NE','CZ','NH1','NH2'], 
'ASN': BACKBONE_ATOMS+['CB','CG','OD1','ND2'],
'ASP': BACKBONE_ATOMS + ['CB','CG','OD1','OD2'],
'CYS': BACKBONE_ATOMS + ['CB','SG'],
'GLU': BACKBONE_ATOMS + ['CB','CG','CD','OE1','OE2'],
'GLN': BACKBONE_ATOMS + ['CB','CG','CD','OE1','NE2'],
'GLY': BACKBONE_ATOMS,
'HIS': BACKBONE_ATOMS + ['CB','CG','ND1','CD2','CE1','NE2'],
'ILE': BACKBONE_ATOMS + ['CB','CG1','CG2','CD1'],
'LEU': BACKBONE_ATOMS + ['CB','CG','CD1','CD2'],
'LYS': BACKBONE_ATOMS + ['CB','CG','CD','CE','NZ'],
'MET': BACKBONE_ATOMS + ['CB','CG','SD','CE'],
'PHE': BACKBONE_ATOMS + ['CB','CG','CD1','CD2','CE1','CE2','CZ'],
'PRO': BACKBONE_ATOMS + ['CB','CG','CD'],
'SER': BACKBONE_ATOMS + ['CB','OG'],
'THR': BACKBONE_ATOMS + ['CB','OG1','CG2'],
'TRP': BACKBONE_ATOMS + ['CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
'TYR': BACKBONE_ATOMS + ['CB','CG','CD1','CD2','CE1','CE2','CZ','OH'],
'VAL': BACKBONE_ATOMS + ['CB','CG1','CG2']}




# AMINO_ACID_ATOM ={'ALA': BACKBONE_ATOMS,
# 'ARG': BACKBONE_ATOMS , 
# 'ASN': BACKBONE_ATOMS,
# 'ASP': BACKBONE_ATOMS,
# 'CYS': BACKBONE_ATOMS,
# 'GLU': BACKBONE_ATOMS ,
# 'GLN': BACKBONE_ATOMS ,
# 'GLY': BACKBONE_ATOMS,
# 'HIS': BACKBONE_ATOMS ,
# 'ILE': BACKBONE_ATOMS,
# 'LEU': BACKBONE_ATOMS,
# 'LYS': BACKBONE_ATOMS ,
# 'MET': BACKBONE_ATOMS ,
# 'PHE': BACKBONE_ATOMS ,
# 'PRO': list(set(BACKBONE_ATOMS)),
# 'SER': BACKBONE_ATOMS ,
# 'THR': BACKBONE_ATOMS ,
# 'TRP': BACKBONE_ATOMS ,
# 'TYR': BACKBONE_ATOMS,
# 'VAL': BACKBONE_ATOMS }
AMINO_ACIDS_3_TO_1= {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

AMINO_ACIDS_1_TO_3 = {v: k for k, v in AMINO_ACIDS_3_TO_1.items()}

ATOMS = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG']

# ATOMS = []

# ATOMS = ATOMS + BACKBONE_ATOMS
# for key,value in AMINO_ACID_ATOM.items():
#     ATOMS += value
# ATOMS.append('OXT')
# ATOMS = list(set(ATOMS))

# ATOMS = sorted(ATOMS)
# print(ATOMS)






RESI_THREE_TO_1: Dict[str, str] = {
    "ALA": "A",
    "ASX": "B",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "GLX": "Z",
    "CSD": "C",
    "HYP": "P",
    "BMT": "T",
    "3HP": "X",
    "4HP": "X",
    "5HP": "Q",
    "ACE": "X",
    "ABA": "A",
    "AIB": "A",
    "NH2": "X",
    "CBX": "X",
    "CSW": "C",
    "OCS": "C",
    "DAL": "A",
    "DAR": "R",
    "DSG": "N",
    "DSP": "D",
    "DCY": "C",
    "CRO": "TYG",
    "DGL": "E",
    "DGN": "Q",
    "DHI": "H",
    "DIL": "I",
    "DIV": "V",
    "DLE": "L",
    "DLY": "K",
    "DPN": "F",
    "DPR": "P",
    "DSN": "S",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DVA": "V",
    "FOR": "X",
    "CGU": "E",
    "IVA": "X",
    "KCX": "K",
    "LLP": "K",
    "CXM": "M",
    "FME": "M",
    "MLE": "L",
    "MVA": "V",
    "NLE": "L",
    "PTR": "Y",
    "ORN": "A",
    "SEP": "S",
    "SEC": "U",
    "TPO": "T",
    "PCA": "Q",
    "PVL": "X",
    "PYL": "O",
    "SAR": "G",
    "CEA": "C",
    "CSO": "C",
    "CSS": "C",
    "CSX": "C",
    "CME": "C",
    "TYS": "Y",
    "BOC": "X",
    "TPQ": "Y",
    "STY": "Y",
    "UNK": "X",
}


amino_acid_atom ={'ALA'}
# print(os.system('cd ..; cd .. ;pwd'))
PATH_PREFIX = subprocess.check_output("cd ..;pwd", shell=True).decode("utf-8")[0:-1]


# PATH_PREFIX = os.system('cd ..; cd .. ;pwd')
PROJECT_NAME = 'PreMut'
PROJECT_DIR = os.path.join(PATH_PREFIX,PROJECT_NAME)
PDB_DIR = os.path.join(PROJECT_DIR,'PDB')
# PDB_REINDEXED_DIR = os.path.join(PROJECT_DIR,'MutData2023_PDB')
PDB_REINDEXED_DIR = os.path.join(PROJECT_DIR,'MutData2023_PDB')

# JSON_PATH = os.path.join(PROJECT_DIR,'thermomutdb.json')
# DATA_DIR = os.path.join(PROJECT_DIR,'PDB')
CHECKPOINT_PATH = os.path.join(PROJECT_DIR,'Checkpoints_V3')
NUM_WORKERS = 8
GRAPH_MODEL_NAME = 'GATCONV'
MAX = -1
# for files in os.listdir(CHECKPOINT_PATH):
#     if GRAPH_MODEL_NAME in files:
#         epoch_num = int(files.split('=')[1].split('-')[0])
#         if epoch_num > MAX:
#             MAX = epoch_num
#             MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,files)

# MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,'proper_GraphUNet_mutation_structure_prediction_non_weighted_loss-epoch=17-validation/tmscore_epoch=0.98.ckpt')
# MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,'proper_EGNN_ablation_atom_type_only_mutation_structure_prediction_1_weighted_loss-epoch=55-validation/tmscore_epoch=0.9893.ckpt')
# MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,'proper_EGNN_ablation_residue_type_only_mutation_structure_prediction_1_weighted_loss-epoch=66-validation/tmscore_epoch=0.9828.ckpt')
# MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,'proper_EGNN_mutation_structure_prediction_1_weighted_loss-epoch=68-validation/tmscore_epoch=0.9641.ckpt')
# MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,'proper_EGNN_ablation_atom_types_only_mutation_structure_prediction_1_weighted_loss-epoch=48-validation/tmscore_epoch=0.9694.ckpt')
# MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,'proper_EGNN_ablation_residue_types_only_mutation_structure_prediction_1_weighted_loss-epoch=56-validation/tmscore_epoch=0.9882.ckpt')
# MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,'proper_EGNN_mutation_structure_prediction_1_weighted_loss-epoch=14-validation/tmscore_epoch=0.9915.ckpt')

MODEL_CHECKPOINT_PATH = 'Saved_Model/model.ckpt'
EFFECT_PREDICTION = False
MODEL_NAME = 'PreMut'
MIN_MAX_SCALE = True
NUM_HIDDEN_LAYERS = 5
LOSS_WEIGHT = 1
MAX = 600 #562.44
MIN = -300 #-296.669

MASK = False

