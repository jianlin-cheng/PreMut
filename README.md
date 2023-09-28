# PreMut
Accurate prediction of single-site mutation induced changes on protein structures with equivariant graph neural networks. PreMut takes as input a wild-type protein structure and the mutation information to predict the structure of the mutated protein. 

## Installation
* Setup environment by running the following command
```
conda env create -f environment.yml
```

## Make prediction

* To make a prediction, save the wild pdb file in a folder and create a folder to save the prediction.
* Run the following command
```
python src/prediction.py WILDPDB MUTATIONINFO WILDPDBDIR PREDICTIONDIR
```
* Here WILDPDB is the pdb code with the chain id (e.g. 1ert_A), MUTATIONINFO is the information regarding the wild residue, mutated position (0 indexed) and the mutated residue (e.g. D_59_N), WILDPDBDIR is the directory where the wild pdb file is stored, and PREDICTIONDIR is the directory to save the predicted PDB file.
* Example
```
python src/prediction.py 1ert_A D_59_N MutData2022 predictions
```

## Training
* Download and install TM-score and TM-align from this link: (https://zhanggroup.org/TM-score/, https://zhanggroup.org/TM-align/)
* Download the pdb files in the MutData2022 and MutData2023 datasets from the link (https://zenodo.org/record/8339451). Uncompress the files, and then move the uncompressed folders MutData2022_PDB and MutData2023_PDB to the root direcctory of the PreMut repository.
* Run the following command to train the model
```
python src/train.py
```
* After the completion of training, the model weights are stored in a folder titled Checkpoints.
* You can select the model with the best validation performance from the Checkpoints folder to test.


