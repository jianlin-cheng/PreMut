# PreMut
Accurate prediction of single-site mutation induced changes on protein structures with equivariant graph neural networks

## Prerequisites
* setup environment by running the following command
```
conda env create -f environment.yml
```
* Download and install TMScore and TMAlign from this link: (https://zhanggroup.org/TM-score/, https://zhanggroup.org/TM-align/)
* Download the datasets MutData2022 and MutData2023 from the following link (https://zenodo.org/record/8339451). Uncompress the files, then save the folders MutData2022_PDB and MutData2023_PDB in the root of the repository.
## Run prediction

* To do a prediction, save the wild pdb file in a folder, also create a folder to save the prediction.
* run the following command
```
python src/prediction.py WILDPDB MUTATIONINFO WILDPDBDIR PREDICTIONDIR
```
* Here WILDPDB is the pdb code with the chain id (e.g 1ert_A), MUTATIONINFO is the information regarding the wild residue, mutated position (0 indexed) and the mutated residue (e.g D_59_N), WILDPDBDIR is the directory where the wild pdb is stored, and PREDICTIONDIR is the directory to save the predicted PDB files.

## Training
* run the following command to train the model.
```
python src/train.py
```


