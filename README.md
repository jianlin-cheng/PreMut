# PreMut
<p align="center">
  <img src="model.png" width="750" title="hover text">
</p>
Accurate prediction of the structure of any protein mutant with a single-site mutation with equivariant graph neural networks. PreMut takes as input a wild-type protein structure and a single-site mutation to predict the structure of the mutated protein with the mutation. 

## Installation
* Download or clone this repository

* Setup environment by running the following command
```
conda env create -f environment.yml
```

* Some packages may fail to install. In that case, use the following commands to install these packages necessary for full environment installation.

* Firstly, activate the created environment
```
conda activate PreMut
```
* pip install the following packages
```
pip install lightning
```
```
pip install biopandas
```
```
pip install rmsd
```


## Make prediction

* To make a prediction, save the wild pdb file in a folder and create a folder to save the prediction.
* Run the following command
```
python src/prediction.py wild_pdb_path mutation_info chain_id output_dir
```
* Here wild_pdb_path is the path to the structure file of the wild protein in the pdb format (e.g. /path/to/8b0s.pdb), mutation_info is the information regarding the wild residue, mutated position (0 indexed) and the mutated residue (e.g. C_144_A), chain_id is the id of the chain in the wild pdb file (e.g. A), and output_dir is the directory to save the predicted structure for the mutant as a pdb file.
* Example
```
python src/prediction.py examples/8b0s.pdb C_144_A A predictions
```

## Refine predictions
* ATOMRefine can be utilized to further refine the predictions made by PreMut. Install ATOMRefine from this link (https://github.com/BioinfoMachineLearning/ATOMRefine) and follow the instructions provided in that repository to apply refinement to the predictions made by PreMut.

## Training
* Download and install TM-score and TM-align from this link: [TM-score](https://zhanggroup.org/TM-score/), [TM-align](https://zhanggroup.org/TM-align/)
* Download all the files in the MutData2022 and MutData2023 datasets from the link [PreMut](https://zenodo.org/record/8401256). Uncompress the files, and then move the uncompressed folders to the installation directory of the PreMut repository.
* Run the following command to train the model
```
python src/train.py

```
* After the completion of training, the model weights are stored in a folder titled Checkpoints.
* You can select the model with the best validation performance from the Checkpoints folder to test.
  
## Evaluation
* Evaluation script is provided to check the reported performance in the paper.
* Make sure the files are downloaded, uncompressed and moved to the root of the directory as instructed in the previous section.
* Install SPECS from following the instructions from this link [SPECS](http://watson.cse.eng.auburn.edu/SPECS/).
* To get the evaluation metrics on MutData2022_test dataset, run this command
```
python Evaluation.py MutData2022
```
* To get the evaluation metrics on MutData2023_test dataset, run this command
```
python Evaluation.py MutData2023
```

## Acknowlegements
The EGNN model code is partially adapted and built upon the source code from the following project [egnn](https://github.com/vgsatorras/egnn). We thank all the contributors and maintainers.
## Reference

Sajid Mahmud, Alex Morehead,  Jianlin Cheng. Accurate prediction of protein tertiary structural changes induced by single-site mutations with equivariant graph neural networks. bioRxiv. (https://doi.org/10.1101/2023.10.03.560758)


