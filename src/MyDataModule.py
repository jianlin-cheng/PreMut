import pytorch_lightning as pl
from typing import Optional
from dataset import *
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from constants import *

class MyDataModule(pl.LightningDataModule):
    

    def __init__(self, ):
        super(MyDataModule).__init__()

    def prepare_data(self):
        
        
        pass

    def prepare_data_per_node(self):
        pass

    def _log_hyperparams(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here

        if stage == "fit" or stage is None:
            
            train_set_full = dataset
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            # self.test = YourCustomDataset(
            #     root_path="/Users/yourusername/path/to/data/test_set/",
            #     ipt="input/",
            #     tgt="target/",
            #     tgt_scale=25,
            #     train_transform=False)
            self.test = test_dataset

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        # return DataLoader(self.train, batch_size=32, num_workers=8)

        return DataLoader(dataset=self.train,batch_size=1,num_workers=NUM_WORKERS)

    def val_dataloader(self):
        # return DataLoader(self.validate, batch_size=32, num_workers=8)
        return DataLoader(dataset=self.validate,batch_size=1,num_workers=NUM_WORKERS)


    def test_dataloader(self):
        # return DataLoader(self.test, batch_size=32, num_workers=8)
        return DataLoader(dataset=self.test,batch_size=1,num_workers=NUM_WORKERS)