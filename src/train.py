

from MyDataModule import MyDataModule
from LitModule import LitGat
import pytorch_lightning as pl
from constants import *
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
from model import egnn_ablation_atom_types_only
from torch_geometric.nn import GraphUNet


checkpoint_callback = ModelCheckpoint(save_top_k=10, monitor="validation/tmscore_epoch", mode="max", dirpath=CHECKPOINT_PATH, filename="PreMut-{epoch:02d}-{validation/tmscore_epoch:.4f}")
project_name = '{0}_MUTATION_STRUCTURE_PREDICTION'.format(MODEL_NAME.upper())
# wandb_logger = WandbLogger(project=project_name,name=MODEL_NAME)
class trainClass():
    def __init__(self,checkpoint_callback = checkpoint_callback
                 ,epochs=10
                 ):
        self.trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=epochs,
                             callbacks=[checkpoint_callback])
    def initialize_data_module(self):
        self.data = MyDataModule()
        self.data.setup('fit')
        return self.data
    def initialize_network(self, model_name =MODEL_NAME):
        
        
        model = egnn_ablation_atom_types_only(in_channels=[3,37,21],hidden_channels=32,num_classes=3,num_hidden_layers=4)
        
        return model
    def initialize_Lit_module(self,model_name):
        net = self.initialize_network(model_name=model_name)
        lit_module = LitGat(MODEL=net,loss_weight=LOSS_WEIGHT)
        return lit_module

    def fit(self,model_name = 'egnn'):

        datamodule = self.initialize_data_module()
        lit_module = self.initialize_Lit_module(model_name=model_name)
        # os.system('wandb login --anonymously')

        self.trainer.fit(model=lit_module,datamodule=datamodule)


training = trainClass(epochs=10)

training.fit(model_name='egnn')




