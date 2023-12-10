from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
from utils import *
from models.CompGCN import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.TransE import *
from models.TransH import *
from Data_Loader import *
# Use Pytorch lighting module to write a trainer 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from easydict import EasyDict
import yaml
class Training_system(pl.LightningModule):
    '''
    The lightning module, need to over write following method: training_step, configure_optimizers
    '''
    def __init__(self, hparams, loss_f ,path):
        # The hyper parameter of the model
        super(Training_system, self).__init__()
        self.params = hparams
        
        # Include the following config:
        # max_steps
        self.max_epochs = self.params.Trainer.max_epochs
        
        # config file, include all parameters
        # dataset name
        self.dataset = self.params.Trainer.dataset
        self.data = Data(self.dataset, True, path)
        self.train_batchsize = self.params.Trainer.train_bs
        self.val_batchsize = self.params.Val.val_bs
        self.test_batchsize = self.params.Test.test_bs
        # training setting
        self.initlr = self.params.Trainer.initlr
        self.num_workers = self.params.Trainer.num_workers
        
        # loss function
        self.loss_f = loss_f
        self.model_params = self.params.Model
        self.model = Comp_dismult(self.data,**self.model_params.settings)
#         self.model = TransE(self.data.num_ents, self.data.num_rels, **self.model_params.settings)
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.valid_output = {}
        self.test_output = {}
        
    def train_dataloader(self):
        train_set = KG_dataset(self.data.train_set, self.data.num_ents, self.data.num_rels, self.data.train_label)
        train_loader = DataLoader(train_set, batch_size = self.train_batchsize, shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        valid_set = KG_dataset(self.data.valid_set, self.data.num_ents, self.data.num_rels, self.data.valid_label)
        valid_loader = DataLoader(valid_set, batch_size=self.val_batchsize, shuffle=False)
        return valid_loader
    
    def test_dataloader(self):
        test_set = KG_dataset(self.data.test_set,self.data.num_ents, self.data.num_rels, self.data.test_label)
        test_loader = DataLoader(test_set, batch_size=self.test_batchsize, shuffle=False)
        return test_loader
    
    def configure_optimizers(self):
        parameters = [{"params": self.model.parameters()}]
        print(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.Adam(parameters, lr = self.initlr)
        schedule = get_cosine_schedule_with_warmup(optimizer= optimizer, num_warmup_steps = 0, num_training_steps= len(self.train_dataloader()), num_cycles = 0.5)
        return [optimizer], [schedule]
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        pos, label = batch
        model_output = self.model(pos[0], pos[1])
        loss = self.loss_f(model_output, label)
#         model_output = self.model(pos, neg)
#         loss = self.loss_f(*model_output)
        self.manual_backward(loss)
        opt.step()
        sch = self.lr_schedulers()
        sch.step()
        
        self.log("train_loss", loss/len(self.data.train_set), prog_bar = True)
        return loss
    
    def on_validation_epoch_end(self):
        '''
        Validation epoch end will output a list, each of them are the validation step output;
        Here use the on validation epoch end to calculate the total result;
        '''
        
        self.valid_output["mrr"] = round(self.valid_output["mrr"]/self.data.valid_set.__len__(), 5)
        self.valid_output["hit_10"] = round(self.valid_output["hit_10"]/self.data.valid_set.__len__(), 5)
        self.valid_output["hit_3"] = round(self.valid_output["hit_3"]/self.data.valid_set.__len__(), 5)
        self.valid_output["hit_1"] = round(self.valid_output["hit_1"]/self.data.valid_set.__len__(), 5)
        # calculate the result
        self.log("Valid_mrr",self.valid_output["mrr"])
        self.log("Valid_hit10",self.valid_output["hit_10"])
        self.log("Valid_hit3",self.valid_output["hit_3"])
        self.log("Valid_hit1",self.valid_output["hit_1"])
        self.valid_output = {}
        
    def on_test_epoch_end(self):
        '''
        Validation epoch end will output a list, each of them are the validation step output;
        Here use the on validation epoch end to calculate the total result;
        '''
        self.test_output["mrr"] = round(self.test_output["mrr"]/self.data.test_set.__len__(), 5)
        self.test_output["hit_10"] = round(self.test_output["hit_10"]/self.data.test_set.__len__(), 5)
        self.test_output["hit_3"] = round(self.test_output["hit_3"]/self.data.test_set.__len__(), 5)
        self.test_output["hit_1"] = round(self.test_output["hit_1"]/self.data.test_set.__len__(), 5)
        # calculate the result
        self.log("test_mrr",self.test_output["mrr"])
        self.log("test_hit10",self.test_output["hit_10"])
        self.log("test_hit3",self.test_output["hit_3"])
        self.log("test_hit1",self.test_output["hit_1"])
        
    def validation_step(self, batch, batch_idx):
        pos, label = batch
        hit_10, hit_3, hit_1, mrr = self.model.predict(pos, label ,k = 10)
        self.valid_output["hit_10"] = hit_10 + self.valid_output.get("hit_10", 0.0)
        self.valid_output["hit_3"] = hit_3 + self.valid_output.get("hit_3", 0.0)
        self.valid_output["hit_1"] = hit_1 + self.valid_output.get("hit_1", 0.0)
        self.valid_output["mrr"] = mrr + self.valid_output.get("mrr", 0.0)
        return hit_10, hit_3, hit_1, mrr
    
    def test_step(self, batch,batch_idx):
        pos, label = batch
        hit_10, hit_3, hit_1, mrr = self.model.predict(pos, label ,k = 10)
        self.test_output["hit_10"] = hit_10 + self.test_output.get("hit_10", 0.0)
        self.test_output["hit_3"] = hit_3 + self.test_output.get("hit_3", 0.0)
        self.test_output["hit_1"] = hit_1 + self.test_output.get("hit_1", 0.0)
        self.test_output["mrr"] = mrr + self.test_output.get("mrr", 0.0)
        return hit_10, hit_3, hit_1, mrr
    
if __name__ == "__main__":
    path = "dataset"
    config_path = 'options/CompGCN_Hyparams.yaml'
    # config_path = r'C:\\Users\\Franklin\\code\\TKG reasoning\\options\\TransE_Hyparams.yaml'
    # config_path = r'C:\\Users\\Franklin\\code\\TKG reasoning\\options\\TransH_Hyparams.yaml'
    with open(config_path,"r") as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)
    output_dir = './logs'
    logger = TensorBoardLogger(name=config.log_name,save_dir = output_dir)
    #loss = Rank_loss(**config.loss_params) 
    loss = nn.BCEWithLogitsLoss()
    #loss = Lagrange_loss(**config.loss_params) 
    model = Training_system(config, loss, path)
    checkpoint_callback = ModelCheckpoint(
        monitor='Valid_hit10',
        filename='epoch{epoch:02d}-Valid_hit10-{Valid_hit10:.3f}-mrr-{Valid_mrr:.3f}',
        auto_insert_metric_name=False,   
        every_n_epochs=1,
        save_top_k=5,
        mode = "max",
        save_last=True
        )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
            check_val_every_n_epoch=config.Trainer.check_val_every_n_epoch,
            max_epochs=config.Trainer.max_epochs,
            accelerator=config.Trainer.accelerator,
            devices=config.Trainer.devices,
            precision=config.Trainer.precision,
            accumulate_grad_batches = config.Trainer.accumulate_grad_batches,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=config.Trainer.log_every_n_step,
            callbacks = [checkpoint_callback,lr_monitor_callback]
        )
    # for train
    trainer.fit(model=model, ckpt_path=None)
    # for validation
    # ckpt_path = None
    # trainer.validate(model=model, ckpt_path=ckpt_path)

