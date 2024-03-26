import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import pytorch_lightning as pl
from image import get_loader
from torch.nn.functional import softmax
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model import resnet
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
# # import os
# # from torch import optim, nn, utils, Tensor
# # from torchvision.datasets import MNIST
# # from torchvision.transforms import ToTensor
# # import lightning as L

# # class FlowerDataModule(pl.LightningDataModule):
# #     def __init__(self, train_loader, val_loader):
# #         super().__init__()
# #         self.train_loader = train_loader
# #         self.val_loader = val_loader

# #     def train_dataloader(self):
# #         return self.train_loader

# #     def val_dataloader(self):
# #         return self.val_loader

class FlowerModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr):
        super(FlowerModule, self).__init__()
        self.lr = lr
        if(model == "yolov5"):
            self.model = attempt_load("yolov5s")
            self.model.eval()
        # Define your CNN model # Replace with your CNN model
        # Define loss function and optimizer
        self.loss_fn = loss_fn
        
    def forward(self, data):
        return self.model(data)
    # this is called a 'hook' in pytorch lightning documentation

    def configure_optimizers(self):
        #TODO: Modify optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output, target)
        # pdb.set_trace()
        predicted = torch.argmax(output.data, dim = -1)
        # pdb.set_trace()
        accuracy = (predicted == target.data).float().mean()
        values = {"train_loss": loss, "train_acc": accuracy}
        self.log_dict(values, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output, target)
        # pdb.set_trace()
        predicted = torch.argmax(output.data, dim = -1)
        accuracy = (predicted == target.data).float().mean()
        values = {"val_loss": loss, "val_acc": accuracy}
        self.log_dict(values, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output, target)
        predicted = torch.argmax(output.data, dim = -1)
        accuracy = (predicted == target.data).float().mean()
        values = {"test_loss": loss, "test_acc": accuracy}
        self.log_dict(values, on_step=False, on_epoch=True)

    
tr_loader = get_loader(task = "train",batch_size = 32)
va_loader = get_loader(task = "validation",batch_size = 32)
# te_loader = get_loader(task = "test",batch_size=32)
#TODO: create a new model file within the model folder
#      import that model here and apply it to the training algorithm
cnn = "yolov5"
#TODO: modify loss function if you want
#      modify learning rate
model = FlowerModule(cnn, torch.nn.CrossEntropyLoss(), lr=0.01)
# Initantiate Trainer and start training
#TODO: modify max epochs as well as patience
trainer = pl.Trainer(log_every_n_steps=32,callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience = 10)])  # Change settings as needed
trainer.fit(model, tr_loader, va_loader)
#TODO: Uncomment the next 2 line to test the model on the test parition (edit which epoch)
te_loader = get_loader(task="test",batch_size=64)
trainer.test(model, te_loader)


### from pytorch documentation


# # define any number of nn.Modules (or use your current ones)
# encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
# decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# # define the LightningModule
# class LitAutoEncoder(L.LightningModule):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
#         # it is independent of forward
#         x, y = batch
#         x = x.view(x.size(0), -1)
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         loss = nn.functional.mse_loss(x_hat, x)
#         # Logging to TensorBoard (if installed) by default
#         self.log("train_loss", loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer


# # init the autoencoder
# autoencoder = LitAutoEncoder(encoder, decoder)

# # setup data
# dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
# train_loader = utils.data.DataLoader(dataset)


# # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
# trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
# trainer.fit(model=autoencoder, train_dataloaders=train_loader)


# # load checkpoint
# checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
# autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# # choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()

# # embed 4 fake images!
# fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
# embeddings = encoder(fake_image_batch)
# print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)


# # if we want to visualize experiments, can use tensorboard
# # tensorboard --logdir .

# # train on 4 GPUs
# trainer = Trainer(
#     devices=4,
#     accelerator="gpu",
#  )

# # train 1TB+ parameter models with Deepspeed/fsdp
# trainer = L.Trainer(
#     devices=4,
#     accelerator="gpu",
#     strategy="deepspeed_stage_2",
#     precision=16
#  )

# # 20+ helpful flags for rapid idea iteration
# trainer = L.Trainer(
#     max_epochs=10,
#     min_epochs=5,
#     overfit_batches=1
#  )

# # access the latest state of the art techniques
# trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])