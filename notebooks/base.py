from torch import nn, optim
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryFBetaScore, BinaryPrecision, BinaryRecall

class BaseModel(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.loss = nn.BCEWithLogitsLoss()
        self.f_beta = BinaryFBetaScore(beta=0.5)
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.f_beta(y_hat, y)
        self.accuracy(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.log_dict({"val_f_beta": self.f_beta, "val_accuracy": self.accuracy, "val_precision": self.precision, "val_recall": self.recall}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.f_beta(y_hat, y)
        self.accuracy(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.log_dict({"val_f_beta": self.f_beta, "val_accuracy": self.accuracy, "val_precision": self.precision, "val_recall": self.recall}, prog_bar=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        # return optimizer
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=50, steps_per_epoch=70)
        # return [optimizer], [scheduler]