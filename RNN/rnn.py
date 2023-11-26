import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import IMDBDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy


class RNN(L.LightningModule):
    def __init__(self, hidden_size, input_size, output_size):
        super().__init__()

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()

        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        self.hidden_to_pred = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.input_to_hidden(combined)
        output = self.hidden_to_pred(hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        hidden = self.init_hidden(batch_size=x.size()[0]).to(x.device)

        # batch dimensions (batch size, timestep, features)
        for i in range(x.size()[1]):
            # grab all inputs for that for that batch
            x_t = x[:, i]

            # output & hidden variables are updated on every iteration
            # we only care about the last output
            output, hidden = self.forward(x_t, hidden)

        predictions = (output > 0.5).float().squeeze()

        self.train_accuracy(predictions, target)
        self.log('train_acc', self.train_accuracy, prog_bar=True, on_step=True)

        return self.criterion(output.squeeze(), target.float())

    def test_step(self, test_batch):
        x, target = test_batch
        hidden = self.init_hidden(batch_size=x.size()[0]).to(x.device)

        # batch dimensions (batch size, timestep, features)
        for i in range(x.size()[1]):
            # grab all inputs for that for that batch
            x_t = x[:, i]

            # output & hidden variables are updated on every iteration
            # we only care about the last output
            output, hidden = self.forward(x_t, hidden)

        predictions = (output > 0.5).float().squeeze()
        correct = (predictions == target).sum()

        # Log test loss and accuracy
        self.log('test_accuracy', correct.float() / target.size(0))

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)


if __name__ == "__main__":
    batch_size = 32

    dataset_train = IMDBDataset(split="train")
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    vocab_size = len(dataset_train.char_set)
    model = RNN(hidden_size=128, input_size=vocab_size, output_size=1)

    trainer = L.Trainer(accelerator='cpu', max_epochs=6)
    trainer.fit(model=model, train_dataloaders=train_loader)

    dataset_test = IMDBDataset(split="test")
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    trainer.test(ckpt_path='best', dataloaders=test_loader)
