import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB


class LitAutoEncoder(L.LightningModule):
    def __init__(self, hidden_size, input_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_to_pred = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.input_to_hidden(combined)
        output = self.softmax(self.hidden_to_pred(combined))

        return output, hidden

    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        hidden = torch.zeros(1, self.hidden_size)

        # batch dimensions (batch size, timestep, features)
        for i in range(x.size()[1]):
            # grab all inputs for that for that batch
            x_t = x[:, i: :]
            output, hidden = self.forward(x_t, hidden)

        return self.criterion(output, target)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.005)

    def validation_step(self):
        pass


if __name__ == "__main__":
    train_iter = iter(IMDB(split="train"))
    print(next(train_iter))
