from dataclasses import dataclass, asdict

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange

from cipher_data import CipherTxtData, CipherNGramData

class NGramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, X):
        embeds = self.embeddings(X).view((1, -1))
        z = self.linear1(embeds)
        z = F.relu(z)
        z = self.linear2(z)
        log_probs = F.log_softmax(z, dim=1)

        return log_probs


@dataclass
class Hyperparams:
    context_size: int
    embedding_dim: int
    vocab_size: int


def main():
    data = CipherTxtData(mode="train", split=True)
    ngram_data = CipherNGramData(data.X, context_size=3)

    hparams = Hyperparams(context_size=3,
                          embedding_dim=10,
                          vocab_size=ngram_data.vocab_size)

    model = NGramLanguageModel(**asdict(hparams))
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train(model, ngram_data, criterion, optimizer)


def train(model, data, criterion, optimizer, n_epoch=3):
    model.train()
    losses = []
    for epoch in trange(n_epoch):
        print(f"Epoch: {epoch}.")
        for i, (X, y) in enumerate(tqdm(data)):
            log_probs = model(X)
            loss = criterion(log_probs, y)
            if i % 10000 == 0:
                print(f"\titeration: {i} \t loss: {loss.detach()}")
                losses.append(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #return losses

if __name__ == '__main__':
    main()
