import argparse
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from model import Model
from data import DataLoader, Dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser('main')
    parser.add_argument('--data', type=str, default='data.txt')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=.99)
    parser.add_argument('--clip_norm', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if args.cuda else 'cpu'

    data = np.loadtxt(args.data, dtype=int)
    num_emb = len(set(data[:,0]))
    num_class = len(set(data[:,1]))
    model = Model(num_emb, args.emb_dim, args.hidden_dim, num_class).to(device)
    optimizer = optim.SGD(model.parameters(), args.lr, args.momentum)
    loss_fn = CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10, verbose=True)

    dataset = Dataset(data)
    dataloader = DataLoader(dataset, args.batch_size, True, args.num_workers)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x,y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)
        print('epcoh:%d\tloss:%f' % (epoch, total_loss))


if __name__ == '__main__':
    main()