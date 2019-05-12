import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from model import Model
from data import DataLoader, Dataset
from tqdm import tqdm


def negative_sampling_loss(logit:torch.Tensor, target:torch.Tensor, neg:torch.Tensor):
    '''
    given logit, and its target indices and negative indices
    calcuate the loss as negative of logit(neg) - logit(target)
    '''
    # logit in [batch_size, num_class]
    # target in [batch_size]
    # neg in [batch_size, num_samples]
    pos = torch.FloatTensor([l[t] for l,t in zip(logit, target)], device=logit.device).mean()
    neg = torch.cat([l[n] for l,n in zip(logit, neg)]).mean()
    return (neg - pos) / len(logit)


def sample_negatives(pos:torch.Tensor, high:int, num_samples:int):
    '''
    given positive, sample num_samples negatives at random from 0 to high
    '''
    samples = np.arange(high, dtype=np.int64)
    neg = []
    for p in pos:
        n = np.random.choice(samples, num_samples + 1, replace=False)
        idx = np.nonzero(p.item() != n)[0][:num_samples]
        neg.append(n[idx])
    neg = torch.LongTensor(neg)
    return neg

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
    parser.add_argument('--save', type=str, default='model.pt')
    parser.add_argument('--loss', type=str, default='softmax')
    parser.add_argument('--num_neg_samples', type=int, default=10)
    args = parser.parse_args()
    device = 'cuda' if args.cuda else 'cpu'

    data = np.loadtxt(args.data, dtype=int)
    num_emb = len(set(data[:,0]))
    num_class = len(set(data[:,1]))
    model = Model(num_emb, args.emb_dim, args.hidden_dim, num_class).to(device)
    optimizer = optim.SGD(model.parameters(), args.lr, args.momentum)
    if args.loss == 'softmax':
        loss_fn = CrossEntropyLoss()
    elif args.loss == 'negative_sampling':
        loss_fn = negative_sampling_loss
    else:
        raise ValueError
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10, verbose=True)

    dataset = Dataset(data)
    dataloader = DataLoader(dataset, args.batch_size, True, args.num_workers)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x,y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            if args.loss == 'negative_sampling':
                neg = sample_negatives(y, num_class, args.num_neg_samples).to(device)

            model.zero_grad()
            pred = model(x)
            if args.loss == 'negative_sampling':
                loss = loss_fn(pred, y, neg)
            elif args.loss == 'softmax':
                loss = loss_fn(pred,y)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)
        torch.save(model, args.save)
        print('epcoh:%d\tloss:%f' % (epoch, total_loss))


if __name__ == '__main__':
    main()