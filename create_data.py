import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser('Create distributional data')
    parser.add_argument('--output', type=str, default='data.txt', help='data output (default: data.txt')
    parser.add_argument('--output_dim', type=int, default=100, help='# output dimension (default: 100')
    parser.add_argument('--nsamples', type=int, default=1000000, help='# samples per distribution (default: 1M')
    args = parser.parse_args()

    # uniform
    uniform = np.random.randint(args.output_dim, size=[args.nsamples])

    # gaussian
    normal = np.random.randn(args.nsamples).clip(max=5)
    normal = np.digitize(normal, np.linspace(-5, 5, args.output_dim), right=True).astype(int)

    x = np.stack([[0]*len(uniform), [1]*len(normal)]).T.flatten()
    y = np.stack([uniform, normal]).T.flatten()

    data = np.stack([x,y])
    np.savetxt(args.output, data.astype(int).T, '%d')

if __name__ == '__main__':
    main()