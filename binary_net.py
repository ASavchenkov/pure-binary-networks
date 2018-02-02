import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


from layers import Factorized_Linear, Factorized_BN

import binary_layers as bl

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Pad(2),
                       transforms.ToTensor()
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.Pad(2),
                       transforms.ToTensor()
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


#deals with getting bytes correct, and batching along bits

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        depth = 32

        self.bn1 = nn.BatchNorm1d(32*32)

        self.fc1_list = nn.ModuleList([nn.Linear(32*32,32*32) for i in range(depth)])
        self.bn1_list = nn.ModuleList([nn.BatchNorm1d(32*32) for i in range(depth)])
        self.fc2 = nn.Linear(32*32,10)

    def forward(self, x):
        x = x.view(-1,32*32)

        for linear, bn in zip(self.fc1_list, self.bn1_list):
            x = x + linear(F.relu(bn(x)))
        x = self.fc2(x)
        return F.log_softmax(x)

#be careful using this on labels, as that results in a bunch of
#duplicates (octuplicates technically)
def preprocess_binary_data(data):
    data = data*255.9
    data = data.byte()
    data = data.numpy()

    data = np.unpackbits(data, axis=1)
    data = np.packbits(data, axis=0)
    
    data = torch.ByteTensor(data)
    return data


class binary_MLP(nn.Module):
    def __init__(self,width, depth, end_width):
        super().__init__()
        self.end_width = end_width
        self.layers = nn.Sequential(*[bl.Residual_Binary(width) for i in range(depth)])
    
    def forward(self, x):
        x = self.layers(x)
        #just chop off what you don't need. This should make zero gradients for every
        #other output, making this a true "reduction" without much work
        return x[:,:self.end_width]
model = MLP()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # data = preprocess_binary(data) #do this for binary variants
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
