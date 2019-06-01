import torch
from torch import optim
import argparse
import generate
import utils

parser = argparse.ArgumentParser(description='Train a ReLU network on a synthetic dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--lmbda', type=float, default=1e-4)
parser.add_argument('--num_hidden', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_points', type=int, default=10)
parser.add_argument('--cuda', dest='cuda', action='store_true')
args = parser.parse_args()

def create_model():
    model = torch.nn.Sequential()
    model.add_module("linear1", torch.nn.Linear(1, args.num_hidden))
    model.add_module("relu1", torch.nn.ReLU())
    for i in range(2,args.num_layers):
        model.add_module("linear%s"%i, torch.nn.Linear(args.num_hidden, args.num_hidden))
        model.add_module("relu%s"%i, torch.nn.ReLU())
    model.add_module("linear%s"%args.num_layers, torch.nn.Linear(args.num_hidden, 1))
    
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.uniform_(m.bias, -1, 1)
            
    return model
    
def cost(model):
    named_weights = filter(lambda p: 'weight' in p[0] and p[1].requires_grad, model.named_parameters())
    C = 0.5 * sum(w[1].norm(2)**2 for w in named_weights)
    return C

def train(model, criterion, optimizer, x, y):
    output = model(x).view(-1)
    cost_loss = cost(model)
    pred_loss = criterion(output, y)
    total_loss = pred_loss + args.lmbda * cost_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return pred_loss.item(), cost_loss.item()

def main():
    x, y, lx, elx = generate.random_generate(args.num_points)
    tx, ty = torch.tensor(x).float(), torch.tensor(y).float()
    model = create_model()
    if args.cuda: tx, ty, model = tx.cuda(), ty.cuda(), model.cuda()
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    for epoch in xrange(int(5e8)):
        loss = train(model, criterion, optimizer, tx, ty)
        if epoch % 5e3 == 0: utils.plot(epoch, model, loss, args.cuda, args.num_points, args.num_layers, args.num_hidden, x, y, lx, elx, args.lmbda)
            
main()
