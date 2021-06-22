import torch
import argparse
from model import NetA, NetB, NetC
from torch.utils.data import DataLoader
rom torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--root',type = str, help = 'Dictionary of dataset')
    parser.add_argument('--size',type = int, help = 'Size of CTC block needs')
    parser.add_argument('--num-blocks',type = int, help = 'Number of CTC blocks')
    parser.add_argument('--lr', type = float, help = 'Initial learning rate of SGD')
    parser.add_argument('--num-channels', type = int, help = 'Number of channels in each convolution layer')
    parser.add_argument('--dropout', type = float, help = 'Parameter of dropout layer')
    parser.add_argument('--num-heads', type = int, help = 'Number of heads in transformer encoder layers')
    parser.add_argument('--batch-size', type = int, help = 'Batch size')
    parser.add_argument('--device', type = str, help = 'cuda device')
    args = parser.parse_args()
    return args

def train(dataloader, model, loss_func, optimizer, epoch, writer = None):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for batch, data in enumerate(dataloader):
        images = data[0].to(device)
        labels = data[1].to(device)

        preds = model(images)
        loss = loss_func(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if writer != None:
            writer.add_scalar('Training loss',
                         loss.item()/batch_size,
                         epoch * size + batch)

        if batch % 500 == 0:
            loss, current = loss.item()/batch_size, batch * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_func, epoch, writer = None):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    if writer != None:
        writer.add_scalar('Testing loss',
                     test_loss,
                     epoch)
        writer.add_scalar('Testing accuracy',
                     correct*100,
                     epoch)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

if __name__ == '__main__':
    args = parse_args()
    
    device = args.device
    root = args.root
    training_data = torchvision.datasets.CIFAR10(root = root, train = True, transform = ToTensor())
    testing_data = torchvision.datasets.CIFAR10(root = root, train = False, transform = ToTensor())
    Net1 = NetA(num_class = 10, size = args.size, num_channel = args.num_channels, num_head = args.num_heads, num_layers = 0, num_blocks = args.num_blocks, dropout = args.dropout)
    Net1 = Net1.to(device)
    Net2 = NetB(num_class = 10, size = args.size, num_channel = args.num_channels, num_blocks = args.num_blocks, dropout = args.dropout)
    Net2 = Net2.to(device)
    Net3 = NetC(num_class = 10, size = args.size, num_channel = args.num_channels, num_blocks = args.num_blocks, dropout = args.dropout)
    Net3 = Net3.to(device)
    train_dataloader = DataLoader(training_data, batch_size = 20, shuffle = True)
    test_dataloader = DataLoader(testing_data, batch_size = 20)
    loss_func = nn.CrossEntropyLoss()
    print('# Parameters of NetA:{}'.format(sum([p.numel() for p in Net1.parameters()])))
    print('# Parameters of NetB:{}'.format(sum([p.numel() for p in Net2.parameters()])))
    print('# Parameters of NetC:{}'.format(sum([p.numel() for p in Net3.parameters()])))
    
    Net_list = {'NetA':Net1, 'NetB':Net2, 'NetC':Net3}
    for net in Net_list:
        epochs = 30
        lr = args.lr
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            if t % 10 == 0:
                lr = lr * 0.1
            optimizer = torch.optim.SGD(Net_list[net].parameters(), lr=lr, weight_decay = 1e-4, momentum=0.9)
            train(train_dataloader, Net_list[net], loss_func, optimizer, t)
            test(test_dataloader, Net_list[net], loss_func, t)
        print("Done!")
        path = './runs/{}/{}_blocks/{}.pth'.format(args.size**2, args.num_blocks,net)
        torch.save(Net_list[net], path)