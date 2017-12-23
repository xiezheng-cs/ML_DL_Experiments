import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from model import LeNet

# define hyperparameter
batch_size = 128
learning_rate = 1e-2
num_epoches = 500

Download=False
transforms=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))])

# download MNIST dtaaset
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms, download=Download)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms,download=Download)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4)

#define LeNet model
model = LeNet(1, 10)  
use_gpu = torch.cuda.is_available()  
if use_gpu:
    model = model.cuda()

# define loss and optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# start training
for epoch in range(num_epoches):

    train_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        
        # forward propagate
        out = model(img)
        loss = criterion(out, label)
        train_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        train_acc += num_correct.data[0]
        
        # back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, train_loss / (len(train_dataset)),train_acc / (len(train_dataset))))
    
torch.save(model, 'LeNet.pkl')     #save model