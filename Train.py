import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
#from autoaugment import ImageNetPolicy
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import Net as net

num_epochs = 5
loader_params = {'batch_size': 4, 'shuffle': True, 'num_workers': 4}

#transform function
transform = transforms.Compose(
    [transforms.RandomRotation(30),
    transforms.RandomResizedCrop(32),
    #transforms.CenterCrop(32),
    transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

transform_test = transforms.Compose(
    [transforms.Resize((32,32),2),
    transforms.CenterCrop((32,32)),
    transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
     #([0.485,0.456,0.406], [0.229,0.224,0.225])])
     #((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#train_data = torchvision.datasets.ImageFolder('./images/baklava', transform=transform)
#train_data = torchvision.datasets.ImageFolder('./smallimgs', transform=transform)
train_data = torchvision.datasets.ImageFolder('./images', transform=transform)
train_data_subset = data.Subset(train_data, np.random.choice(len(train_data), 10000, replace=False))

#dataloader
data_loader = data.DataLoader(train_data_subset, **loader_params)

print (len(train_data))
print (len(data_loader))

#categories
with open('./classes.txt') as file:
        classes = file.readlines()
classes = [c.strip() for c in classes]

#print categories
#print (classes)


def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('testimg.png')

# get some random training images
#dataiter = iter(data_loader)
#images, labels = dataiter.next()

#show images
#imshow(torchvision.utils.make_grid(images))
#print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 360) 
        self.fc2 = nn.Linear(360, 140)
        self.fc3 = nn.Linear(140, len(classes)) #101; len(classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #maybe no momentum

#training


for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        
        inputs, labels = data# get the inputs; data is a list of [inputs, labels]

        optimizer.zero_grad()# zero the parameter gradients

        # forward + backward + optimize
        outputs = net(inputs) #net.forward(inputs)?
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() # Does the update

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print ('Finished Training')

PATH = './food_net.pth'
torch.save(net.state_dict(), PATH)




dataiter = iter(data_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('Reality: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
PATH = './food_net.pth'
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in data_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))