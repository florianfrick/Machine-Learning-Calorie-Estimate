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
import Train
import Net as net

num_epochs = 4
loader_params = {'batch_size': 4, 'shuffle': True, 'num_workers': 4}

#transform function
transform_test = transforms.Compose(
    [transforms.Resize((32,32),2),
    transforms.CenterCrop((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     #([0.485,0.456,0.406], [0.229,0.224,0.225])])


train_data = torchvision.datasets.ImageFolder('./images', transform=transform_test)
train_data_subset = data.Subset(train_data, np.random.choice(len(train_data), 20000, replace=False))

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

dataiter = iter(data_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('Reality: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = net.Net()
PATH = './food_net.pth'
net.load_state_dict(torch.load(PATH))

outputs = net(images)

predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))