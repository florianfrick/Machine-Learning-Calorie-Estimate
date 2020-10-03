import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils import data

import torch.optim as optim

from HDF5Dataset import HDF5Dataset
from Net import Net


num_epochs = 50
loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 4}

#transform function
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#dataset
dataset = HDF5Dataset('./Datasets', recursive=True, load_data=False, data_cache_size=4, transform=transform)

setsize = __len__(dataset)
print ('size: ' + setsize)

#dataloader
data_loader = data.DataLoader(dataset, **loader_params)



net = Net()



# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

#training
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in data_loader:
        # here comes your training loop

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data



        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() # Does the update

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        pass


print ('Finished Training')

PATH = './food_net.pth'
torch.save(net.state_dict(), PATH)