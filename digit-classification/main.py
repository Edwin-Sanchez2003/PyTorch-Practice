"""
    PyTorch Example - Mini Project

    Building and Training a Digit Classifier 
    for the CIFAR Dataset using LeNet 5

    This is from the Jupyter Notebook #4:
    "A Simple PyTorch Training Loop"
    In the video1/video1/ directory.

    Credit: Introduction to PyTorch (Video Name), 
            PyTorch Beginner Series (Series Name), 
            PyTorch (YouTube Channel)
    Link: https://youtu.be/IC0_FRiX-sw?si=K6t_d8e_pZZ7HwGO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Step 1: Load the Data

    # Defines how to transform the data. In this example,
    # the data will be first converted to a tensor (transforms.ToTensor()),
    # then normalized to help the model to train (transforms.Normalize())
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # downloads or retrieves the CIFAR train dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    # downloads or retrieves the test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    # list of the 10 classes in the CIFAR dataset
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    data_iter = iter(trainloader)
    images, labels = next(data_iter)

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))   
    # show images
    imshow(torchvision.utils.make_grid(images))

    # Step 2: Define and Initialize the Model
    # initialize our model
    model = LeNet()

    # define our loss function
    # Entropy Loss is common for classification models
    loss_fn:nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    # define our optimizer - how we do backprop to train the network
    # this is a simple optimizer, Stochastic Gradient Descent (SGD)
    # we need to define 2 params, the learning rate (how far to step out each time)
    # and the momentum (a extra trait that allows are model to build up momentum when
    # moving in a certain direction along the gradient)
    # additionally, we pass in the parameters (weights) of the model, which the optimizer
    # will adjust based off of the method of learning (in this case, SGD)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Step 3: Training
    # define the training loop
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        # loop over a set of batches of the training dataset
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # pass the image in, get the model's prediction
            outputs = model(inputs)
            # calculate the loss - how far we are from what the answer should be
            loss = loss_fn(outputs, labels)
            # perform backgropagation to calculate the gradients that will direct the learning
            loss.backward()
            # optimizes the parameters of the model
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            # end if
        # end for loop over batches of the training datset
    # end for loop over the number of epochs to train for
    print('Finished Training!')
    
    # Step 4: Testing Our Model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # end for loop over test data
    # end with torch.no_grad()

    # print the accuracy of our model
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
# end main

# shows an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# end imshow


# LeNet variant
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    # end __init__

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # end forward
# end LeNet


if __name__ == "__main__":
    main()
