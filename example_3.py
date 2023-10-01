"""
    PyTorch Example 3
    Building a PyTorch Model

    LeNet 5: One of the early CNNs
"""

import torch                    # all pytorch stuff
import torch.nn as nn           # for torch.nn.Module, the parent object for pytorch models
import torch.nn.functional as F # for activation and max pooling fns


def main():
    # create an instance of our model
    net = LeNet()
    print(net)  # the object tells us a few things about itself

    input = torch.rand(1, 1, 32 , 32) # placehold for a 32 x 32 black & white image
    print('Shape of input image:')
    print(input.shape)

    output = net(input) # we don't call forward() directly
    print('Raw output:')
    print(output)
    print(output.shape)
# end main function


# create LeNet in PyTorch
class LeNet(nn.Module):
    # __init__ for construction of layer that goes into the computation graph
    # loads any data artifacts it might need
    def __init__(self):
        super(LeNet, self).__init__()
        
        # assemble the network components
        # 2 convolutional layers, 3 fully connected layers

        # 1 input image channel (black & white), 
        # 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6 from the image dimension
        self.fc2 = nn.Linear(120, 84)   
        self.fc3 = nn.Linear(84, 10)
    # end __init__

    
    # where the actual computation happens
    # input -> output, add properties
    # defines any transformations between layers?
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        # flatten features
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # end forward


    # gets us the number of features when flattened?
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    # end num_flat_features


if __name__ == "__main__":
    main()
