## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def print_tensor_info(x):
    print("x type:", type(x))
    print("x shape:", x.shape)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) 
        # to avoid overfitting

        # W is the input size (in this case, the width of the input image).
        # F is the filter size (in this case, the width of the convolutional filter/kernel).
        # S is the stride (the number of pixels by which we move the filter/kernel across the input image).        
        
        
        
        
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        # the output Tensor for one image, will have the dimensions: (32, 221, 221)
        # after one pool layer, this becomes (32, 110, 110); 110.5 is rounded down
        self.conv1 = nn.Conv2d(1, 32, 4)  # Convolutional layer 1
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for one image, will have the dimensions: (64, 108, 108)
        # after one pool layer, this becomes (64, 54, 54);        
        self.conv2 = nn.Conv2d(32, 64, 3)  # Convolutional layer 2
        
        
        # third conv layer: 64 inputs, 128 outputs, 2x2 conv
        ## output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # the output Tensor for one image, will have the dimensions: (128, 53, 53)
        # after one pool layer, this becomes (128, 26, 26); 26.5 is rounded down        
        self.conv3 = nn.Conv2d(64, 128, 2)  # Convolutional layer 3
        
        # fourth conv layer: 128 inputs, 256 outputs, 1x1 conv
        ## output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (256, 26, 26)
        # after one pool layer, this becomes (256, 13, 13);        
        self.conv4 = nn.Conv2d(128, 256, 1)  # Convolutional layer 4
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 256 inputs * the 4*4 filtered/pooled map size
        # 1000 output channels
        
        self.fc1 = nn.Linear(256 * 13 * 13, 1000)  # Fully connected layer 1
        
        # 1000 inputs, 1000 output channels
        self.fc2 = nn.Linear(1000, 1000)  # Fully connected layer 2
        # 1000 input channels, 136 output channels(136 values, 2 for each of the 68 keypoint (x, y) pairs)
        self.fc3 = nn.Linear(1000, 136)  # Fully connected layer 3
        
        self.dropout = nn.Dropout(p=0.4)  # Dropout layer

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Define layers: NAIMISHNET LAYER-WISE ARCHITECTURE
        #Layer Number, Layer Name, Layer Shape

        # Apply convolutional layers with ReLU activation and pooling
        # 1 Input1 (1, 224, 224)
        
        # 2 Convolution2d1 (32, 221, 221)***
        # 3 Activation1 (32, 221, 221)
        # 4 Maxpooling2d1 (32, 110, 110)
        # 5 Dropout1 (32, 110, 110)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)        
                
        # 6 Convolution2d2 (64, 108, 108)***
        # 7 Activation2 (64, 108, 108)
        # 8 Maxpooling2d2 (64, 54, 54)
        # 9 Dropout2 (64, 54, 54)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
       
        # 10 Convolution2d3 (128, 53, 53)***
        # 11 Activation3 (128, 53, 53)
        # 12 Maxpooling2d3 (128, 26, 26)
        # 13 Dropout3 (128, 26, 26)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        # 14 Convolution2d4 (256, 26, 26)***
        # 15 Activation4 (256, 26, 26)
        # 16 Maxpooling2d4 (256, 13, 13)
        # 17 Dropout4 (256, 13, 13)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        
#         print_tensor_info(x)
        
        # 18 Flatten1 (43264)
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 256 * 13 * 13)
        
#         print_tensor_info(x)
        
        # Apply fully connected layers with ReLU activation and dropout
        # 19 Dense1 (1000)
        # 20 Activation5 (1000)
        # 21 Dropout5 (1000)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 22 Dense2 (1000)
        # 23 Activation6 (1000)
        # 24 Dropout6 (1000)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 25 Dense3 (136)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x


