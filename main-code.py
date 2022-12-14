import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

# import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
abspath = os.path.abspath(__file__)
print(abspath)

data_path = "images/train/"

products = pd.read_csv("products.csv")

# label list
classes = np.unique(products['GS1 Form'])

# Define relevant variables for the ML task
batch_size = 64
num_classes = len(classes)
learning_rate = 0.001
num_epochs = 10

# change this variable to save and load specific models
# to change the classifier optimizer you still need to change
# it in the code 
classifier_saved = 'models/model_ai_Adamax_2.pth'

# normalizing data ...
transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                        std=[0.2023, 0.1994, 0.2010])
                                    ])

# Device will determine whether to run the training on GPU or CPU.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Loading data with PyTorch"""

# loading data
train_data = torchvision.datasets.ImageFolder(root=data_path, 
                                            transform=transform)

train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                        batch_size = batch_size,
                                        pin_memory = True,
                                        shuffle = True)
# print(train_loader)


"""AI generated CNN"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


"""AI generated optimizer"""

cnn = CNN() # define your CNN
optimizer = optim.Adamax(cnn.parameters(), lr=learning_rate) # define the optimizer

# print("test")
"""Neural network training"""

def epoch(num_epochs, train_loader):
    # We use the pre-defined number of epochs to determine how many iterations to train the network on
    total_time = 0
    for epoch in range(num_epochs):
        epoch_time_start = time.time()
        #Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(train_loader): 
            if i % 10 == 0:
                print(i)
            # print(i)
            # Move tensors to the configured device
            # images = images.to(device)
            # labels = labels.to(device)
            
            images = images.to()
            labels = labels.to()
            
            # forward pass
            output = cnn(images) # pass the input data through the network
            loss = F.cross_entropy(output, labels) # compute the cross-entropy loss

            # backward pass
            optimizer.zero_grad() # zero the gradient buffers
            loss.backward() # compute the gradients
            optimizer.step() # update the network's parameters
        epoch_time_end = time.time()
        epoch_time = round(epoch_time_end - epoch_time_start, 2)
        total_time += epoch_time

        print('Epoch time: ', epoch_time)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
    print('Total time: ', round(total_time, 2))
        

def accuracy(cnn_):
    output = cnn_
    _, preds = torch.max(output.data, 1)
    acc_all = (preds == classes).float().mean()
    print(acc_all)


    # print("test")

def saver(cnn_):
    # saving model
    torch.save(cnn_.state_dict(), classifier_saved)
    print("Saved current model.")

def test():
    
    for i in range(1,14):
        print("article: ", i)
        img_path = 'images/test/test_' + str(i) + '.jpg'
        # print(img_path)
        # loading model
        cnn = CNN()
        cnn.load_state_dict(torch.load(classifier_saved))



        # prediction testing

        # img_path = img

        img = Image.open(img_path)
        img = transform(img)
        img = img.view(1, 3, 32, 32)

        output = cnn(img)

        prediction = int(torch.max(output.data, 1)[1].numpy())

        print(classes[prediction])

        plt.figure(figsize=(3,3))
        img = mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.axis('off')
        # plt.show()
    # accuracy(cnn)

if __name__ == '__main__':
    epoch(num_epochs, train_loader)

    saver(cnn)

    #img = 'images/test/test_4.jpg'
    test()
