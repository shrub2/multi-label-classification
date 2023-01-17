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
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
abspath = os.path.abspath(__file__)
# print(abspath)

print(f'Do I have GPU? : {torch.cuda.is_available()}\n')  # if True you have gpu which can be used, otherwise not!

data_path = "images/train/"

products = pd.read_csv("products.csv")
train_products = pd.read_csv("products_train.csv")
valid_products = pd.read_csv("products_valid.csv")


# label list
classes = np.unique(products['GS1 Form'])
train_classes = np.unique(train_products['GS1 Form'])
valid_classes = np.unique(valid_products['GS1 Form'])



# Define relevant variables for the ML task
batch_size = 64
num_classes = len(classes)
learning_rate = 0.001
num_epochs = 20

# change this variable to save and load specific models
# to change the classifier optimizer you still need to change
# it in the code 
classifier_saved = 'models/model_Adamax_form.pth'

# normalizing data ...
transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                        std=[0.2023, 0.1994, 0.2010])
                                    ])

# Device will determine whether to run the training on GPU or CPU.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)  # check used device "cuda:0" for gpu or "cpu"
# print(torch.cuda.current_device())  # prints 0 for gpu, -1 for cpu

"""Loading data with PyTorch"""

# loading data
train_data = torchvision.datasets.ImageFolder(root=data_path, 
                                            transform=transform)

train_data, valid_data = torch.utils.data.random_split(train_data, [len(train_products), len(valid_products)])


train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                        batch_size = batch_size,
                                        pin_memory = True,
                                        shuffle = True)

valid_loader = torch.utils.data.DataLoader(dataset = valid_data,
                                        batch_size = batch_size,
                                        pin_memory = True,
                                        shuffle = True)

print(f'Size train_data: {len(train_loader)}')
print(f'Size valid_data: {len(valid_loader)}\n')


"""AI generated CNN"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # self.conv3 = nn.Conv2d(64, 128, 3) # new conv layer
        # self.conv4 = nn.Conv2d(128, 256, 3) # new conv layer

        self.fc1 = nn.Linear(50176, 128)
        self.dropout = nn.Dropout(0.2) # new regularization layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.conv3(x) # new layer
        # x = F.relu(x) # new layer
        # x = self.conv4(x) # new layer
        # x = F.relu(x) # new layer
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.dropout(x) # new regularization layer
        x = F.relu(x)
        x = self.fc2(x)
        return x


"""AI generated optimizer"""

cnn = CNN() # define your CNN

# cnn.to(device) # move the model to GPU
# images = images.to(device)
# labels = labels.to(device)

# weigh_decay = 1e-5 is a new regularization layer
optimizer = optim.Adamax(cnn.parameters(), lr=learning_rate, weight_decay=1e-5) # define the optimizer
# optimizer.to(device)

# print(cnn.conv1.weight.device)  # prints "cuda:0" for gpu or "cpu"

# print("test")
"""Neural network training"""
criterion = nn.CrossEntropyLoss()
def epoch(num_epochs, train_loader):
    # We use the pre-defined number of epochs to determine how many iterations to train the network on
    total_time = 0
    accuracies = []
    loss_train = []

    accuracies_eval = []
    loss_eval = []

    for epoch in range(num_epochs):
        # print(f'Epoch: {epoch + 1}')
        epoch_time_start = time.time()
        #Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=(f'Epoch {epoch + 1}'))): 
            # if i % 10 == 0:
                # print(i)
            # print(i)
            # Move tensors to the configured device
            # images = images.to(device)
            # labels = labels.to(device)
            
            # images = images.to()
            # labels = labels.to()
            
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
        
        loss_train.append(loss.item())
        print('Epoch time: ', epoch_time)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        # measure accuracy per epoch
        # num_iterations = num_epochs
        # for epoch in range(num_iterations):
            # epoch(1, train_loader)
        print(f'\nMeasuring Epoch {epoch + 1}')
        accuracy = test_accuracy(cnn, train_loader)
        accuracies.append(accuracy)
        print(f'Stored accuracies: {accuracies}')

        # validation loss, accuracy per epoch
        avg_loss, avg_acc = evaluate(cnn, valid_loader)
        loss_eval.append(avg_loss)        
        accuracies_eval.append(avg_acc)
        print(f'Stored loss_eval: {loss_eval}')
        print(f'Stored accuaracies_eval: {accuracies_eval}\n')
        

    print('Total time: ', round(total_time, 2))
    return accuracies, loss_train, accuracies_eval, loss_eval
        

def accuracy(cnn_):
    output = cnn_
    _, preds = torch.max(output.data, 1)
    acc_all = (preds == classes).float().mean()
    print(acc_all)


    # print("test")

# test loader
# test_data = torchvision.datasets.ImageFolder(root=data_path, 
#                                             transform=transform)
# test_loader = torch.utils.data.DataLoader(dataset = test_data,
#                                         batch_size = batch_size,
#                                         pin_memory = True,
#                                         shuffle = True)


def test_accuracy(model, train_loader):
    model.eval() # set model to evaluation mode
    correct = 0
    total = 0
    print(f'Testing accuracy cumulative epochs\n please wait...')
    with torch.no_grad(): # temporarily set all the requires_grad flag to false
        for images, labels in tqdm(train_loader, desc="Accuracy test"):
            # print(f'Doing something: {labels}')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy_raw = correct / total
    accuracy = 100 * accuracy_raw
    print('Accuracy of the network on the test images: %d %%\n' % (accuracy))
    return accuracy_raw


# valid loader is after the train_loader 
def evaluate(model, valid_loader):
    model.eval()  # set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():  # turn off gradients to speed up computation
        for i, (images, labels) in enumerate(tqdm(valid_loader, desc="Validation model")):
            outputs = model(images)  # forward pass
            _, predicted = torch.max(outputs.data, 1)  # get the predicted class
            total_correct += (predicted == labels).sum().item()  # count number of correct predictions
            total += labels.size(0)  # count total number of images
            loss = criterion(outputs, labels)  # compute loss
            total_loss += loss.item()  # accumulate loss

    # compute average loss and accuracy
    avg_loss = total_loss / len(valid_loader)
    avg_acc = total_correct / total
    print(f'\nValidation Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')
    return avg_loss, avg_acc


# graphs
# def graph_accuracy(accuracy):
#     plt.plot(accuracy)
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.show()
def graph_accuracy(acc_train, acc_eval):
    plt.plot(range(1, num_epochs + 1), acc_train, label = 'Training')
    plt.plot(range(1, num_epochs + 1), acc_eval, label = 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# def graph_loss(loss):
#     plt.plot(loss)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss')
#     plt.show()
def graph_loss(loss_train, loss_eval):
    plt.plot(range(1, num_epochs + 1), loss_train, label = 'Training')
    plt.plot(range(1, num_epochs + 1), loss_eval, label = 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




def saver(cnn_):
    # saving model
    torch.save(cnn_.state_dict(), classifier_saved)
    print("Saved current model.")

def test():
    
    for i in range(1,15):
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
        plt.show()
    # accuracy(cnn)

if __name__ == '__main__':
    st = time.perf_counter()
    # epoch(num_epochs, train_loader)
    acc_train, loss_train, acc_eval, loss_eval = epoch(num_epochs, train_loader)
    # graph_accuracy(epoch(num_epochs, train_loader))
    # test_accuracy(cnn, test_loader)

    graph_accuracy(acc_train, acc_eval)
    graph_loss(loss_train, loss_eval)
    evaluate(cnn, valid_loader)
    saver(cnn)

    et = time.perf_counter()
    total = et-st
    print(f'Total runtime: {round(total / 60, 2)} minutes.')

    #img = 'images/test/test_4.jpg'
    test()
