import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data



# Device config - checks if a CUDA enabled GPU is available, if not uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 4  # Number of times the entire dataset will be used to train the model
batch_size = 4  # Number of samples that will be fed to the model at once
learning_rate = 0.001  # A value that determines how much the model will adjust its parameters during training based on the error it is making

# Dataset has PILImage images of range [0, 1]
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # Normalizes the pixel values of the image tensor to have a mean of 0.5 and a standard deviation of 0.5 for each channel (RGB)
)

# Load the CIFAR10 dataset and apply the transform
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

# Use DataLoader to load the datasets in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False)

# CIFAR10 has 10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer'
        'dog', 'frog', 'horse', 'ship', 'truck')

# Implement conv net using PyTorch's nn.Module
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # input channel, output channel, kernel size
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling with stride 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # convolution layer, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # reshape tensor for fully connected layers
        x = F.relu(self.fc1(x))  # fully connected layers with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # final fully connected layer with 10 outputs
        return x


# Instantiate the ConvNet model and move it to the device
model = ConvNet().to(device)

# Define the loss function (cross-entropy) and optimizer (stochastic gradient descent)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('finished training')

# Evaluate the model on the test set after training is complete
with torch.no_grad(): # disable gradient computation to save memory
    n_correct = 0  # number of correctly classified samples
    n_samples = 0  # total number of samples
    n_class_correct = [0 for i in range(10)]  # number of correctly classified samples per class
    n_class_samples = [0 for i in range(10)]  # total number of samples per class
    for images, labels in test_loader:  # loop over the test dataset
        images = images.to(device)  # move the images to the device (CPU or GPU)
        labels = labels.to(device)  # move the labels to the device (CPU or GPU)
        outputs = model(images)  # forward pass: compute the predicted outputs for the images
        # torch.max returns the maximum value and its index along a given axis, here we want the index along the 1st axis
        # since each row in the output tensor corresponds to a sample and each column corresponds to a class
        _, predicted = torch.max(outputs, 1)  # get the index of the class with the highest probability
        n_samples += labels.size(0)  # increment the total number of samples by the batch size
        n_correct += (predicted == labels).sum().item()  # increment the number of correctly classified samples by the number of samples for which the predicted class equals the true class
        
        # loop over the samples in the batch
        for i in range(batch_size):
            label = labels[i]  # get the true class of the i-th sample in the batch
            pred = predicted[i]  # get the predicted class of the i-th sample in the batch
            if (label == pred):  # if the true and predicted classes match
                n_class_correct[label] += 1  # increment the number of correctly classified samples for that class by 1
            n_class_samples[label] += 1  # increment the total number of samples for that class by 1

    # compute the overall accuracy on the test set and print it
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    # compute the accuracy per class on the test set and print it
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')



