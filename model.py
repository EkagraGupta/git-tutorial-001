import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)
        self.fc2 =  nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def load_mnist_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])
    
    train_set = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)
    test_set = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transform)
    
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epochs[{epoch+1}/{epochs}]-Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0., 0.

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    accuracy = correct/total

    print(f'Test Accuracy: {accuracy*100}')
def save_model(model, filename='mnist_model.pth'):
    torch.save(model.state_dict(), filename)
    print(f'Model is saved as {filename}')

def save_model(model, filename='mnist_model.pth'):
    torch.save(model.state_dict(), filename)
    print(f'Model is saved as {filename}')

def main():
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),
                           lr=1e-3)
    train_loader, test_loader = load_mnist_data()

    train_model(model, train_loader, criterion, optimizer)

    save_model(model)

    evaluate_model(model, test_loader)


if __name__=='__main__':
    main()


