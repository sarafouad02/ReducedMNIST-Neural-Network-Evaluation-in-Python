import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transform: convert to tensor and normalize to [0,1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Load the full training and test datasets (assume dataset folder structure)
train_dataset = torchvision.datasets.ImageFolder(
    root="D:/UNI/EECE 4 second term/Nueral Networks/Projects Mohsen/assignment1_mohsen/Reduced MNIST Data/Reduced Trainging data", 
    transform=transform
)
test_dataset = torchvision.datasets.ImageFolder(
    root="D:/UNI/EECE 4 second term/Nueral Networks/Projects Mohsen/assignment1_mohsen/Reduced MNIST Data/Reduced Testing data", 
    transform=transform
)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    start_time = time.time()  # Start time for this epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
    
    end_time = time.time()  # End time for this epoch
    print(f"Epoch {epoch} training time: {end_time - start_time:.2f} seconds")

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    start_time = time.time()  # Start time for testing

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    end_time = time.time()  # End time for testing
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    print(f"Testing time: {end_time - start_time:.2f} seconds")
    return accuracy

#add Conv layer
class LeNetVar4(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetVar4, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 1x28x28 -> 6x28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)         # 6x28x28 -> 6x14x14

        # Second convolutional block
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)              # 6x14x14 -> 16x10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)         # 16x10x10 -> 16x5x5

        # Extra convolutional layer
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # 16x5x5 -> 32x5x5
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)         # 32x5x5 -> 32x2x2

        # Fully connected layers; adjust input dimensions accordingly (32*2*2 = 128)
        self.fc1 = nn.Linear(32*2*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Optionally add dropout if you want to regularize further
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # Conv1 block with LeakyReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = self.pool1(x)
        
        # Conv2 block with LeakyReLU
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.pool2(x)
        
        # Conv3 block with LeakyReLU
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(-1, 32*2*2)
        
        # Fully connected layers with dropout and LeakyReLU
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.dropout(x)
        x = self.fc3(x)  # Logits (no softmax)
        return x



# Instantiate the model, define loss and optimizer
model_var4 = LeNetVar4(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_var4.parameters(), lr=0.001)

# Train and evaluate
epochs = 10
total_start_time = time.time()

for epoch in range(1, epochs + 1):
    train_one_epoch(model_var4, device, train_loader, optimizer, criterion, epoch)

total_training_time = time.time() - total_start_time  # Compute total training time
print(f"Total training time: {total_training_time:.2f} seconds.")

accuracy = test(model_var4, device, test_loader, criterion)
