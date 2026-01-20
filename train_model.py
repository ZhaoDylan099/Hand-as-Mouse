from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from loaddata import FreiHandDataset
from torchvision import models
from torch import nn
import torch.optim as optim
import os

transform = transforms.Compose([
    transforms.ToTensor(),                    # Convert PIL Image to PyTorch tensor (C,H,W) scaled [0,1]
    transforms.Resize((224, 224)),           # Optional if not already resized in Dataset
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # Normalize to [-1,1] or standardize
])

dataset = FreiHandDataset(os.getcwd(), transform=transform)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f'Using device: {device}')

num_keypoints = 21  # 21 hand joints
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_keypoints*3)  # x,y,z
model = model.to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(5):
    print(f'Starting epoch {epoch+1}/{5}')
    model.train()
    running_loss = 0.0
    i = 0
    for images, keypoints in train_loader:
        images = images.to(device)
        keypoints = keypoints.view(keypoints.size(0), -1).to(device)  # Flatten keypoints

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        if i % 100 == 99:
            print(f'Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        i += 1

    
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, keypoints in test_loader:
        images = images.to(device)
        keypoints = keypoints.view(keypoints.size(0), -1).to(device)

        outputs = model(images)
        test_loss += criterion(outputs, keypoints).item() * images.size(0)

avg_test_loss = test_loss / len(test_dataset)
print(f'Average Test Loss: {avg_test_loss:.4f}')
print('Training complete.')

torch.save(model.state_dict(), "resnet_hand_keypoints.pth")