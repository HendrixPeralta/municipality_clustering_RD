#%%
# PyTorch and utilities
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Torchvision for image processing and datasets
import torchvision 
import torchvision.transforms as transforms 
from torchvision.datasets import ImageFolder

# timm for pretrained models
import timm 

# Visualization and data handling
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np   

import sys
# %%
# libraries versions

print("System version:", sys.version)
print("PyTorch version", torch.__version__)
print("Torchvision version", torchvision.__version__)
print("Numpy version", np.__version__)
print("Pandas version", pd.__version__)

# %%
# %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")

# print("Path to dataset files:", path)
path = r"C:\Users\hendr\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\train"

# %%
class PlayingCardDataset (Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
# %%
dataset = PlayingCardDataset(data_dir=path)

# %%
len(dataset)
# %%
image, label = dataset[6000]
image

# %%
# get a dictionary associating target values with folder names 
data_dir = path 
target_to_class = {v:k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

# %%
# resize the images and convert them into pytorch tensors
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

# %%
dataset = PlayingCardDataset(path,transform)
# %%
image, label = dataset[100]
image.shape
# %%
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# %%
for images, labels in dataloader:
    break
# %%
images.shape
# %%
# Classificator model 
# %%
class simpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(simpleCardClassifier, self).__init__()
        
        # Definition of all parts of the model 
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        enet_out_size = 1280
        # make classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)
    
    def forward(self,x):
        x = self.features(x) 
        output = self.classifier(x)
        return output
# %%
model = simpleCardClassifier()
print(model)
# %%
example_out = model(images)
example_out.shape #[batch size, num of classes]
# %%
# Training loop 

# Loss function 
criterion = nn.CrossEntropyLoss()
# Optimizer 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Test 
criterion(example_out, labels)
# %%

train_folder = r"C:\Users\hendr\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\train"
valid_folder = r"C:\Users\hendr\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\valid"
test_folder = r"C:\Users\hendr\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\test"

train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# Loop 

num_epoch = 5 
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = simpleCardClassifier(num_classes=53)
model.to(device)

# Loss function 
criterion = nn.CrossEntropyLoss()
# Optimizer 
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epoch): 
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    
    # validation phase 
    
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
        val_loss = running_loss / len(valid_loader.dataset)
        val_losses.append(val_loss)
        
    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss}, Validation loss: {val_loss}")
        
        # %% 
        
    # Visualize losses 
    
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.show()
# %%


# Reevaluation 

import torch
import torchvision.transforms as transforms
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np 

# Load and process the images 
def preprocessImage(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# predict using the model 
def predict (model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# Visualization
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1,2, figsize=(14,7))
    
    # Display image 
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0,1)
    
    plt.tight_layout
    plt.show()
    

  # %%
  
from glob import glob
  
test_images = glob(r"C:\Users\hendr\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\test\*\*")
test_examples = np.random.choice(test_images, 10)

for example in test_examples:
    original_image, image_tensor = preprocessImage(example, transform)  
    probabilities = predict(model, image_tensor, device)
    
    #Gives the class names 
    class_names = dataset.classes
    visualize_predictions(original_image, probabilities, class_names)
# %%
