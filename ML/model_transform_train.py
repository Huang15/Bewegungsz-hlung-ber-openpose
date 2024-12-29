import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import Pre_Date
import numpy as np


##############

test_exl_path="E:\github\lunges_time.xlsx"
train_data =Pre_Date.main(test_exl_path,'Rudern')


# Initialize an empty list to store all features and labels
all_x = []
all_y = []
all_score = []

# Iterate through each dataset
for sample in train_data:
    x = np.array(sample['x'])
    y = sample['y']
    score = np.array(sample['score'])
    score=np.repeat(score, 2)
    # Extract the ninth feature as the center
    center = x[8]
    # Convert the coordinates of all features to coordinates relative to the ninth feature
    x_relative = x - center
    x_relative = x_relative.flatten()
    x1=(x_relative*score).tolist()
    # Add processed features and labels to the master list
    all_x.append(x1)
    all_y.append(y)
all_y=torch.tensor(np.array(all_y).ravel(),dtype=torch.float)
all_x=torch.tensor(np.array(all_x),dtype=torch.float)

#############################


# Defining datasets and data loaders
dataset = TensorDataset(all_x, all_y)
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add a dimension to match Transformer input requirements
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # Remove added dimensions
        x = self.fc(x)
        return self.sigmoid(x)

model = TransformerClassifier(input_dim=50, hidden_dim=128, num_classes=1)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training model
num_epochs = 50
best_val_loss = float('inf')
best_model_path = 'best_transformer_rudern.pth'

model.train()

for epoch in range(num_epochs):
    # training phase
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()

    # validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output.squeeze(), target)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')

    # Check if it is the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved Best Model: Epoch {epoch+1}, Validation Loss: {val_loss}')

    model.train()

print("Training complete.")

# Load the best model and test it
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    test_loss = 0
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output.squeeze(), target)
        test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')

