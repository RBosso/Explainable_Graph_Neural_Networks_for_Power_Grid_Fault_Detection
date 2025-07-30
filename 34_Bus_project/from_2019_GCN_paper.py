
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 256, K=3)
        self.conv2 = GCNConv(256, 256, K=4)
        self.conv3 = GCNConv(256, 256, K=5)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Define the model
model = GCN(num_features=dataset.num_features, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.CrossEntropyLoss()

# Training loop
def train(model, optimizer, criterion, train_loader, val_loader, epochs):
    for epoch in range(1, epochs+1):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
        
        # Validate
        val_acc = test(model, val_loader)
        print(f'Epoch: {epoch}, Validation Accuracy: {val_acc:.4f}')

# Validation function
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
    return correct / total

# Train the model
train(model, optimizer, criterion, train_loader, val_loader, epochs=400)
