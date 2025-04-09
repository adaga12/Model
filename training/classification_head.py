import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from bnt_model import BNTContrastiveEncoder
import yaml

# Get directories from config.yaml
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

corr_matrices_file = config["corr_matrices_file"]
labels_abide_file = config["labels_abide_file"]

# Create a Labeled Dataset Class
# This dataset will return a (correlation_matrix, label) pair at each index
class LabeledGraphDataset(Dataset):
    def __init__(self, corr_matrices, labels):
        self.X = corr_matrices
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)   # shape: (200, 200)
        y = torch.tensor(self.y[idx], dtype=torch.long)      # label: 0 or 1
        return x, y

# Load the data
X = np.load(corr_matrices_file)
y = np.load(labels_abide_file)

print(X.shape, y.shape)  # should be (884, 200, 200), (884,)

# Split into train/test
# Use stratified sampling to preserve ASD/TDC balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create DataLoaders
train_dataset = LabeledGraphDataset(X_train, y_train)
test_dataset = LabeledGraphDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the classifier model
class BNTClassifier(nn.Module):
    def __init__(self, pretrained_path=None, freeze_encoder=False):
        super().__init__()
        self.encoder = BNTContrastiveEncoder()

        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(state_dict)
            print("✅ Loaded pretrained BNT weights.")

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classifier head: takes 128-D embedding → predicts 2 classes
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Output logits for 2 classes (TDC, ASD)
        )

    def forward(self, x):
        z = self.encoder(x)       # (B, 128)
        out = self.classifier(z)  # (B, 2)
        return out

# Initialize for training 
device = torch.device("cpu")  # since CUDA is not available
model = BNTClassifier(pretrained_path="bnt_contrastive_pretrained.pth", freeze_encoder=True)
model.to(device)

# Set up training 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop 
for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")

# Save trained classifier
torch.save(model.state_dict(), "bnt_classifier_trained.pth")
print("✅ Saved trained classifier to bnt_classifier_trained.pth")

# Evaluate on test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"🧪 Test Accuracy: {correct / total:.4f}")
