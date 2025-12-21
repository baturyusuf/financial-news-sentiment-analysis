import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)


class ProjectionClassifier(nn.Module):
    def __init__(self, input_dim=768, proj_dim=64):
        super().__init__()
        self.proj = ProjectionHead(input_dim, proj_dim)
        self.classifier = nn.Linear(proj_dim, 1)

    def forward(self, x):
        z = self.proj(x)
        return self.classifier(z).squeeze(1), z


def train_projection(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    lr=1e-3,
    device="cpu"
):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ProjectionClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Projection] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model
