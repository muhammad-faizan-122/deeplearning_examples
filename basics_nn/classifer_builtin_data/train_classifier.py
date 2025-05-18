import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


# define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TrainClassifier:
    def __init__(self, batch_size=BATCH_SIZE, epochs=EPOCHS):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_loader, self.test_loader = self._load_data()
        self.model = NeuralNetwork().to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.train_losses = []

    def _load_data(self):
        train_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        return train_loader, test_loader

    def _train_epoch(self):
        self.model.train()
        size = len(self.train_loader.dataset)
        running_loss = 0.0
        for batch_idx, (X, y) in enumerate(self.train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Forward pass
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                current = (batch_idx + 1) * len(X)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

        avg_train_loss = running_loss / size
        self.train_losses.append(avg_train_loss)

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        size = len(self.test_loader.dataset)

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        avg_loss = test_loss / len(self.test_loader)
        accuracy = correct / size
        print(f"Test Accuracy: {accuracy:.1f}, Avg Loss: {avg_loss:.6f}")

    def train(self):
        print(f"Using device: {DEVICE}")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self._train_epoch()
            self._test()
        print("Training complete.")
        self._save_model()
        self._plot_metrics()

    def _save_model(self, path="model.pth"):
        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print("Failed to save model.")
            raise e

    def _plot_metrics(self):
        epochs = range(1, self.epochs + 1)

        plt.figure(figsize=(12, 5))
        # Plot loss
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    trainer = TrainClassifier()
    trainer.train()
