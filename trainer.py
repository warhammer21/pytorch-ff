import argparse
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
from nn_model import Net  # Assuming your model is in nn_model.py
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.data = df.to_numpy()  # Convert to NumPy array for easier indexing

    def __len__(self):
        return self.data.shape[0]  # Number of samples

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx, :-1], dtype=torch.float32)  # Features
        label = torch.tensor(self.data[idx, -1], dtype=torch.float32)      # Label
        return features, label


def train_model(args):
    # Load dataset
    df = pd.read_csv(args.dataset_path)
    df = df[['temp', 'hour_of_day', 'traffic_volume']]  # Select relevant columns

    # Create the dataset and data loader
    traffic_dataset = TrafficDataset(df)
    train_loader = DataLoader(traffic_dataset, batch_size=args.batch_size, shuffle=True)

    # Define model, loss function, and optimizer
    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        epoch_loss = 0
        for features, labels in train_loader:
            predictions = model(features)  # Forward pass
            loss = criterion(predictions.squeeze(), labels)  # Compute loss
            epoch_loss += loss.item()

            optimizer.zero_grad()  # Reset gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a feed-forward neural network on traffic data.")
    parser.add_argument("--dataset_path", type=str, default="train_scaled.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to save the trained model.")
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
