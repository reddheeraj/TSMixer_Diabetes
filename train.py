import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtsmixer import TSMixer
from utils.OhioDataset import OhioDataset, prepare_global_data
from utils.opt import Options

def calculate_metrics(outputs, targets):
    mse = torch.mean((outputs - targets) ** 2).item()
    mape = torch.mean(torch.abs((outputs - targets) / (targets + 1e-8))).item() * 100  # Avoid division by zero
    return mse, mape

# Prepare Data
def prepare_data(config):
    global_training_set, train_dataset, test_dataset = prepare_global_data(
        data_dir=config.data_dir,
        patient_id=config.patient_id,
        example_len=config.example_len + config.prediction_len,
        unimodal=config.unimodal,
    )

    global_training_loader = DataLoader(global_training_set, batch_size=config.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader

# Train Function
def train(model, train_loader, criterion, optimizer, config, device):
    model.train()
    total_loss = 0

    for i, example in enumerate(train_loader):
        # Separate inputs and targets
        inputs = example[:, :-config.prediction_len, :]  # Inputs: all but last 6 time steps
        targets = example[:, -config.prediction_len:, :]  # Targets: last 6 time steps
        # print("Shape of inputs: ", inputs.shape)
        # print("Shape of targets: ", targets.shape)
        inputs, targets = inputs.to(device), targets.to(device)

        batch_size = inputs.shape[0]

        # Forward pass
        outputs = model(inputs)  # Ensure the model accepts this input shape
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

    return total_loss / len(train_loader.dataset)

# Test Function
def test(model, test_loader, criterion, config, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, example in enumerate(test_loader):
            inputs = example[:, :-config.prediction_len, :]  # Inputs: all but last 6 time steps
            targets = example[:, -config.prediction_len:, :]  # Targets: last 6 time steps
            inputs, targets = inputs.to(device), targets.to(device)

            batch_size = inputs.shape[0]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size

    return total_loss / len(test_loader.dataset)

# Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Options().parse()

    # Data preparation
    train_loader, test_loader = prepare_data(config)

    # Model
    model = TSMixer(
        sequence_length=config.example_len,
        prediction_length=config.prediction_len,
        input_channels=config.input_channels,
        output_channels=config.output_channels,
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    best_test_loss = float("inf")
    train_losses = []
    test_losses = []

    for epoch in range(config.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, config, device)
        test_loss = test(model, test_loader, criterion, config, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # # Save the model if it improves
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "config": config.__dict__,
                },
                f"models/best_model_patient{config.patient_id}_lr{config.lr}.pth",
            )
            print("Saved best model with losses!")

    print("Training complete.")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        total_mse, total_mape = 0, 0
        num_batches = 0
        for example in test_loader:
            inputs = example[:, :-config.prediction_len, :].to(device)
            targets = example[:, -config.prediction_len:, :].to(device)
            outputs = model(inputs)
            mse, mape = calculate_metrics(outputs, targets)
            total_mse += mse
            total_mape += mape
            num_batches += 1

        avg_mse = total_mse / num_batches
        avg_mape = total_mape / num_batches
        print(f"Test MSE: {avg_mse:.4f}, Test MAPE: {avg_mape:.2f}%")

if __name__ == "__main__":
    main()
