import torch
import matplotlib.pyplot as plt
from torchtsmixer import TSMixer
from utils.opt import Options
import warnings
import os
from train import prepare_data

# Suppress the FutureWarning for torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# Set environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_and_plot(saved_model_path):
    # Load saved model and losses
    checkpoint = torch.load(saved_model_path)
    train_losses = checkpoint["train_losses"]
    test_losses = checkpoint["test_losses"]
    config = checkpoint["config"]

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend()
    plt.grid()
    plt.show()
    # print("config is", config)
    # plt.savefig(f"losses_{config['patient_id']}.png")

    print(f"Model and losses loaded successfully from {saved_model_path}!")

import matplotlib.pyplot as plt
import torch

def plot_target_distribution(data_loader, title="Target Value Distribution"):
    """
    Plots the distribution of target values from a data loader.

    Parameters:
        data_loader: torch.utils.data.DataLoader
            The data loader containing your dataset.
        title: str
            Title for the plot.
    """
    target_values = []

    # Collect all target values
    for example in data_loader:
        # Assuming the last feature in the dataset is the target
        targets = example[:, -1, 0]  # Adjust indexing based on your data shape
        target_values.extend(targets.numpy())

    # Convert to numpy array for easier plotting
    target_values = torch.tensor(target_values).numpy()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(target_values, bins=50, color="blue", alpha=0.7, edgecolor="black")
    plt.title(title, fontsize=16)
    plt.xlabel("Target Values", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def plot_actual_vs_predicted(saved_model_path, data_loader, config):
    """
    Plots actual vs. predicted values using a trained model.

    Parameters:
        saved_model_path: str
            Path to the saved model file.
        data_loader: torch.utils.data.DataLoader
            DataLoader for the test set.
        config: Namespace
            Configuration object containing model and dataset details.
    """
    # Load the model
    model = TSMixer(
        sequence_length=config.example_len,
        prediction_length=config.prediction_len,
        input_channels=config.input_channels,
        output_channels=config.output_channels,
    )
    model.load_state_dict(torch.load(saved_model_path)["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    actuals = []
    predictions = []

    with torch.no_grad():
        for example in data_loader:
            inputs = example[:, :-config.prediction_len, :].to(device)  # Input sequence
            targets = example[:, -config.prediction_len:, 0].cpu().numpy()  # Actual targets
            outputs = model(inputs).cpu().numpy()  # Model predictions

            actuals.extend(targets.flatten())  # Flatten to 1D
            predictions.extend(outputs.flatten())  # Flatten to 1D

    # Plot actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual", color="blue", alpha=0.7)
    plt.plot(predictions, label="Predicted", color="orange", alpha=0.7)
    plt.title("Actual vs Predicted Values", fontsize=16)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

if __name__ == "__main__":
    config = Options().parse()
    saved_model_path = f"models/best_model_patient{config.patient_id}_lr{config.lr}.pth"  # Update with actual path
    load_and_plot(saved_model_path)

    train_loader, test_loader = prepare_data(Options().parse())
    # plot_target_distribution(train_loader, title="Train Target Value Distribution")
    # plot_target_distribution(test_loader, title="Test Target Value Distribution")

    plot_actual_vs_predicted(saved_model_path, test_loader, config)
