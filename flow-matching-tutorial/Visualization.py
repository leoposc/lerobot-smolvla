import matplotlib.pyplot as plt 
import torch
import numpy as np 

# from FlowMatching import generate


def plot_training_distribution(loader):
    """
    Plot the distribution of training data from the provided DataLoader as histogram for each digit. 
    
    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing training data.
    """

    digit_counts = [0] * 10
    for _, labels in loader:
        for label in labels:
            digit_counts[label.item()] += 1


    plt.bar(range(10), digit_counts)
    plt.title("Training Data Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.show()

def plot_loss(loss_list):
    """
    Plot the training loss over epochs.
    
    Args:
        loss_list (list): List of loss values for each epoch.
    """
    plt.plot(loss_list)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_losses(losses: np.ndarray, model_names: list[str]):
    """
    Plot training losses for multiple models.

    Args:
        losses: 2D array of shape (num_models, num_epochs).
        model_names: Names of the models, one per row in losses.
    """
    losses = np.asarray(losses)

    if losses.ndim != 2:
        raise ValueError("losses must be a 2D array.")
    if len(model_names) != losses.shape[0]:
        raise ValueError("Number of model names must match the number of rows.")

    for loss, name in zip(losses, model_names):
        plt.plot(loss, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def show_all_digits(model, generator_fn, config):
    digits = torch.arange(10, device=config["device"])
    config["print_generation_process"] = False
    samples = generator_fn(
        model, digits, config=config
    )
    config["print_generation_process"] = True


    fig, axes = plt.subplots(1, 10, figsize=(15, 2))

    for i, ax in enumerate(axes):
        ax.imshow(
            samples[i, 0].cpu(),
            cmap="gray",
        )

        ax.set_title(str(i))
        ax.axis("off")
    plt.show()
