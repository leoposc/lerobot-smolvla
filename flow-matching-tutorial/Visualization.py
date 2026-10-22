import matplotlib.pyplot as plt 
import torch

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