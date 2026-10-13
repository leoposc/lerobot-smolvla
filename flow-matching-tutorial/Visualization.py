import matplotlib.pyplot as plt 


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