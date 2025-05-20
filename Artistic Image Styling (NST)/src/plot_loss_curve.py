import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_loss_curves(results: dict[str, list[float]]) -> None:
    """
    Plots the style loss and total loss curves over epochs based on the provided results dictionary.

    Args:
        results (dict[str, list[float]]): A dictionary containing the loss values.
            Expected keys are 'style_loss' and 'total_loss', with corresponding lists of loss values for each epoch.

    Returns:
        None

    Example:
        results = {
            'style_loss': [0.8, 0.6, 0.4, 0.2],
            'total_loss': [1.0, 0.7, 0.5, 0.3]
        }
        plot_loss_curves(results)
    """
    style_loss = results['style_loss']
    total_loss = results['total_loss']

    epochs = range(len(style_loss))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, style_loss, label='Style Loss')
    plt.plot(epochs, total_loss, label='Total Loss')
    plt.title('Loss: Style vs Total')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


def plot_loss_curve_grid(results: dict[str, list[float]]) -> None:
    """
    Plots the content loss, style loss, and total loss curves over epochs using a grid layout.

    Args:
        results (dict[str, list[float]]): A dictionary containing the loss values.
            Expected keys are 'content_loss', 'style_loss', and 'total_loss', with corresponding lists of loss values for each epoch.

    Returns:
        None

    Example:
        results = {
            'content_loss': [1.0, 0.8, 0.6, 0.4],
            'style_loss': [0.9, 0.7, 0.5, 0.3],
            'total_loss': [1.9, 1.5, 1.1, 0.7]
        }
        plot_loss_curve_grid(results)
    """
    content_loss = results['content_loss']
    style_loss = results['style_loss']
    total_loss = results['total_loss']

    epochs = range(len(content_loss))

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, content_loss, marker='o', color='b', label='Content Loss')
    ax1.set_title('Content Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, style_loss, marker='s', color='r', label='Style Loss')
    ax2.set_title('Style Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, total_loss, marker='o', color='g', label='Total Loss')
    ax3.set_title('Total Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()