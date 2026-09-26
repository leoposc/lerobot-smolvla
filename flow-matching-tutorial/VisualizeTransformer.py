import re
import torch 
import matplotlib.pyplot as plt 
import ipywidgets as widgets
from IPython.display import display


def visualize_params(state_dict, initial_state_dict=None):
    """
    Interactive visualization of parameter statistics across Transformer layers.

    Parameters 
    ---------- 
    model: 
        Traned/current PyTorch model. 

    initial_state_dict: 
        State dict of the model before training.
        If provided, additional metrics such as relative parameter change become 
        availble. 

    Example usage before training: 

    inital_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    """

    layer_params = {}

    pattern = re.compile(
        r"^layers\.(\d+)\.(.+)$"
    )

    for name, tensor in state_dict:

        match = pattern.match(name)

        if match is None:
            continue

        layer_idx = int(match.group(1))
        param_name = match.group(2)

        # Only parameters, not buffers
        if not torch.is_tensor(tensor):
            continue

        layer_params.setdefault(layer_idx, {})
        layer_params[layer_idx][param_name] = tensor.detach().cpu().float()

    if not layer_params: 
        raise ValueError(
            "Could not automatically find Transformer layers. "
            "Check model.named_parameters()."
        )

    def categorize(name):
        if "self_attn.in_proj" in name:
            return "Attention QKV"
        if "self_attn.out_proj" in name: 
            return "Attention Output"
        if "linear1" in name: 
            return "FFN Input"
        if "linear2" in name: 
            return "FFN Output"
        if "norm1" in name: 
            return "LayerNorm 1"
        if "norm2" in name: 
            return "LayerNorm 2"
    
        return "Other"

    categories = [
        "Attention QKV",
        "Attention Output",
        "FFN Input", 
        "FFN Output",
        "LayerNorm 1",
        "LayerNorm 2",
        "Other",
    ]
    category_selector = widgets.SelectMultiple(
        options=categories,
        value=tuple(categories[:4]),
        description="Params:",
        layout=widgets.Layout(
            width="300px",
            height="150px"
        ),
    )

    metric_options = [
        "L2 norm",
        "Mean absolute value",
        "Standard deviation",
        "Relative change",
        "Absolute change",
    ]

    if initial_state_dict is None:
        metric_options = metric_options[:3]

    metric_selector = widgets.Dropdown(
        options=metric_options,
        value=metric_options[0],
        description="Metric:",
        layout=widgets.Layout(width="300px"),
    )

    log_scale = widgets.Checkbox(
        value=False,
        description="Log scale"

    )

    output = widgets.Output()

    # Calculate metrics 

    def calculate_metric(full_name, tensor, metric):

        tensor = tensor.detach().cpu().float()

        if metric == "L2 norm":
            return float(torch.linalg.vector_norm(tensor))
        elif metric == "Mean absolute value":
            return float(tensor.abs().mean())
        elif metric == "Standard deviation":
            return float(tensor.std())

        # Metrics below require initial_state_dict
        if initial_state_dict is None:
            return np.nan
        if full_name not in initial_state_dict:
            return np.nan

        initial = (
            initial_state_dict[full_name]
            .detach()
            .cpu()
            .float()
        )

        difference = tensor - initial

        if metric == "Absolute change":
            return float(
                torch.linalg.vector_norm(difference)
            )

        elif metric == "Relative change":

            epsilon = 1e-8

            return float(
                torch.linalg.vector_norm(difference)
                /
                (
                    torch.linalg.vector_norm(initial)
                    + epsilon
                )
            )
        return np.nan

    def update(*args):

        with output:
            output.clear_output(wait=True)

            selected = category_selector.value
            metric = metric_selector.value

            if not selected:
                print("Select at least one parameter group.")
                return

            matrix = []
            row_labels = []

            layer_indices = sorted(layer_params.keys())

            for selected_category in selected:
                values = []

                for layer_idx in layer_indices:

                    category_values = []

                    for param_name, tensor in layer_params[layer_idx].items():

                        # Check whether this parameter belongs
                        # to the selected category
                        if categorize(param_name) != selected_category:
                            continue

                        full_name = (
                            f"layers.{layer_idx}.{param_name}"
                        )

                        value = calculate_metric(
                            full_name,
                            tensor,
                            metric,
                        )

                        # Only add valid values
                        if value is not None and np.isfinite(value):
                            category_values.append(float(value))

                    # --------------------------------------------
                    # Important: handle missing parameters
                    # --------------------------------------------

                    if category_values:
                        values.append(
                            float(np.mean(category_values))
                        )
                    else:
                        values.append(np.nan)

                matrix.append(values)
                row_labels.append(selected_category)

            matrix = np.asarray(matrix, dtype=float)

            # --------------------------------------------
            # Plot
            # --------------------------------------------

            plot_matrix = matrix.copy()

            if log_scale.value:
                plot_matrix = np.log10(
                    np.maximum(
                        np.abs(plot_matrix),
                        1e-12,
                    )
                )

            fig, ax = plt.subplots(
                figsize=(
                    max(8, len(layer_indices) * 1.3),
                    max(4, len(row_labels) * 0.8),
                )
            )

            im = ax.imshow(
                plot_matrix,
                aspect="auto",
                interpolation="nearest",
            )

            ax.set_xticks(range(len(layer_indices)))
            ax.set_xticklabels(
                [f"Layer {i}" for i in layer_indices]
            )

            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)

            ax.set_xlabel("Transformer layer")
            ax.set_ylabel("Parameter group")

            ax.set_title(
                f"Transformer parameters — {metric}"
            )

            plt.colorbar(
                im,
                ax=ax,
                label=(
                    f"log10({metric})"
                    if log_scale.value
                    else metric
                ),
            )

            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):

                    value = matrix[row, col]

                    if np.isfinite(value):
                        ax.text(
                            col,
                            row,
                            f"{value:.3g}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=9,
                        )

            plt.tight_layout()
            plt.show()
        # end of update function

    category_selector.observe(update, names="value")
    metric_selector.observe(update, names="value")
    log_scale.observe(update, names="value")

    display(
        widgets.VBox([
            widgets.HBox([
                category_selector,
                widgets.VBox([
                    metric_selector,
                    log_scale,
                ]),
            ]),
            output,
        ])
    )
    
    update()

def show_all_digits(model, denoising_steps=50):
    digits = torch.arange(10, device=device)
    samples = generate(
        model, digits, steps=denoising_steps
    )

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))

    for i, ax in enumerate(axes):
        ax.imshow(
            samples[i, 0].cpu(),
            cmap="gray",
        )

        ax.set_title(str(i))
        ax.axis("off")
    plt.show()