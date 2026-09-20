import torch 
import numpy as np 
from torch import nn, Tensor 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import override 
import matplotlib.pyplot as plt 
from enum import Enum

import ipywidgets as widgets
from IPython.display import display
from pprint import pprint

class Mode(Enum):
    CNN = 1
    BasicTransformer = 2

    def to_string(self):
        return self.name

MODE = Mode.BasicTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# Version 1: CNN Architecture
# ================================================================
class FlowMatching(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.digit_embedding = nn.Embedding(10, 64)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.net = nn.Sequential(
            nn.Conv2d(1 + 64 + 64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )



    @override 
    def forward(self, x, t, digit):

        B, _, H, W = x.shape 
        
        digit_emb = self.digit_embedding(digit)

        # Embed digit: [B, 64] -> [B, 64, H, W]
        digit_emb = digit_emb[:, :, None, None]
        digit_emb = digit_emb.expand(-1, -1, H, W)

        # Embed time: [B] -> [B, 1] -> [B, 64]
        time_emb = self.time_embedding(t[:, None])

        # [B, 64] -> [B, 64, H, W]
        time_emb = time_emb[:, :, None, None].expand(-1, -1, H, W)

        # concatenate everything 
        x = torch.cat([x, digit_emb, time_emb], dim=1)

        return self.net(x)


# ================================================================
# version 2: transformer architecture
# ================================================================
class FlowMatching_with_Transformer(nn.Module):
    def __init__(self):
        super().__init__() 

        self.embedding_dim=64

        self.digit_embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=self.embedding_dim
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=self.embedding_dim,
            kernel_size=4,
            stride=4,
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 49, self.embedding_dim)
        )

        self.num_layers = 4
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])

        self.output_proj = nn.Linear(
            self.embedding_dim, 
            4 * 4,
        )



    @override
    def forward(self, x, t, digit):

        digit_emb = self.digit_embedding(digit)
        digit_emb = digit_emb.unsqueeze(1)

        time_emb = self.time_embedding(t.unsqueeze(1)).unsqueeze(1)

        patch = self.patch_embedding(x)
        patch = patch.flatten(2)
        patch = patch.transpose(1, 2)

        image_tokens = patch + self.pos_embedding

        x = torch.cat([
            digit_emb, 
            time_emb,
            image_tokens,
            ], dim=1
        )

        for layer in self.layers:
            x = layer(x)


        # remove digit and time tokens again 
        image_tokens = x[:, 2:, :]

        # convert each token back into a 4x4 patch 
        patches = self.output_proj(image_tokens)
        patches = patches.transpose(1, 2) # back to [B, 16, 49]

        # reconstruct the velocity image 
        velocity = F.fold(
            patches, 
            output_size=(28, 28),
            kernel_size=4,
            stride=4,
        ) # [B, 1, 28, 28]

        return velocity

# ================================================================
# Training
# ================================================================

def create_loader(training_size=1024, batch_size=128) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="./data", 
        train=True,
        download=True,
        transform=transform,
    )

    indices = torch.randperm(len(dataset))[:training_size]
    dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def train(model, loader, training_epochs=20):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    for epoch in range(training_epochs):

        total_loss = 0 

        for x1, digit in loader:

            x1 = x1.to(device)
            digit = digit.to(device)

            B = x1.shape[0]
            x0 = torch.randn_like(x1) # gausssian source distribution

            t = torch.rand(B, device=device)

            t_img = t[:, None, None, None]
            xt = ((1 - t_img) * x0 + t_img * x1)

            target_velocity = x1 - x0
            predicted_velocity = model(xt, t, digit)

            loss = F.mse_loss(
                predicted_velocity,
                target_velocity,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            total_loss += loss.item() 

        print(
            f"Epoch {epoch + 1}: "
            f"loss = {total_loss / len(loader):.4f}"
        )

@torch.no_grad()
def generate(model, digits, steps=10):

    model.eval() 

    B = len(digits)
    x = torch.randn(
        B, 1, 28, 28,
        device=device,
    )

    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full(
            (B,),
            i / steps,
            device=device
        ) # Creates tensor of batch_size with the current time step

        velocity = model(
            x,
            t,
            digits,
        )

        x += dt * velocity

    return x 

def show_all_digits(model):
    digits = torch.arange(10, device=device)
    samples = generate(
        model, digits, steps=50
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

def save_model(model) -> None:
    filename = f"{MODE.to_string()}.pt"
    model.to("cpu")
    model.eval()
    torch.save(model.state_dict(), filename)

def load_model(model):
    filename = f"{MODE.to_string()}.pt"
    model.load_state_dict(torch.load(filename, map_location=device))
    return model

def visualize_params(model, initial_state_dict=None):
    """
    Interactive visualization of parameter statistics across Transformer layers.

    Parameters 
    ---------- 
    model: 
        Traned/current PyTorch model. 

    initial_state_dict: 
        State dict of the model before training.
        If provided, additional metrics such as relative parameter change become 
        available. 

    Example usage before training: 

    initial_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    """

    layer_params = {} 

    for name, param in model.named_parameters():
        # Try to identify the Transformer encoder layers 

        
        if ".layers" not in name: 
            continue

        parts = name.split(".layers.") 

        if len(parts) != 2:
            continue 

        layer_part, param_name = parts 

        try: 
            layer_idx, param_name = param_name.split(".", 1)
            layer_idx = int(layer_idx)
        except ValueError:
            continue 

        layer_params.setdefault(layer_idx, {})
        layer_params[layer_idx][param_name] = param 

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
            return "FFN Outpu"
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
        "L2 Norm",
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

    def calculate_metric(param, layer_idx, param_name, metric):
        tensor = param.detach().cpu().float() 

        if metric == "L2 norm":
            return torch.linalg.vector_norm(tensor).item()
        if metric == "Mean absolute value":
            return tensor.abs().mean().item()
        if metric == "Standard deviation":
            return tensor.std().item()
        if "Change" in metric:

            if initial_state_dict is None:
                return np.nan

            full_name = None 

            for name, p in model.named_parameters():
                if p is param:
                    full_name = name
                    break 

            if full_name not in initial_state_dict:
                return np.nan 

            initial = initial_state_dict[full_name].float() 

            if "Absolute" in metric:
                return torch.lingalg.vector_norm(
                    tensor - initial
                ).item()
            else:
                epsilon = 1e-8 
                return (
                    torch.linalg.vector_norm(tensor - initial)
                    / 
                    (torch.linalg.vector_norm(initial) + epsilon)
                ).item()

    def update(*args):

        with output: 
            output.clear_output(wait=True)

            selected = category_selector.value 
            metric = metric_selector.value 

            if not selected:
                print("Select at least one parameter category.")
                return 

            layer_indices = sorted(layer_params.keys())

            matrix = []
            labels = []

            for category in selected:
                values = [] 

                for layer_idx in layer_indices:
                    matching_params = [
                        (name, param) for name, param in layer_params[layer_idx].items()
                            if categorize(name) == category
                    ]

                    if not matching_params: 
                        values.append(np.nan)
                        continue 

                    # usually there is one tensor per category/ layer. If there are several, aggregate them 
                    category_values = []
                
                    for name, param in matching_params:
                        value = calculate_metric(
                            param, layer_idx, name, metric,
                        )
                        category_values.append(value)

                    values.append(np.nanmean(category_values))

                matrix.append(values)
                labels.append(category)

            matrix = np.array(matrix)

            fig, ax = plt.subplots(
                figsize=(
                    max(8, len(layer_indices) * 1.3),
                    max(3, len(labels) * 0.8),
                )
            )

            plot_matrix = matrix.copy()

            if log_scale: 
                plot_matrix = np.log10(
                    np.maximum(np.abs(plot_matrix), 1e-12)
                )

            im = ax.imshow(
                plot_matrix,
                aspect="auto",
                interpolation="nearest",
            )

            ax.set_xticks(range(len_layer_indices))
            ax.set_xticklabels(
                [f"Layer {i}" for i in layer_indices]
            )
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Transformer layer")
            ax.set_ylabel("Parameter group")
            ax.set_title(f"Transformer parameters - {metric}")

            plt.colorbar(
                im, ax=ax, label=(
                    "log10(value)" if log_scale else metric
                ),
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




            



def main():

    match MODE:
        case Mode.CNN:
            model = FlowMatching().to(device)
        case Mode.BasicTransformer:
            model = FlowMatching_with_Transformer().to(device)
    
    initial_state = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }

    # loader = create_loader(training_size=10, batch_size=10)
    # train(model, loader)
    # show_all_digits(model)
    #
    # save_model(model)

    model = load_model(model)
    show_all_digits(model)

    visualize_params(model, initial_state_dict=initial_state)


if __name__ == "__main__":
    main()
