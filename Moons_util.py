


@torch.no_grad()
def plot_velocity_field(flow, times=(0.0, 0.25, 0.5, 0.75, 1.0)):
    import torch 
    import matplotlib.pyplot as plt
    print("aölksdjfölaksdjfö")

    flow.eval()

    # Create a grid of 2D states
    resolution = 25
    coords = torch.linspace(-3, 3, resolution)
    xx, yy = torch.meshgrid(coords, coords, indexing="xy")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    fig, axes = plt.subplots(
        1, len(times), figsize=(5 * len(times), 5),
        sharex=True, sharey=True
    )

    for ax, t in zip(axes, times):
        t_batch = torch.full((len(grid), 1), t)
        velocity = flow(grid, t_batch)

        u = velocity[:, 0].reshape(resolution, resolution)
        v = velocity[:, 1].reshape(resolution, resolution)
        speed = torch.sqrt(u**2 + v**2)

        ax.quiver(
            xx, yy, u, v, speed,
            cmap="viridis",
            angles="xy",
        )
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x")
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    axes[0].set_ylabel("y")
    plt.tight_layout()
    plt.show()
