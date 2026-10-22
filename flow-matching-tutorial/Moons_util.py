
import torch 
import matplotlib.pyplot as plt 


@torch.no_grad()
def plot_velocity_field(flow, times=(0.0, 0.25, 0.5, 0.75, 1.0)):
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




@torch.no_grad()
def plot_trajectories(flow, n_samples=100, n_steps=50):
    flow.eval()

    x = torch.randn(n_samples, 2)
    times = torch.linspace(0, 1, n_steps + 1)

    trajectory = [x.clone()]

    for i in range(n_steps):
        x = flow.step(x, times[i], times[i + 1])
        trajectory.append(x.clone())

    trajectory = torch.stack(trajectory).numpy()

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(n_samples):
        ax.plot(
            trajectory[:, i, 0],
            trajectory[:, i, 1],
            alpha=0.35,
            linewidth=0.8,
        )

    ax.scatter(
        trajectory[0, :, 0], trajectory[0, :, 1],
        s=12, label="Initial noise"
    )
    ax.scatter(
        trajectory[-1, :, 0], trajectory[-1, :, 1],
        s=12, label="Final samples"
    )

    ax.set_title("Flow Matching trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend()
    plt.show()
