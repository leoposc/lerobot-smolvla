"""Inference smoke test for CustomSmolVLA: feeds dataset frames through the full
preprocessor -> policy.select_action -> postprocessor chain and prints the tensor shapes
at every stage of the model. No simulator and no trained checkpoint required.

    python -m CustomSmolVLA.smoke_test                                  # random-init policy
    python -m CustomSmolVLA.smoke_test --checkpoint outputs/train/custom_smolvla/checkpoints/last/pretrained_model
    python -m CustomSmolVLA.smoke_test --device cpu --frames 3 --no-trace
    python -m CustomSmolVLA.smoke_test --policy smolvla                 # lerobot's shipped SmolVLA
"""

import argparse
import functools
import time

import torch
from torch import Tensor

import CustomSmolVLA  # noqa: F401  registers the "custom_smolvla" config type
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_policy, make_policy_config, make_pre_post_processors


def describe(x, max_items: int = 8):
    """Compact, recursive shape/dtype description of tensors and containers."""
    if isinstance(x, Tensor):
        return f"Tensor{tuple(x.shape)} {str(x.dtype).removeprefix('torch.')} {x.device.type}"
    if isinstance(x, dict):
        return {k: describe(v) for k, v in list(x.items())[:max_items]}
    if isinstance(x, (list, tuple)):
        if all(isinstance(v, str) for v in x):
            return f"{type(x).__name__}[str] x{len(x)}"
        return type(x)([describe(v) for v in x[:max_items]])
    if isinstance(x, str):
        return repr(x[:60])
    return type(x).__name__


def trace(obj, method_names, max_prints: int = 10):
    """Wrap methods on `obj` so their inputs/outputs are printed (first `max_prints` calls)."""
    counts = {}
    for name in method_names:
        fn = getattr(obj, name)

        @functools.wraps(fn)
        def wrapper(*args, __fn=fn, __name=name, **kwargs):
            counts[__name] = counts.get(__name, 0) + 1
            n = counts[__name]
            if n <= max_prints:
                print(f"\n>> {type(obj).__name__}.{__name}  (call {n})")
                for i, a in enumerate(args):
                    print(f"   arg{i}: {describe(a)}")
                for k, v in kwargs.items():
                    print(f"   {k}: {describe(v)}")
            out = __fn(*args, **kwargs)
            if n <= max_prints:
                print(f"<< {__name} -> {describe(out)}")
            elif n == max_prints + 1:
                print(f"   ({__name} called again, further calls not printed)")
            return out

        setattr(obj, name, wrapper)
    return counts


def to_batch(frame: dict, device) -> dict:
    """Single dataset frame -> batch of 1 (what a DataLoader / env step would produce)."""
    batch = {}
    for k, v in frame.items():
        if isinstance(v, Tensor):
            batch[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="custom_smolvla", choices=["custom_smolvla", "smolvla"])
    p.add_argument("--repo-id", default="lerobot/aloha_mobile_cabinet")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, default=3, help="number of consecutive frames to run")
    p.add_argument("--checkpoint", default=None, help="path to .../pretrained_model (default: random init)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-trace", action="store_true")
    args = p.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id)
    dataset = LeRobotDataset(args.repo_id, episodes=[args.episode])
    print(f"dataset: {args.repo_id}  episode {args.episode}  ({dataset.num_frames} frames, {meta.fps} fps)")

    if args.checkpoint:
        policy = get_policy_class(args.policy).from_pretrained(args.checkpoint)
        policy.config.device = args.device
        policy.to(args.device)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=args.checkpoint,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
    else:
        cfg = make_policy_config(args.policy, device=args.device)
        policy = make_policy(cfg, ds_meta=meta)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    policy.eval()
    cfg = policy.config

    print(f"\n=== policy config ({type(policy).__module__}) ===")
    print(f"input features : { {k: tuple(f.shape) for k, f in cfg.input_features.items()} }")
    print(f"output features: { {k: tuple(f.shape) for k, f in cfg.output_features.items()} }")
    print(f"chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps} num_steps={cfg.num_steps}")
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"params: {n_total / 1e6:.1f}M total, {n_train / 1e6:.1f}M trainable")

    if not args.no_trace:
        # Only trace the methods the chosen implementation actually has.
        trace(policy, ["prepare_images", "prepare_state", "_get_action_chunk"])
        model_methods = ["embed_prefix", "embed_suffix", "forward_layers", "sample_actions", "euler_integration",
                         "denoise_step", "grouped_query_attention", "eager_attention_forward"]
        trace(policy.model, [m for m in model_methods if hasattr(policy.model, m)])

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    for i in range(min(args.frames, dataset.num_frames)):
        frame = dataset[i]
        batch = to_batch(frame, args.device)
        gt_action = batch.pop("action")

        print(f"\n{'=' * 20} frame {i} {'=' * 20}")
        if i == 0:
            print(f"raw batch: {describe({k: v for k, v in batch.items()}, max_items=50)}")

        obs = preprocessor(batch)
        if i == 0:
            print(f"\npreprocessed: {describe(obs, max_items=50)}")

        t0 = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(obs)
        dt = time.perf_counter() - t0
        action = postprocessor(action)

        err = (action.cpu().float() - gt_action.float()).abs().mean().item()
        print(f"\naction {describe(action)}  |a - gt|_mean={err:.4f}  ({dt * 1000:.0f} ms, "
              f"{len(policy._queues['action'])} actions left in queue)")

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("\nsmoke test OK")


if __name__ == "__main__":
    main()
