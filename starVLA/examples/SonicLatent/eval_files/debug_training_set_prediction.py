#!/usr/bin/env python
"""Compare a SonicLatent checkpoint against high-signal training samples.

This is an offline diagnostic tool. It answers two concrete questions:
1. On training-set observations, does the model reproduce forward motion tokens?
2. On training-set observations, does the model reproduce hand-closing actions?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework
from starVLA.training.trainer_utils.trainer_tools import resize_images


ACTION_BLOCKS = {
    "motion": slice(0, 64),
    "left_hand": slice(64, 71),
    "right_hand": slice(71, 78),
}


def denorm_action(normalized: np.ndarray, stats: dict) -> np.ndarray:
    action_min = np.asarray(stats["min"], dtype=np.float32)
    action_max = np.asarray(stats["max"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(action_min, dtype=bool)), dtype=bool)
    normalized = np.clip(normalized, -1.0, 1.0)
    denorm = 0.5 * (normalized + 1.0) * (action_max - action_min) + action_min
    return np.where(mask, denorm, normalized).astype(np.float32)


def fmt(name: str, value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    return (
        f"{name}: shape={value.shape}, min={value.min():.4f}, max={value.max():.4f}, "
        f"first={np.array2string(value[0], precision=3, suppress_small=True)}, "
        f"mid={np.array2string(value[len(value)//2], precision=3, suppress_small=True)}, "
        f"last={np.array2string(value[-1], precision=3, suppress_small=True)}"
    )


def cosine_by_timestep(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    numerator = np.sum(target * pred, axis=-1)
    denominator = np.linalg.norm(target, axis=-1) * np.linalg.norm(pred, axis=-1)
    return numerator / np.maximum(denominator, 1e-8)


def fmt_metrics(name: str, target: np.ndarray, pred: np.ndarray) -> str:
    target = np.asarray(target, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    diff = pred - target
    cosine = cosine_by_timestep(target, pred)
    key_indices = sorted(set([0, len(cosine) // 2, len(cosine) - 1]))
    cosine_samples = ", ".join(f"{idx}:{cosine[idx]:.3f}" for idx in key_indices)
    return (
        f"{name}: mae={np.mean(np.abs(diff)):.4f}, "
        f"rmse={np.sqrt(np.mean(diff ** 2)):.4f}, "
        f"max_abs_err={np.max(np.abs(diff)):.4f}, "
        f"cosine(first/mid/last)={cosine_samples}"
    )


def score_hand(action: np.ndarray) -> float:
    left = action[:, 64:71]
    right = action[:, 71:78]
    return float(max(np.max(np.abs(left)), np.max(np.abs(right))))


def score_motion(action: np.ndarray) -> float:
    return float(np.max(np.abs(action[:, :64])))


def candidate_key(selection: str, hand_score: float, motion_score: float) -> tuple[float, float]:
    if selection == "hand":
        return hand_score, motion_score
    if selection == "motion":
        return motion_score, hand_score
    return min(hand_score, motion_score), hand_score + motion_score


def parse_indices(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def predict_action_safe(model, sample: dict, device: str) -> dict:
    try:
        return model.predict_action([sample])
    except RuntimeError as exc:
        if device != "cpu" or "must have the same dtype" not in str(exc):
            raise
        print("CPU dtype mismatch in model.predict_action; retrying with explicit fp32 action head inputs.")
        return predict_action_cpu_fp32(model, [sample])


def predict_action_cpu_fp32(model, examples: list[dict]) -> dict:
    """CPU-only fallback for QwenGR00T checkpoints with bf16 VLM outputs."""
    batch_images = [to_pil_preserve(example["image"]) for example in examples]
    instructions = [example["lang"] for example in examples]
    state = [example["state"] for example in examples] if "state" in examples[0] else None

    train_obs_image_size = getattr(model.config.framework, "obs_image_size", None)
    if train_obs_image_size:
        batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images,
        instructions=instructions,
    )
    qwenvl_outputs = model.qwen_vl_interface(
        **qwen_inputs,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = qwenvl_outputs.hidden_states[-1].float()
    state_tensor = (
        torch.from_numpy(np.asarray(state)).to(last_hidden.device, dtype=torch.float32)
        if state is not None
        else None
    )
    with torch.autocast("cuda", enabled=False):
        pred_actions = model.action_model.predict_action(last_hidden, state_tensor)
    return {"normalized_actions": pred_actions.float().detach().cpu().numpy()}


def image_to_uint8(image) -> np.ndarray:
    if isinstance(image, (list, tuple)):
        image = image[0]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_debug_artifacts(
    output_dir: Path,
    index: int,
    sample: dict,
    target: np.ndarray,
    pred: np.ndarray,
    target_norm: np.ndarray,
    pred_norm: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"training_idx_{index:06d}"
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        target_raw=target,
        pred_raw=pred,
        target_norm=target_norm,
        pred_norm=pred_norm,
        state=np.asarray(sample.get("state"), dtype=np.float32),
        lang=np.asarray(sample.get("lang")),
    )
    image_arr = image_to_uint8(sample["image"])
    Image.fromarray(image_arr).save(output_dir / f"{stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yaml",
        default="examples/SonicLatent/train_files/train_sonic_latent.yaml",
    )
    parser.add_argument(
        "--ckpt-path",
        default="playground/Checkpoints/sonic_latent_v1/checkpoints/steps_100000_pytorch_model.pt",
    )
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--selection",
        choices=["both", "hand", "motion"],
        default="both",
        help="How to choose high-signal samples when --indices is not provided.",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated dataset indices to inspect, e.g. '0,123,456'. Overrides --selection.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for per-sample npz/png debug artifacts.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="test")

    selected_indices = parse_indices(args.indices)
    scan_indices = selected_indices if selected_indices else list(range(min(args.num_candidates, len(dataset))))

    candidates = []
    for i in scan_indices:
        sample = dataset[i]
        action = np.asarray(sample["action"], dtype=np.float32)
        candidates.append((score_hand(action), score_motion(action), i, sample))

    candidates.sort(reverse=True, key=lambda x: candidate_key(args.selection, x[0], x[1]))
    selected = candidates[: args.top_k]

    print(f"Selected training samples by selection={args.selection!r}:")
    for hand_score, motion_score, index, sample in selected:
        print(
            f"  idx={index} hand_abs_max={hand_score:.4f} "
            f"motion_abs_max={motion_score:.4f} lang={sample['lang']!r}"
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available in this shell; falling back to CPU. This will be slow for prediction.")
        device = "cpu"

    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(device).eval()
    if device == "cpu":
        # QwenGR00T predict_action uses CUDA autocast. On CPU the VLM may still
        # emit bf16 hidden/state tensors while the action head remains fp32.
        model = model.float()

    dataset_key = next(iter(model.norm_stats.keys()))
    action_stats = model.norm_stats[dataset_key]["action"]

    with torch.inference_mode():
        for hand_score, motion_score, index, sample in selected:
            output = predict_action_safe(model, sample, device)
            pred_norm = output["normalized_actions"][0]
            pred = denorm_action(pred_norm, action_stats)
            target_norm = np.asarray(sample["action"], dtype=np.float32)
            target = denorm_action(target_norm, action_stats)

            print("\n" + "=" * 80)
            print(
                f"idx={index} hand_abs_max={hand_score:.4f} "
                f"motion_abs_max={motion_score:.4f}"
            )
            print("Raw metrics:")
            for block_name, block_slice in ACTION_BLOCKS.items():
                print(fmt_metrics(block_name, target[:, block_slice], pred[:, block_slice]))
            print("Raw action scale:")
            print(fmt("target_motion", target[:, :64]))
            print(fmt("pred_motion", pred[:, :64]))
            print(fmt("target_left", target[:, 64:71]))
            print(fmt("pred_left", pred[:, 64:71]))
            print(fmt("target_right", target[:, 71:78]))
            print(fmt("pred_right", pred[:, 71:78]))
            print("Normalized metrics:")
            for block_name, block_slice in ACTION_BLOCKS.items():
                print(fmt_metrics(f"{block_name}_norm", target_norm[:, block_slice], pred_norm[:, block_slice]))
            print("Normalized action scale:")
            print(fmt("target_norm_left", target_norm[:, 64:71]))
            print(fmt("pred_norm_left", pred_norm[:, 64:71]))
            print(fmt("target_norm_right", target_norm[:, 71:78]))
            print(fmt("pred_norm_right", pred_norm[:, 71:78]))
            if args.output_dir:
                save_debug_artifacts(
                    output_dir=Path(args.output_dir),
                    index=index,
                    sample=sample,
                    target=target,
                    pred=pred,
                    target_norm=target_norm,
                    pred_norm=pred_norm,
                )


if __name__ == "__main__":
    main()
